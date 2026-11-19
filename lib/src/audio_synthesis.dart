/// Text-to-speech with the LFM3-Audio and LFM3-Omni bundles that
/// [AudioGgufSpec] and [OmniGgufSpec] install.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'audio_generation.dart';
import 'audio_gguf.dart';
import 'local_model.dart' show InferenceBackend;
import 'model.dart';
import 'model_loading.dart';
import 'model_store.dart' show AudioGgufSpec, OmniGgufSpec;
import 'omni_gguf.dart' show speechMmprojIn;
import 'tokenizer.dart';

const _languages = {
  'ar': 'Arabic',
  'da': 'Danish',
  'de': 'German',
  'el': 'Greek',
  'en': 'American English',
  'en-us': 'American English',
  'en-gb': 'British English',
  'fi': 'Finnish',
  'fr': 'French',
  'he': 'Hebrew',
  'it': 'Italian',
  'ja': 'Japanese',
  'ko': 'Korean',
  'nl': 'Dutch',
  'pl': 'Polish',
  'pt': 'Portuguese',
  'es': 'Spanish',
  'sv': 'Swedish',
  'tr': 'Turkish',
  'zh': 'Chinese',
};

/// The official single-text TTS prompt for [text].
///
/// Speech is conditioned by [language] and [voice] (`female` or `male`), not
/// by a reference speaker recording. [language] is a code such as `en`,
/// `en-gb` or `ja`, or a language name the model is asked to speak. The
/// prompt ends at `<|audio_start|>`, where speech frames begin. Throws
/// [ArgumentError] for empty text, text over 16000 characters or holding
/// `<|`, an invalid language or another voice.
String renderAudioTtsPrompt(
  String text, {
  String language = 'en',
  String voice = 'female',
  Lfm3SpeechVariant variant = Lfm3SpeechVariant.audio,
}) {
  if (text.trim().isEmpty ||
      text.length > 16000 ||
      text.contains('<|') ||
      !RegExp(r'^[A-Za-z][A-Za-z -]{0,63}$').hasMatch(language) ||
      !['female', 'male'].contains(voice)) {
    throw ArgumentError('invalid TTS text, language or voice');
  }
  final name = _languages[language.toLowerCase()] ?? language;
  return '<|startoftext|><|im_start|>system\nPerform TTS in $name using a $voice voice.<|im_end|>\n'
      '<|im_start|>user\n$text<|im_end|>\n<|im_start|>assistant\n'
      '${variant.assistantOpen}<|audio_start|>';
}

/// Thrown by [AudioSynthesizer.synthesize] after [AudioSynthesizer.cancel].
final class AudioSynthesisCancelled implements Exception {
  const AudioSynthesisCancelled();
  @override
  String toString() => 'Audio synthesis cancelled';
}

/// Synthesized speech: mono float PCM in -1..1.
final class AudioSynthesisResult {
  AudioSynthesisResult(
    this.samples,
    this.sampleRate,
    this.frames,
    this.stopReason,
  );

  /// The waveform, [sampleRate] samples per second.
  final Float32List samples;

  /// Samples per second (24000 for LFM3) and generated speech frames.
  final int sampleRate, frames;

  /// `eos` or `length`. A length-limited result is partial speech.
  final String stopReason;

  /// The speech's duration.
  double get seconds => samples.length / sampleRate;
}

/// Speaks text with an LFM3-Audio or LFM3-Omni model.
///
/// ```dart
/// final synth = await AudioSynthesizer.load(
///   store.bundlePath(AudioGgufSpec.lfm3AudioQ4),
/// );
/// final speech = await synth.synthesize('Hello!', voice: 'male');
/// File('hello.wav').writeAsBytesSync(
///   encodeSpeechWav(speech.samples, sampleRate: speech.sampleRate),
/// );
/// await synth.close();
/// ```
///
/// Each frame's sampling and codec decoding is synchronous CPU work, so an
/// app should synthesize in a worker isolate. One synthesis runs at a time.
final class AudioSynthesizer {
  /// A synthesizer over a caller-owned [model], [tokenizer] and [generator],
  /// which it uses exclusively while it synthesizes and never closes. Apps
  /// use [load]; this is for `package:dianoia/lab.dart`, where the language
  /// model is shared with transcription or conversation.
  AudioSynthesizer(this.model, this.tokenizer, this.generator)
    : backend = null,
      _owned = false;

  AudioSynthesizer._owned(
    this.model,
    this.tokenizer,
    this.generator,
    this.backend,
  ) : _owned = true;

  /// Loads the speech model installed in [dir]: an [AudioGgufSpec] or
  /// [OmniGgufSpec] bundle, whose `language/` holds the imported language
  /// model and whose mmproj holds the speech generator and codec.
  ///
  /// [backend], [contextLength] and [threads] are as for [LocalModel.load];
  /// a prompt and its frames must fit in [contextLength]. With [int8Inputs]
  /// the speech generator's quantized products take int8 inputs, about 1.4
  /// times faster than float inputs and slightly different speech; it is
  /// what keeps an iPhone 15 Pro Max at real time.
  static Future<AudioSynthesizer> load(
    String dir, {
    InferenceBackend backend = InferenceBackend.auto,
    int contextLength = 2048,
    int threads = 4,
    bool int8Inputs = true,
  }) async {
    final language = '$dir/language', mmproj = speechMmprojIn(dir);
    bundleWeights(language); // fails before anything is loaded
    final tokenizer = Lfm3Tokenizer.load(language);
    ensureNativeKernels(threads);
    final generator = Lfm3AudioGenerator.load(mmproj, int8Inputs: int8Inputs);
    try {
      final (model, used) = await openLanguageBundle(
        language,
        backend: backend,
        contextLength: contextLength,
      );
      return AudioSynthesizer._owned(model, tokenizer, generator, used);
    } catch (_) {
      generator.close();
      rethrow;
    }
  }

  /// The language model, its tokenizer and the speech generator.
  final Lfm3Model model;
  final Lfm3Tokenizer tokenizer;
  final Lfm3AudioGenerator generator;

  /// Where a [load]ed synthesizer's language model runs:
  /// [InferenceBackend.gpu] or [InferenceBackend.cpu]. Null for one made
  /// from caller-owned parts.
  final InferenceBackend? backend;

  final bool _owned;
  bool _busy = false, _cancelled = false, _closed = false;

  /// Whether [close] has run.
  bool get isClosed => _closed;

  /// Stops the running synthesis at its next frame: its
  /// [synthesize] throws [AudioSynthesisCancelled].
  void cancel() {
    _cancelled = true;
  }

  /// Cancels any running synthesis and, for a [load]ed synthesizer, frees
  /// the language model and the speech generator. Parts passed to the
  /// constructor stay open. Closing twice does nothing.
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    cancel();
    if (_owned) {
      generator.close();
      await model.close();
    }
  }

  void _check() {
    if (_cancelled) throw const AudioSynthesisCancelled();
  }

  /// Speaks [text] in [language] with a `female` or `male` [voice] (see
  /// [renderAudioTtsPrompt]), at most [maxFrames] frames of 80 ms (1..750).
  ///
  /// Sampling uses the model's defaults unless [temperature], [topK] or
  /// [repetitionPenalty] override them; [seed] makes it repeatable.
  /// [onAudio] receives each new piece of the waveform as it is decoded and
  /// [onCodes] each frame's codes. Throws [StateError] while another
  /// synthesis runs or after [close], [ArgumentError] for invalid options
  /// or a prompt that does not fit the context, and
  /// [AudioSynthesisCancelled] after [cancel].
  Future<AudioSynthesisResult> synthesize(
    String text, {
    String language = 'en',
    String voice = 'female',
    int maxFrames = 375,
    int seed = 0,
    double? temperature,
    int? topK,
    double? repetitionPenalty,
    void Function(Float32List samples)? onAudio,
    void Function(List<int> codes)? onCodes,
  }) async {
    if (_closed) throw StateError('Audio synthesizer is closed');
    if (_busy) throw StateError('Synthesis is already running');
    final prompt = tokenizer.encode(
      renderAudioTtsPrompt(
        text,
        language: language,
        voice: voice,
        variant: generator.variant,
      ),
    );
    final start = tokenizer.addedTokenId('<|audio_start|>'),
        output = tokenizer.addedTokenId('<|audio_out|>');
    if (start == null ||
        output == null ||
        prompt.last != start ||
        prompt.any((id) => id < 0 || id >= model.config.vocabSize) ||
        output >= model.config.vocabSize ||
        model.config.hiddenSize != generator.hiddenSize ||
        maxFrames < 1 ||
        maxFrames > 750 ||
        prompt.length + maxFrames > model.maxSeq ||
        (temperature != null && (!temperature.isFinite || temperature < 0)) ||
        (topK != null && topK < 1) ||
        (repetitionPenalty != null &&
            (!repetitionPenalty.isFinite || repetitionPenalty < 1))) {
      throw ArgumentError(
        'TTS model, tokenizer, context or sampling options differ',
      );
    }
    final codec = generator.startCodecStream();
    _busy = true;
    _cancelled = false;
    final chunks = <Float32List>[], history = <List<int>>[];
    final random = math.Random(seed);
    var stop = 'length', frames = 0;
    void emit(Float32List samples) {
      if (samples.isEmpty) return;
      chunks.add(samples);
      onAudio?.call(Float32List.fromList(samples));
    }

    try {
      await model.reset();
      for (var at = 0; at < prompt.length; at += model.maxChunk) {
        _check();
        final end = math.min(at + model.maxChunk, prompt.length);
        await model.forward(
          prompt.sublist(at, end),
          computeLogits: end == prompt.length,
        );
        await Future<void>.delayed(Duration.zero);
        _check();
      }
      for (var i = 0; i < maxFrames; i++) {
        _check();
        final codes = generator.sampleFrame(
          model.lastHiddenState,
          random,
          temperature: temperature,
          topK: topK,
          repetitionPenalty: repetitionPenalty,
          history: history,
        );
        if (codes.contains(generator.eosCode)) {
          stop = 'eos';
          break;
        }
        history.add(codes);
        frames++;
        onCodes?.call(List.unmodifiable(codes));
        _check();
        if (i + 1 < maxFrames) {
          // The codec decodes this frame on the CPU while the GPU feeds it
          // back to the language model.
          await model.forward(
            [output],
            embeddingOverrides: {0: generator.feedback(codes)},
            meanwhile: () => emit(codec.add(codes)),
          );
        } else {
          emit(codec.add(codes));
        }
        await Future<void>.delayed(Duration.zero);
        _check();
      }
      emit(codec.finish());
      _check();
      final samples = Float32List(chunks.fold(0, (n, c) => n + c.length));
      var offset = 0;
      for (final chunk in chunks) {
        samples.setAll(offset, chunk);
        offset += chunk.length;
      }
      if (samples.length != frames * generator.frameSamples) {
        throw StateError('codec output length mismatch');
      }
      return AudioSynthesisResult(samples, generator.sampleRate, frames, stop);
    } finally {
      codec.close();
      _busy = false;
    }
  }
}

/// [samples] as a mono 16-bit PCM WAV file at [sampleRate], clipped to
/// full scale. Throws [ArgumentError] for a non-finite sample.
Uint8List encodeSpeechWav(Float32List samples, {int sampleRate = 24000}) {
  if (sampleRate < 1 ||
      sampleRate > 192000 ||
      samples.any((s) => !s.isFinite) ||
      samples.length > 0x7fffff00) {
    throw ArgumentError('invalid WAV samples');
  }
  final bytes = Uint8List(44 + samples.length * 2),
      data = ByteData.sublistView(bytes);
  void tag(int at, String value) {
    bytes.setAll(at, value.codeUnits);
  }

  tag(0, 'RIFF');
  data.setUint32(4, bytes.length - 8, Endian.little);
  tag(8, 'WAVE');
  tag(12, 'fmt ');
  data.setUint32(16, 16, Endian.little);
  data.setUint16(20, 1, Endian.little);
  data.setUint16(22, 1, Endian.little);
  data.setUint32(24, sampleRate, Endian.little);
  data.setUint32(28, sampleRate * 2, Endian.little);
  data.setUint16(32, 2, Endian.little);
  data.setUint16(34, 16, Endian.little);
  tag(36, 'data');
  data.setUint32(40, samples.length * 2, Endian.little);
  for (var i = 0; i < samples.length; i++) {
    data.setInt16(
      44 + i * 2,
      (samples[i].clamp(-1.0, 1.0) * 32768).round().clamp(-32768, 32767),
      Endian.little,
    );
  }
  return bytes;
}
