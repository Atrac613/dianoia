/// Text-to-speech orchestration for the prepared official LFM3-Audio Q4 bundle.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'audio_generation.dart';
import 'audio_gguf.dart';
import 'model.dart';
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

/// The official single-text TTS prompt. Speech output is conditioned by language
/// and the model's female/male instruction, not by a reference speaker recording.
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

final class AudioSynthesisCancelled implements Exception {
  const AudioSynthesisCancelled();
  @override
  String toString() => 'Audio synthesis cancelled';
}

final class AudioSynthesisResult {
  AudioSynthesisResult(
    this.samples,
    this.sampleRate,
    this.frames,
    this.stopReason,
  );
  final Float32List samples;
  final int sampleRate, frames;

  /// `eos` or `length`. A length-limited result is partial speech.
  final String stopReason;
  double get seconds => samples.length / sampleRate;
}

/// Caller-owned model/generator, used exclusively during a synthesis operation.
/// CPU generation/codec work is synchronous per frame: use a worker isolate.
final class AudioSynthesizer {
  AudioSynthesizer(this.model, this.tokenizer, this.generator);
  final Lfm3Model model;
  final Lfm3Tokenizer tokenizer;
  final Lfm3AudioGenerator generator;
  bool _busy = false, _cancelled = false;
  void cancel() {
    _cancelled = true;
  }

  void _check() {
    if (_cancelled) throw const AudioSynthesisCancelled();
  }

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
        emit(codec.add(codes));
        _check();
        if (i + 1 < maxFrames) {
          await model.forward(
            [output],
            embeddingOverrides: {0: generator.feedback(codes)},
          );
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

/// PCM16 WAV with explicit finite-sample validation and full-scale clipping.
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
