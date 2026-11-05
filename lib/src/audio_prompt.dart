/// The single-clip ASR subset of the official LFM3-Audio and LFM3-Omni
/// ChatML templates.
library;

import 'dart:typed_data';

import 'audio_gguf.dart';
import 'model.dart';
import 'tokenizer.dart';

/// Research prompt for transcription, with one audio-token run.
/// No tool calls, speech output or history is added; LFM3-Omni's prompt
/// wraps the run in span markers and opens the reply with `<think></think>`.
final class AudioTranscriptionPrompt {
  AudioTranscriptionPrompt._(
    this.text,
    List<int> ids,
    this.audioStart,
    this.audioTokens,
    this.hiddenSize,
    Set<int> suppressed,
    Set<int> stops,
  ) : ids = List.unmodifiable(ids),
      suppressedTokenIds = Set.unmodifiable(suppressed),
      stopTokenIds = Set.unmodifiable(stops);

  final String text;
  final List<int> ids;
  final int audioStart, audioTokens, hiddenSize;
  final Set<int> suppressedTokenIds;
  final Set<int> stopTokenIds;

  /// Validates/copies all rows before clearing model state. Use the matching
  /// Audio backbone; this lab API checks dimensions, not checkpoint identity.
  Future<void> prefill(Lfm3Model model, Float32List projected) async {
    if (model.isClosed) throw StateError('Language model is closed');
    if (model.config.hiddenSize != hiddenSize ||
        ids.length > model.maxSeq ||
        ids.any((id) => id < 0 || id >= model.config.vocabSize) ||
        projected.length != audioTokens * hiddenSize) {
      throw ArgumentError('audio prompt and language model dimensions differ');
    }
    final owned = Float32List.fromList(projected);
    if (owned.any((v) => !v.isFinite)) {
      throw ArgumentError('nonfinite audio embeddings');
    }
    final overrides = {
      for (var t = 0; t < audioTokens; t++)
        audioStart + t: Float32List.sublistView(
          owned,
          t * hiddenSize,
          (t + 1) * hiddenSize,
        ),
    };
    await model.reset();
    await model.forward(ids, embeddingOverrides: overrides);
  }

  /// Greedy text-mode selection, suppressing the input placeholders (Omni's
  /// image and video ones too) and speech/mixed mode markers, as upstream
  /// generation_mode="text" does.
  int greedyToken(Float32List logits) {
    var best = -1;
    var value = double.negativeInfinity;
    for (var i = 0; i < logits.length; i++) {
      if (!suppressedTokenIds.contains(i) && !logits[i].isFinite) {
        throw StateError('nonfinite text logits');
      }
      if (!suppressedTokenIds.contains(i) && logits[i] > value) {
        best = i;
        value = logits[i];
      }
    }
    if (best < 0) throw StateError('no finite candidate text token');
    return best;
  }
}

AudioTranscriptionPrompt prepareAudioTranscriptionPrompt({
  required Lfm3Tokenizer tokenizer,
  required int audioTokens,
  required int hiddenSize,
  String? language,
  Lfm3SpeechVariant variant = Lfm3SpeechVariant.audio,
}) {
  if (audioTokens <= 0 ||
      hiddenSize <= 0 ||
      (language != null &&
          !RegExp(r'^[A-Za-z][A-Za-z -]{0,63}$').hasMatch(language))) {
    throw ArgumentError('invalid audio dimensions or language name');
  }
  int token(String text) =>
      tokenizer.addedTokenId(text) ??
      (throw FormatException('audio tokenizer lacks $text'));
  final bos = token('<|startoftext|>'), start = token('<|im_start|>');
  final end = token('<|im_end|>'), audio = token('<|audio|>');
  final suppressed = {
    audio,
    token('<|audio_out|>'),
    token('<|audio_start|>'),
    token('<|mixed_start|>'),
    if (variant == Lfm3SpeechVariant.omni) ...[
      token('<|image|>'),
      token('<|video|>'),
    ],
  };
  final system = 'Perform ASR${language == null ? '' : ' in $language'}.';
  final prefix =
      '<|startoftext|><|im_start|>system\n$system<|im_end|>\n<|im_start|>user\n'
      '${variant.audioOpen}';
  final suffix =
      '${variant.audioClose}<|im_end|>\n<|im_start|>assistant\n'
      '${variant.assistantOpen}';
  final head = tokenizer.encode(prefix), tail = tokenizer.encode(suffix);
  final close = variant == Lfm3SpeechVariant.omni
      ? [token('<|vision_end|>'), end]
      : [end];
  // Check token boundaries: no marker may be split or silently encoded as BPE.
  if (head.first != bos ||
      head.where((id) => id == start).length != 2 ||
      head.where((id) => id == end).length != 1 ||
      (variant == Lfm3SpeechVariant.omni &&
          head.last != token('<|vision_start|>')) ||
      tail.length < close.length ||
      tail.sublist(0, close.length).join(',') != close.join(',')) {
    throw const FormatException('incompatible audio tokenizer markers');
  }
  return AudioTranscriptionPrompt._(
    '$prefix<|audio|>$suffix',
    [...head, ...List.filled(audioTokens, audio), ...tail],
    head.length,
    audioTokens,
    hiddenSize,
    suppressed,
    {end, token('<|text_end|>')},
  );
}

/// Incrementally feeds one recording's projected rows into the ASR backbone.
///
/// Start before accepting audio, call [addProjected] for each nonempty encoder
/// output, then [finish] once to append the assistant prefix and compute logits.
/// No recording or projected-row history is stored here. The caller owns the
/// model and encoder and must use the model exclusively during this operation.
/// This is input streaming; generated text starts after the recording finishes.
final class AudioTranscriptionPrefill {
  AudioTranscriptionPrefill._(this._model, this._template, this.reservedTokens);

  static Future<AudioTranscriptionPrefill> start({
    required Lfm3Model model,
    required Lfm3Tokenizer tokenizer,
    required int hiddenSize,
    String? language,
    int reservedTokens = 128,
    Lfm3SpeechVariant variant = Lfm3SpeechVariant.audio,
  }) async {
    if (model.isClosed) throw StateError('Language model is closed');
    if (hiddenSize != model.config.hiddenSize || reservedTokens < 0) {
      throw ArgumentError('invalid language dimensions or token reserve');
    }
    final template = prepareAudioTranscriptionPrompt(
      tokenizer: tokenizer,
      audioTokens: 1,
      hiddenSize: hiddenSize,
      language: language,
      variant: variant,
    );
    if (template.ids.any((id) => id < 0 || id >= model.config.vocabSize) ||
        template.ids.length + reservedTokens > model.maxSeq) {
      throw ArgumentError('audio prompt and token reserve exceed context');
    }
    final result = AudioTranscriptionPrefill._(model, template, reservedTokens);
    await model.reset();
    await model.forward(
      template.ids.sublist(0, template.audioStart),
      computeLogits: false,
    );
    return result;
  }

  final Lfm3Model _model;
  final AudioTranscriptionPrompt _template;
  final int reservedTokens;
  int _audioTokens = 0;
  bool _closed = false, _busy = false, _finished = false;

  int get audioTokens => _audioTokens;
  int get promptTokens => _template.ids.length - 1 + _audioTokens;
  bool get isFinished => _finished;
  Set<int> get stopTokenIds => _template.stopTokenIds;
  int greedyToken(Float32List logits) => _template.greedyToken(logits);

  void _checkOpen() {
    if (_closed || _model.isClosed) throw StateError('Audio prefill is closed');
    if (_busy) throw StateError('Audio prefill is busy');
    if (_model.length != _template.audioStart + _audioTokens) {
      _closed = true;
      throw StateError('Language state changed outside audio prefill');
    }
  }

  /// Invalid rows/context overflow are rejected before model state changes.
  /// Calls are serialized; concurrent calls fail instead of mixing recordings.
  /// Any forward-pass failure terminates this prefill; start a new one to retry.
  Future<void> addProjected(Float32List rows) async {
    _checkOpen();
    final h = _template.hiddenSize;
    if (rows.length % h != 0 || rows.any((v) => !v.isFinite)) {
      throw ArgumentError('expected finite projected audio rows');
    }
    final count = rows.length ~/ h;
    if (promptTokens + count + reservedTokens > _model.maxSeq) {
      throw ArgumentError('audio rows and token reserve exceed context');
    }
    if (count == 0) return;
    final owned = Float32List.fromList(rows);
    final id = _template.ids[_template.audioStart];
    _busy = true;
    try {
      await _model.forward(
        List.filled(count, id),
        computeLogits: false,
        embeddingOverrides: {
          for (var t = 0; t < count; t++)
            t: Float32List.sublistView(owned, t * h, (t + 1) * h),
        },
      );
      if (_closed) throw StateError('Audio prefill was closed');
      _audioTokens += count;
    } catch (_) {
      _closed = true;
      rethrow;
    } finally {
      _busy = false;
    }
  }

  /// Appends the user-turn end and assistant prefix without replaying audio.
  /// Afterwards model.logits is ready for text-mode greedy generation.
  Future<void> finish() async {
    _checkOpen();
    if (_audioTokens == 0) {
      throw StateError('Audio prefill needs at least one audio row');
    }
    _busy = true;
    try {
      await _model.forward(_template.ids.sublist(_template.audioStart + 1));
      if (_closed) throw StateError('Audio prefill was closed');
      _finished = true;
      _closed = true;
    } catch (_) {
      _closed = true;
      rethrow;
    } finally {
      _busy = false;
    }
  }

  /// Stops this input operation without closing the caller-owned language model.
  /// An already running forward pass may complete; await it before model reuse.
  void close() {
    _closed = true;
  }
}
