/// The single-clip ASR subset of the official LFM3-Audio ChatML template.
library;

import 'dart:typed_data';

import 'model.dart';
import 'tokenizer.dart';

/// Research prompt for transcription, with one audio-token run.
/// No tool calls, speech output, history or thinking prefix is added.
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

  /// Greedy text-mode selection, suppressing the two input placeholders and
  /// speech/mixed mode markers as upstream generation_mode="text" does.
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
  };
  final system = 'Perform ASR${language == null ? '' : ' in $language'}.';
  final prefix =
      '<|startoftext|><|im_start|>system\n$system<|im_end|>\n<|im_start|>user\n';
  const suffix = '<|im_end|>\n<|im_start|>assistant\n';
  final head = tokenizer.encode(prefix), tail = tokenizer.encode(suffix);
  // Check token boundaries: no marker may be split or silently encoded as BPE.
  if (head.first != bos ||
      head.where((id) => id == start).length != 2 ||
      head.where((id) => id == end).length != 1 ||
      tail.first != end) {
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
