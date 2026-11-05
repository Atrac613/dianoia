/// LFM3-Omni conversations that mix text, images and audio clips.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'audio_conversation.dart';
import 'chat_template.dart';
import 'model.dart';
import 'tokenizer.dart';
import 'vision_config.dart';

enum OmniMediaKind { image, audio }

/// A contiguous run of media rows in an expanded prompt.
final class OmniMediaSpan {
  const OmniMediaSpan(this.kind, this.index, this.start, this.count);
  final OmniMediaKind kind;

  /// The media item's position among the prompt's items of its kind.
  final int index;
  final int start, count;
}

const _wrapped = {
  'image': '<|vision_start|><|image|><|vision_end|>',
  'audio': '<|vision_start|><|audio|><|vision_end|>',
};

Object? _content(Object? value, {bool toolItems = false}) {
  if (value is List) {
    return [
      for (final item in value)
        toolItems && item is List ? _content(item) : _part(item),
    ];
  }
  return _part(value);
}

Object? _part(Object? value) {
  if (value is Map) {
    final type = value['type'];
    if (type == 'input_audio' || type == 'audio_url') {
      return {'type': 'text', 'text': _wrapped['audio']};
    }
    if (_wrapped[type] case final text?) return {'type': 'text', 'text': text};
    if (type == 'video' || type == 'image_url') {
      throw UnsupportedError('Omni prompts do not support $type parts yet');
    }
  }
  return value;
}

/// Renders the official LFM3-Omni template for messages whose content parts
/// may be `{'type': 'image'}` and `{'type': 'audio'}` (also `input_audio`
/// and `audio_url`), each as one span-wrapped placeholder. The reply opens
/// with `<think></think>`, then the [mode]'s marker: none for text,
/// `<|audio_start|>` for speech and `<|mixed_start|>` for interleaved.
String renderOmniChat(
  List<Map<String, Object?>> messages, {
  AudioReplyMode mode = AudioReplyMode.text,
  List<Map<String, Object?>>? tools,
}) =>
    renderChat(
      [
        for (final message in messages)
          {
            ...message,
            if (message.containsKey('content'))
              'content': _content(
                message['content'],
                toolItems: message['role'] == 'tool',
              ),
          },
      ],
      tools: tools,
      enableThinking: false,
    ) +
    switch (mode) {
      AudioReplyMode.text => '',
      AudioReplyMode.audio => '<|audio_start|>',
      AudioReplyMode.mixed => '<|mixed_start|>',
    };

/// A tokenized Omni prompt with every media run placed in message order.
final class OmniPrompt {
  OmniPrompt._(List<int> ids, List<OmniMediaSpan> spans, this.hiddenSize)
    : ids = List.unmodifiable(ids),
      spans = List.unmodifiable(spans);

  final List<int> ids;
  final List<OmniMediaSpan> spans;
  final int hiddenSize;

  /// Resets the language model and feeds the prompt in chunks. [images] and
  /// [audio] hold the projected rows of each item, flattened as
  /// [span.count, hiddenSize], in prompt order. Inputs are validated and
  /// copied before the model is reset. Logits are computed for the last
  /// position only.
  Future<void> prefill(
    Lfm3Model model, {
    List<Float32List> images = const [],
    List<Float32List> audio = const [],
  }) async {
    if (model.isClosed) throw StateError('the model is closed');
    if (model.config.hiddenSize != hiddenSize ||
        ids.isEmpty ||
        ids.length > model.maxSeq ||
        ids.any((id) => id < 0 || id >= model.config.vocabSize)) {
      throw ArgumentError('Omni prompt does not fit the language model');
    }
    final rows = <int, Float32List>{};
    for (final span in spans) {
      final sources = span.kind == OmniMediaKind.image ? images : audio;
      final count = spans.where((s) => s.kind == span.kind).length;
      if (sources.length != count) {
        throw ArgumentError(
          'one projected tensor is required per ${span.kind.name}',
        );
      }
      final source = sources[span.index];
      final owned = source.length == span.count * hiddenSize
          ? Float32List.fromList(source)
          : null;
      if (owned == null || owned.any((v) => !v.isFinite)) {
        throw ArgumentError(
          '${span.kind.name} ${span.index} must contain ${span.count} finite rows',
        );
      }
      for (var row = 0; row < span.count; row++) {
        rows[span.start + row] = Float32List.sublistView(
          owned,
          row * hiddenSize,
          (row + 1) * hiddenSize,
        );
      }
    }
    await model.reset();
    for (var at = 0; at < ids.length; at += model.maxChunk) {
      final end = math.min(at + model.maxChunk, ids.length);
      await model.forward(
        ids.sublist(at, end),
        computeLogits: end == ids.length,
        embeddingOverrides: {
          for (var i = at; i < end; i++)
            if (rows.containsKey(i)) i - at: rows[i]!,
        },
      );
    }
  }
}

/// Tokenizes an Omni conversation and repeats each image and audio
/// placeholder to its encoder row count, as the upstream processor does:
/// [imageTokenCounts] are VisionImageInput.imageTokens, [audioTokenCounts]
/// AudioFeatures.encoderTokens, both in message and content order. Rejects
/// placeholders that are unbound, malformed or embedded in plain text.
OmniPrompt prepareOmniPrompt(
  List<Map<String, Object?>> messages, {
  required Lfm3Tokenizer tokenizer,
  required Lfm3VlConfig config,
  List<int> imageTokenCounts = const [],
  List<int> audioTokenCounts = const [],
  AudioReplyMode mode = AudioReplyMode.text,
  List<Map<String, Object?>>? tools,
}) {
  int marker(String text) =>
      tokenizer.addedTokenId(text) ??
      (throw FormatException('Omni tokenizer lacks $text'));
  final image = marker('<|image|>'), audio = marker('<|audio|>');
  final start = marker('<|vision_start|>'), end = marker('<|vision_end|>');
  final video = marker('<|video|>');
  if (image != config.imageTokenId ||
      video != config.videoTokenId ||
      start != config.visionStartTokenId ||
      end != config.visionEndTokenId) {
    throw const FormatException('Omni tokenizer and config markers differ');
  }
  if ([...imageTokenCounts, ...audioTokenCounts].any((n) => n <= 0)) {
    throw ArgumentError('media token counts must be positive');
  }
  final raw = tokenizer.encode(
    renderOmniChat(messages, mode: mode, tools: tools),
  );
  final ids = <int>[], spans = <OmniMediaSpan>[];
  var images = 0, clips = 0;
  for (var at = 0; at < raw.length; at++) {
    final id = raw[at];
    if (id == video) throw UnsupportedError('video input is not supported');
    if (id != image && id != audio) {
      ids.add(id);
      continue;
    }
    final isImage = id == image;
    final counts = isImage ? imageTokenCounts : audioTokenCounts;
    final index = isImage ? images++ : clips++;
    if (index >= counts.length ||
        at == 0 ||
        at + 1 >= raw.length ||
        raw[at - 1] != start ||
        raw[at + 1] != end) {
      throw ArgumentError('unbound or malformed media placeholder');
    }
    final kind = isImage ? OmniMediaKind.image : OmniMediaKind.audio;
    spans.add(OmniMediaSpan(kind, index, ids.length, counts[index]));
    ids.addAll(List.filled(counts[index], id));
  }
  if (images != imageTokenCounts.length || clips != audioTokenCounts.length) {
    throw ArgumentError('one token count is required per media part');
  }
  return OmniPrompt._(ids, spans, config.text.hiddenSize);
}
