/// VL prompt rendering, image-token expansion and embedding placement.
library;

import 'dart:typed_data';

import 'chat_template.dart';
import 'model.dart';
import 'tokenizer.dart';
import 'vision_config.dart';

const _imagePlaceholder = '<|vision_start|><|image|><|vision_end|>';

/// Renders the still-image subset of the VL template. Images are represented
/// by a single placeholder here; [prepareVisionPrompt] expands it to the
/// number of projected image rows. Image parts use {'type': 'image'} and can
/// be interleaved with text parts. Video is explicitly unsupported.
String renderVisionChat(
  List<Map<String, Object?>> messages, {
  List<Map<String, Object?>>? tools,
  bool addGenerationPrompt = true,
  bool enableThinking = true,
  bool preserveThinking = false,
}) => renderChat(
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
  addGenerationPrompt: addGenerationPrompt,
  enableThinking: enableThinking,
  preserveThinking: preserveThinking,
);

Object? _content(Object? value, {bool toolItems = false}) {
  if (value is List) {
    // Tool responses parse each item as content; ordinary messages treat
    // nested lists as JSON. Match that extra parse level without flattening.
    return [
      for (final item in value)
        toolItems && item is List ? _content(item) : _contentPart(item),
    ];
  }
  return _contentPart(value);
}

Object? _contentPart(Object? value) {
  if (value is Map) {
    if (value['type'] == 'image') {
      return {'type': 'text', 'text': _imagePlaceholder};
    }
    if (value['type'] == 'video' || value['type'] == 'image_url') {
      throw UnsupportedError(
        'VL prompts support image parts only, not ${value['type']}',
      );
    }
  }
  return value;
}

int _imageCount(Object? value, {bool toolItems = false}) {
  if (value is List) {
    return value.fold(
      0,
      (count, part) =>
          count +
          (toolItems && part is List
              ? _imageCount(part)
              : _imagePartCount(part)),
    );
  }
  return _imagePartCount(value);
}

int _imagePartCount(Object? value) =>
    value is Map && value['type'] == 'image' ? 1 : 0;

/// A contiguous run of image tokens in an expanded prompt.
final class VisionTokenSpan {
  const VisionTokenSpan(this.start, this.count);
  final int start, count;
}

/// A tokenized visual prompt with image rows placed in message order.
final class VisionPrompt {
  VisionPrompt._(List<int> ids, List<VisionTokenSpan> images, this.hiddenSize)
    : ids = List.unmodifiable(ids),
      images = List.unmodifiable(images);

  final List<int> ids;
  final List<VisionTokenSpan> images;
  final int hiddenSize;

  /// Resets the language model, then feeds the whole visual prompt. Each
  /// entry in [projectedImages] must be the encoder/projector output for the
  /// corresponding image part, flattened as [span.count, hiddenSize].
  /// All inputs are validated and copied before resetting model state.
  /// The caller must load the matching VL language backbone, not text-chat
  /// weights. This low-level lab API does not load/check checkpoint identity.
  Future<void> prefill(
    Lfm3Model model,
    List<Float32List> projectedImages,
  ) async {
    if (model.isClosed) throw StateError('the model is closed');
    if (model.config.hiddenSize != hiddenSize) {
      throw ArgumentError('VL prompt and language hidden sizes differ');
    }
    if (ids.isEmpty ||
        ids.length > model.maxSeq ||
        ids.any((id) => id < 0 || id >= model.config.vocabSize)) {
      throw ArgumentError('visual prompt does not fit the language model');
    }
    if (projectedImages.length != images.length) {
      throw ArgumentError('one projected tensor is required per image part');
    }
    final overrides = <int, Float32List>{};
    for (var i = 0; i < images.length; i++) {
      final span = images[i];
      // One owned copy per image: the rows are views of it, and the caller
      // cannot change what was checked while reset() is awaited.
      final values = projectedImages[i].length == span.count * hiddenSize
          ? Float32List.fromList(projectedImages[i])
          : null;
      if (values == null || values.any((v) => !v.isFinite)) {
        throw ArgumentError(
          'image $i must contain ${span.count} finite projected rows',
        );
      }
      for (var row = 0; row < span.count; row++) {
        overrides[span.start + row] = Float32List.sublistView(
          values,
          row * hiddenSize,
          (row + 1) * hiddenSize,
        );
      }
    }
    await model.reset();
    await model.forward(ids, embeddingOverrides: overrides);
  }
}

/// Tokenizes a VL conversation and repeats each image placeholder to match
/// [imageTokenCounts], in message/content order, as the upstream processor does.
/// These counts are VisionImageInput.imageTokens (or encoder row counts).
/// Checks tokenizer/config special-token agreement and rejects unbound image
/// or video tokens, including placeholders embedded in plain text.
VisionPrompt prepareVisionPrompt(
  List<Map<String, Object?>> messages, {
  required Lfm3Tokenizer tokenizer,
  required Lfm3VlConfig config,
  required List<int> imageTokenCounts,
  List<Map<String, Object?>>? tools,
  bool enableThinking = true,
  bool preserveThinking = false,
}) {
  final imageCount = messages.fold(
    0,
    (count, message) =>
        count +
        _imageCount(message['content'], toolItems: message['role'] == 'tool'),
  );
  if (imageCount != imageTokenCounts.length ||
      imageTokenCounts.any((n) => n <= 0)) {
    throw ArgumentError('one positive token count is required per image part');
  }
  for (final token in {
    '<|image|>': config.imageTokenId,
    '<|vision_start|>': config.visionStartTokenId,
    '<|vision_end|>': config.visionEndTokenId,
    '<|video|>': config.videoTokenId,
  }.entries) {
    if (tokenizer.addedTokenId(token.key) != token.value) {
      throw FormatException('VL tokenizer/config mismatch for ${token.key}');
    }
  }
  final raw = tokenizer.encode(
    renderVisionChat(
      messages,
      tools: tools,
      enableThinking: enableThinking,
      preserveThinking: preserveThinking,
    ),
  );
  final ids = <int>[];
  final spans = <VisionTokenSpan>[];
  for (var at = 0; at < raw.length; at++) {
    final id = raw[at];
    if (id == config.videoTokenId) {
      throw UnsupportedError('video input is not supported');
    }
    if (id != config.imageTokenId) {
      ids.add(id);
      continue;
    }
    if (spans.length >= imageCount ||
        at == 0 ||
        at + 1 >= raw.length ||
        raw[at - 1] != config.visionStartTokenId ||
        raw[at + 1] != config.visionEndTokenId) {
      throw ArgumentError('unbound or malformed image placeholder');
    }
    final count = imageTokenCounts[spans.length];
    spans.add(VisionTokenSpan(ids.length, count));
    ids.addAll(List.filled(count, id));
  }
  if (spans.length != imageCount) {
    throw ArgumentError('rendered image count mismatch');
  }
  return VisionPrompt._(ids, spans, config.text.hiddenSize);
}
