/// LFM3-Omni conversations that mix text, images, audio clips and videos.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'audio_conversation.dart';
import 'chat_template.dart';
import 'model.dart';
import 'tokenizer.dart';
import 'vision_config.dart';
import 'vision_image.dart';

enum OmniMediaKind { image, audio, video, soundtrack }

/// A contiguous run of media rows in an expanded prompt.
final class OmniMediaSpan {
  const OmniMediaSpan(
    this.kind,
    this.index,
    this.start,
    this.count, [
    this.rowOffset = 0,
  ]);
  final OmniMediaKind kind;

  /// The item's position among the prompt's items of its kind; a soundtrack
  /// shares its video's index.
  final int index;
  final int start, count;

  /// The first row of the item's projected rows this run takes: a video's
  /// frame group, or a soundtrack's share of one group.
  final int rowOffset;
}

/// How one video expands: its frame groups, their times, and optionally the
/// encoder rows of its soundtrack, split between the groups.
final class OmniVideoLayout {
  OmniVideoLayout({
    required List<double> timestamps,
    required List<double> groupStarts,
    required this.tokensPerGroup,
    this.soundtrackTokens,
  }) : timestamps = List.unmodifiable(timestamps),
       groupStarts = List.unmodifiable(groupStarts) {
    if (timestamps.isEmpty ||
        groupStarts.length != timestamps.length ||
        tokensPerGroup <= 0 ||
        (soundtrackTokens != null && soundtrackTokens! < 0)) {
      throw ArgumentError('invalid video layout');
    }
  }

  factory OmniVideoLayout.of(VisionVideoInput video, {int? soundtrackTokens}) {
    final timestamps = video.timestamps;
    return OmniVideoLayout(
      timestamps: timestamps,
      groupStarts: video.groupStarts.take(timestamps.length).toList(),
      tokensPerGroup: video.tokensPerGroup,
      soundtrackTokens: soundtrackTokens,
    );
  }

  final List<double> timestamps, groupStarts;
  final int tokensPerGroup;
  final int? soundtrackTokens;
  int get groups => timestamps.length;

  /// Upstream split points: each group's soundtrack rows start at the
  /// encoder row nearest its first frame, 80 ms per row.
  List<int> get soundtrackBounds {
    final total = soundtrackTokens ?? 0;
    const secondsPerToken = 160 * 8 / 16000;
    final bounds = [
      0,
      for (final start in groupStarts.skip(1))
        math.min(total, _roundEven(start / secondsPerToken)),
      total,
    ];
    for (var i = 1; i < bounds.length; i++) {
      bounds[i] = math.max(bounds[i], bounds[i - 1]);
    }
    return bounds;
  }
}

int _roundEven(double value) {
  final lower = value.floor();
  if (value - lower == 0.5) return lower.isEven ? lower : lower + 1;
  return value.round();
}

/// Python's `f'{x:.1f}'`: the exact binary value, ties to even.
String formatOmniSeconds(double x) {
  final quarters = x * 4;
  if (quarters == quarters.roundToDouble() && quarters.round().isOdd) {
    final tenths = _roundEven(x * 10);
    return '${tenths ~/ 10}.${tenths % 10}';
  }
  return x.toStringAsFixed(1);
}

const _wrapped = {
  'image': '<|vision_start|><|image|><|vision_end|>',
  'audio': '<|vision_start|><|audio|><|vision_end|>',
  'video': '<|vision_start|><|video|><|vision_end|>',
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
    if (type == 'image_url') {
      throw UnsupportedError('Omni prompts take decoded images, not URLs');
    }
  }
  return value;
}

/// Renders the official LFM3-Omni template for messages whose content parts
/// may be `{'type': 'image'}`, `{'type': 'audio'}` (also `input_audio` and
/// `audio_url`) and `{'type': 'video'}`, each as one span-wrapped
/// placeholder. A video's soundtrack is not a separate part here: it is
/// declared in the video's [OmniVideoLayout]. The reply opens with
/// `<think></think>`, then the [mode]'s marker: none for text,
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

  /// Resets the language model and feeds the prompt in chunks. Each list
  /// holds one item's projected rows, flattened as [rows, hiddenSize], in
  /// prompt order: [images], [audio] clips, [videos] (all frame groups in
  /// order) and the [soundtracks] of videos declared with one (null for the
  /// others). Inputs are validated and copied before the model is reset.
  /// Logits are computed for the last position only.
  Future<void> prefill(
    Lfm3Model model, {
    List<Float32List> images = const [],
    List<Float32List> audio = const [],
    List<Float32List> videos = const [],
    List<Float32List?> soundtracks = const [],
  }) async {
    if (model.isClosed) throw StateError('the model is closed');
    if (model.config.hiddenSize != hiddenSize ||
        ids.isEmpty ||
        ids.length > model.maxSeq ||
        ids.any((id) => id < 0 || id >= model.config.vocabSize)) {
      throw ArgumentError('Omni prompt does not fit the language model');
    }
    final sources = {
      OmniMediaKind.image: images,
      OmniMediaKind.audio: audio,
      OmniMediaKind.video: videos,
      OmniMediaKind.soundtrack: soundtracks,
    };
    final counts = {
      OmniMediaKind.image: _count(OmniMediaKind.image),
      OmniMediaKind.audio: _count(OmniMediaKind.audio),
      OmniMediaKind.video: _count(OmniMediaKind.video),
      OmniMediaKind.soundtrack: _count(OmniMediaKind.video),
    };
    for (final kind in OmniMediaKind.values) {
      if (sources[kind]!.length != counts[kind]) {
        throw ArgumentError(
          'one projected tensor is required per ${kind.name}',
        );
      }
    }
    // One owned copy per item: the rows are views of it.
    final owned = <(OmniMediaKind, int), Float32List>{};
    final needed = <(OmniMediaKind, int), int>{};
    for (final span in spans) {
      final key = (span.kind, span.index);
      needed[key] = math.max(
        needed[key] ?? 0,
        (span.rowOffset + span.count) * hiddenSize,
      );
    }
    for (final MapEntry(:key, value: length) in needed.entries) {
      final source = sources[key.$1]![key.$2];
      if (source == null ||
          source.length != length ||
          source.any((v) => !v.isFinite)) {
        throw ArgumentError(
          '${key.$1.name} ${key.$2} must contain ${length ~/ hiddenSize} '
          'finite rows',
        );
      }
      owned[key] = Float32List.fromList(source);
    }
    final rows = <int, Float32List>{};
    for (final span in spans) {
      final values = owned[(span.kind, span.index)]!;
      for (var row = 0; row < span.count; row++) {
        final at = (span.rowOffset + row) * hiddenSize;
        rows[span.start + row] = Float32List.sublistView(
          values,
          at,
          at + hiddenSize,
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

  int _count(OmniMediaKind kind) {
    var count = 0;
    for (final span in spans) {
      if (span.kind == kind) count = math.max(count, span.index + 1);
    }
    return count;
  }
}

/// Tokenizes an Omni conversation and expands each media placeholder as the
/// upstream processor does. Images and audio clips repeat their placeholder
/// per encoder row ([imageTokenCounts]: VisionImageInput.imageTokens,
/// [audioTokenCounts]: AudioFeatures.encoderTokens). A video becomes, per
/// frame group, its `<t seconds>` time, the group's span-wrapped rows and its
/// share of the soundtrack's rows, all inside one outer span ([videos], one
/// layout per video part). Counts are in message and content order.
/// Rejects placeholders that are unbound, malformed or in plain text.
OmniPrompt prepareOmniPrompt(
  List<Map<String, Object?>> messages, {
  required Lfm3Tokenizer tokenizer,
  required Lfm3VlConfig config,
  List<int> imageTokenCounts = const [],
  List<int> audioTokenCounts = const [],
  List<OmniVideoLayout> videos = const [],
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
  final seen = {image: 0, audio: 0, video: 0};
  for (var at = 0; at < raw.length; at++) {
    final id = raw[at];
    if (!seen.containsKey(id)) {
      ids.add(id);
      continue;
    }
    final index = seen[id]!;
    seen[id] = index + 1;
    final available = id == image
        ? imageTokenCounts.length
        : id == audio
        ? audioTokenCounts.length
        : videos.length;
    if (index >= available ||
        at == 0 ||
        at + 1 >= raw.length ||
        raw[at - 1] != start ||
        raw[at + 1] != end) {
      throw ArgumentError('unbound or malformed media placeholder');
    }
    if (id != video) {
      final isImage = id == image;
      final count = (isImage ? imageTokenCounts : audioTokenCounts)[index];
      final kind = isImage ? OmniMediaKind.image : OmniMediaKind.audio;
      spans.add(OmniMediaSpan(kind, index, ids.length, count));
      ids.addAll(List.filled(count, id));
      continue;
    }
    final layout = videos[index];
    final bounds = layout.soundtrackBounds;
    for (var g = 0; g < layout.groups; g++) {
      ids.addAll(
        tokenizer.encode(
          '<${formatOmniSeconds(layout.timestamps[g])} seconds>',
        ),
      );
      ids.add(start);
      spans.add(
        OmniMediaSpan(
          OmniMediaKind.video,
          index,
          ids.length,
          layout.tokensPerGroup,
          g * layout.tokensPerGroup,
        ),
      );
      ids.addAll(List.filled(layout.tokensPerGroup, video));
      ids.add(end);
      final share = bounds[g + 1] - bounds[g];
      if (layout.soundtrackTokens != null && share > 0) {
        spans.add(
          OmniMediaSpan(
            OmniMediaKind.soundtrack,
            index,
            ids.length,
            share,
            bounds[g],
          ),
        );
        ids.addAll(List.filled(share, audio));
      }
    }
  }
  if (seen[image] != imageTokenCounts.length ||
      seen[audio] != audioTokenCounts.length ||
      seen[video] != videos.length) {
    throw ArgumentError('one token count or layout is required per media part');
  }
  return OmniPrompt._(ids, spans, config.text.hiddenSize);
}
