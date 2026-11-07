/// RGB image preparation for the LFM3-VL vision encoder.
/// Reference: LiquidAI/LFM3-VL-2B image_processing_lfm3_vl.py.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

/// One image in the processor's flattened, spatially grouped patch layout.
final class VisionImageInput {
  VisionImageInput._(
    this.pixelValues,
    this.height,
    this.width,
    this.patchSize,
    this.mergeSize,
    this.temporalPatchSize,
  );

  /// Float32 [patchCount, 3 * temporalPatchSize * patchSize * patchSize].
  /// Each still image is repeated along the temporal patch dimension.
  final Float32List pixelValues;
  final int height, width, patchSize, mergeSize, temporalPatchSize;
  (int, int, int) get gridThw => (1, height ~/ patchSize, width ~/ patchSize);
  int get patchCount => (height ~/ patchSize) * (width ~/ patchSize);
  int get imageTokens => patchCount ~/ (mergeSize * mergeSize);
  int get patchDim => 3 * temporalPatchSize * patchSize * patchSize;
}

/// Processes packed RGB uint8 pixels. Decoding JPEG/PNG and applying EXIF
/// orientation are the caller's responsibility. This lab API supports the
/// official processor's RGB, antialiased bicubic and normalization settings.
final class Lfm3ImageProcessor {
  Lfm3ImageProcessor({
    this.patchSize = 16,
    this.mergeSize = 2,
    this.temporalPatchSize = 2,
    this.minPixels = 65536,
    this.maxPixels = 16777216,
  }) {
    if (patchSize <= 0 ||
        mergeSize <= 0 ||
        temporalPatchSize <= 0 ||
        minPixels < factor * factor ||
        maxPixels < minPixels) {
      throw ArgumentError('invalid image patch sizes or pixel budget');
    }
  }

  factory Lfm3ImageProcessor.load(String modelDir) =>
      Lfm3ImageProcessor.fromJson(
        jsonDecode(File('$modelDir/processor_config.json').readAsStringSync())
            as Map<String, Object?>,
      );

  factory Lfm3ImageProcessor.fromJson(Map<String, Object?> json) {
    final image = json['image_processor'];
    if (image is! Map<String, Object?>) {
      throw const FormatException(
        'processor_config.json lacks image_processor',
      );
    }
    for (final entry in {
      'image_processor_type': 'Lfm3VlImageProcessor',
      'do_convert_rgb': true,
      'do_resize': true,
      'do_rescale': true,
      'do_normalize': true,
      'resample': 3,
      'rescale_factor': 1 / 255,
    }.entries) {
      if (image[entry.key] != entry.value) {
        throw FormatException(
          'unsupported image processor ${entry.key}: ${image[entry.key]}',
        );
      }
    }
    for (final key in ['image_mean', 'image_std']) {
      final values = image[key];
      if (values is! List ||
          values.length != 3 ||
          values.any((v) => v != 0.5)) {
        throw FormatException('unsupported image processor $key');
      }
    }
    final size = image['size'];
    if (size is! Map<String, Object?>) {
      throw const FormatException('missing image size budget');
    }
    int integer(Map<String, Object?> map, String key) {
      final value = map[key];
      if (value is! int) throw FormatException('image $key must be an integer');
      return value;
    }

    return Lfm3ImageProcessor(
      patchSize: integer(image, 'patch_size'),
      mergeSize: integer(image, 'merge_size'),
      temporalPatchSize: integer(image, 'temporal_patch_size'),
      minPixels: integer(size, 'shortest_edge'),
      maxPixels: integer(size, 'longest_edge'),
    );
  }

  final int patchSize, mergeSize, temporalPatchSize, minPixels, maxPixels;
  int get factor => patchSize * mergeSize;

  /// The pixel budget in merged image tokens, each [factor] x [factor].
  int get maxImageTokens => maxPixels ~/ (factor * factor);

  /// This processor with its pixel budget lowered to [tokens] merged image
  /// tokens, or this processor if its budget is no larger. The resize rule is
  /// unchanged: the result is upstream's with a smaller longest_edge. As there,
  /// a side that would shrink below [factor] is kept at [factor], so a very
  /// elongated image can still exceed [tokens].
  Lfm3ImageProcessor withMaxImageTokens(int tokens) {
    if (tokens <= 0) {
      throw ArgumentError.value(tokens, 'tokens', 'must be positive');
    }
    if (tokens >= maxImageTokens) return this;
    final area = factor * factor;
    if (tokens * area < minPixels) {
      throw ArgumentError.value(
        tokens,
        'tokens',
        'below the minimum budget of ${(minPixels + area - 1) ~/ area} tokens',
      );
    }
    return Lfm3ImageProcessor(
      patchSize: patchSize,
      mergeSize: mergeSize,
      temporalPatchSize: temporalPatchSize,
      minPixels: minPixels,
      maxPixels: tokens * area,
    );
  }

  /// Upstream rounds ties to even before checking the pixel budget. The
  /// budget is an area, despite the config's shortest_edge/longest_edge names.
  (int height, int width) resizedSize(int height, int width) {
    if (height <= 0 ||
        width <= 0 ||
        math.max(height, width) / math.min(height, width) > 200) {
      throw ArgumentError(
        'image dimensions must be positive with aspect ratio <= 200',
      );
    }
    var h = _roundEven(height / factor) * factor;
    var w = _roundEven(width / factor) * factor;
    if (h * w > maxPixels) {
      final scale = math.sqrt(height * width / maxPixels);
      h = math.max(factor, (height / scale / factor).floor() * factor);
      w = math.max(factor, (width / scale / factor).floor() * factor);
    } else if (h * w < minPixels) {
      final scale = math.sqrt(minPixels / (height * width));
      h = (height * scale / factor).ceil() * factor;
      w = (width * scale / factor).ceil() * factor;
    }
    return (h, w);
  }

  VisionImageInput prepareRgb(
    Uint8List rgb, {
    required int height,
    required int width,
  }) {
    final (outH, outW) = resizedSize(height, width);
    if (rgb.length != height * width * 3) {
      throw ArgumentError('expected height * width * 3 packed RGB bytes');
    }
    final pixels = _resizeRgb(rgb, height, width, outH, outW);
    return _patchify(
      List.filled(temporalPatchSize, pixels),
      outH,
      outW,
      patchSize,
      mergeSize,
    );
  }
}

/// Upstream patchify: merge blocks in raster order, then channel, temporal
/// slot and pixel; [slots] holds one resized RGB frame per temporal slot.
VisionImageInput _patchify(
  List<Uint8List> slots,
  int outH,
  int outW,
  int patchSize,
  int mergeSize,
) {
  final temporalPatchSize = slots.length;
  final gridH = outH ~/ patchSize;
  final gridW = outW ~/ patchSize;
  final result = Float32List(
    gridH * gridW * 3 * temporalPatchSize * patchSize * patchSize,
  );
  // Fused torchvision normalization: (uint8 - 127.5) / 127.5.
  final normalized = Float32List.fromList([
    for (var i = 0; i < 256; i++) (i - 127.5) / 127.5,
  ]);
  var at = 0;
  for (var blockY = 0; blockY < gridH; blockY += mergeSize) {
    for (var blockX = 0; blockX < gridW; blockX += mergeSize) {
      for (var dy = 0; dy < mergeSize; dy++) {
        for (var dx = 0; dx < mergeSize; dx++) {
          for (var channel = 0; channel < 3; channel++) {
            for (final pixels in slots) {
              for (var y = 0; y < patchSize; y++) {
                final row = (blockY + dy) * patchSize + y;
                for (var x = 0; x < patchSize; x++) {
                  final col = (blockX + dx) * patchSize + x;
                  result[at++] =
                      normalized[pixels[(row * outW + col) * 3 + channel]];
                }
              }
            }
          }
        }
      }
    }
  }
  return VisionImageInput._(
    result,
    outH,
    outW,
    patchSize,
    mergeSize,
    temporalPatchSize,
  );
}

/// Frames sampled from one video, prepared as frame groups. Each group pairs
/// [VisionImageInput.temporalPatchSize] consecutive frames and is encoded
/// alone, as upstream attends within each frame group only.
final class VisionVideoInput {
  VisionVideoInput._(this.groups, this.frameIndices, this.fps);

  final List<VisionImageInput> groups;

  /// Source frame indices of the sampled frames, before padding.
  final List<int> frameIndices;

  /// The source clip's frame rate.
  final double fps;

  int get tokensPerGroup => groups.first.imageTokens;

  /// Upstream `_calculate_timestamps`: the mean time of each group's frames,
  /// the last group padded with its final frame.
  List<double> get timestamps {
    final t = groups.first.temporalPatchSize;
    final indices = [...frameIndices];
    while (indices.length % t != 0) {
      indices.add(indices.last);
    }
    return [
      for (var i = 0; i < indices.length; i += t)
        (indices[i] + indices[i + t - 1]) / (2 * fps),
    ];
  }

  /// Start time of each group's first frame.
  List<double> get groupStarts {
    final t = groups.first.temporalPatchSize;
    return [
      for (var i = 0; i < frameIndices.length; i += t) frameIndices[i] / fps,
    ];
  }
}

/// The LFM3-VL video processor as LFM3-Omni configures it: frames sampled at
/// [fps] (between [minFrames] and [maxFrames]), every frame resized to one
/// size from the whole clip's pixel budget, and grouped in temporal pairs.
/// Decoding the video and its soundtrack is the caller's responsibility.
final class Lfm3VideoProcessor {
  Lfm3VideoProcessor({
    this.patchSize = 16,
    this.mergeSize = 2,
    this.temporalPatchSize = 2,
    this.fps = 2,
    this.minFrames = 4,
    this.maxFrames = 384,
    this.minPixels = 4096,
    this.maxPixels = 25165824,
    this.capPixelsPerFrame = false,
    this.maxVideoTokens = 768,
  }) {
    if (patchSize <= 0 ||
        mergeSize <= 0 ||
        temporalPatchSize <= 0 ||
        !fps.isFinite ||
        fps <= 0 ||
        minFrames < temporalPatchSize ||
        maxFrames < minFrames ||
        minPixels <= 0 ||
        maxPixels < minPixels ||
        maxVideoTokens <= 0) {
      throw ArgumentError('invalid video processor settings');
    }
  }

  factory Lfm3VideoProcessor.load(String modelDir) =>
      Lfm3VideoProcessor.fromJson(
        jsonDecode(File('$modelDir/processor_config.json').readAsStringSync())
            as Map<String, Object?>,
      );

  factory Lfm3VideoProcessor.fromJson(Map<String, Object?> json) {
    final video = json['video_processor'];
    if (video is! Map<String, Object?>) {
      throw const FormatException(
        'processor_config.json lacks video_processor',
      );
    }
    for (final entry in {
      'video_processor_type': 'Lfm3VlVideoProcessor',
      'do_convert_rgb': true,
      'do_resize': true,
      'do_rescale': true,
      'do_normalize': true,
      'do_sample_frames': true,
      'resample': 3,
      'rescale_factor': 1 / 255,
    }.entries) {
      if (video[entry.key] != entry.value) {
        throw FormatException(
          'unsupported video processor ${entry.key}: ${video[entry.key]}',
        );
      }
    }
    for (final key in ['image_mean', 'image_std']) {
      final values = video[key];
      if (values is! List ||
          values.length != 3 ||
          values.any((v) => v != 0.5)) {
        throw FormatException('unsupported video processor $key');
      }
    }
    int integer(Map<String, Object?> map, String key) {
      final value = map[key];
      if (value is! int) throw FormatException('video $key must be an integer');
      return value;
    }

    final size = video['size'];
    final fps = video['fps'];
    if (size is! Map<String, Object?> || fps is! num) {
      throw const FormatException('missing video size budget or fps');
    }
    return Lfm3VideoProcessor(
      patchSize: integer(video, 'patch_size'),
      mergeSize: integer(video, 'merge_size'),
      temporalPatchSize: integer(video, 'temporal_patch_size'),
      fps: fps.toDouble(),
      minFrames: integer(video, 'min_frames'),
      maxFrames: integer(video, 'max_frames'),
      minPixels: integer(size, 'shortest_edge'),
      maxPixels: integer(size, 'longest_edge'),
      capPixelsPerFrame: video['cap_pixels_per_frame'] == true,
      maxVideoTokens: integer(video, 'max_video_tokens'),
    );
  }

  final int patchSize, mergeSize, temporalPatchSize, minFrames, maxFrames;
  final int minPixels, maxPixels, maxVideoTokens;
  final double fps;
  final bool capPixelsPerFrame;
  int get factor => patchSize * mergeSize;

  /// A copy with a smaller whole-clip pixel budget, for bounded encoding.
  Lfm3VideoProcessor withMaxPixels(int pixels) => Lfm3VideoProcessor(
    patchSize: patchSize,
    mergeSize: mergeSize,
    temporalPatchSize: temporalPatchSize,
    fps: fps,
    minFrames: minFrames,
    maxFrames: maxFrames,
    minPixels: math.min(minPixels, pixels),
    maxPixels: pixels,
    capPixelsPerFrame: capPixelsPerFrame,
    maxVideoTokens: maxVideoTokens,
  );

  /// Upstream `sample_frames`: rounded linspace over the whole clip.
  List<int> sampleIndices(int totalFrames, double sourceFps) {
    if (totalFrames <= 0 || !sourceFps.isFinite || sourceFps <= 0) {
      throw ArgumentError('video needs frames and a positive frame rate');
    }
    final count = math.min(
      math.max((totalFrames / sourceFps * fps).floor(), minFrames),
      math.min(maxFrames, totalFrames),
    );
    if (count == 1) return [0];
    // numpy.linspace multiplies a precomputed step and pins the endpoint.
    final step = (totalFrames - 1) / (count - 1);
    return [
      for (var i = 0; i < count - 1; i++) _roundEven(i * step),
      totalFrames - 1,
    ];
  }

  /// Upstream video `smart_resize` for [frames] sampled frames.
  (int height, int width) resizedSize(int frames, int height, int width) {
    if (frames < temporalPatchSize || height <= 0 || width <= 0) {
      throw ArgumentError('invalid video frame count or size');
    }
    var budget = maxPixels;
    if (capPixelsPerFrame) {
      final perFrame = math.max(
        math.min(maxVideoTokens * factor * factor, maxPixels ~/ frames),
        (minPixels * 1.05).floor(),
      );
      budget = perFrame * frames;
    }
    if (height < factor || width < factor) {
      final scale = math.max(factor / height, factor / width);
      height = (height * scale).floor();
      width = (width * scale).floor();
    }
    if (math.max(height, width) / math.min(height, width) > 200) {
      throw ArgumentError('video aspect ratio must be <= 200');
    }
    var h = _roundEven(height / factor) * factor;
    var w = _roundEven(width / factor) * factor;
    final t = _roundEven(frames / temporalPatchSize) * temporalPatchSize;
    if (t * h * w > budget) {
      final beta = math.sqrt(frames * height * width / budget);
      h = math.max(factor, (height / beta / factor).floor() * factor);
      w = math.max(factor, (width / beta / factor).floor() * factor);
    } else if (t * h * w < minPixels) {
      final beta = math.sqrt(minPixels / (frames * height * width));
      h = (height * beta / factor).ceil() * factor;
      w = (width * beta / factor).ceil() * factor;
    }
    return (h, w);
  }

  /// Resizes and groups already sampled frames: packed RGB, one size for all,
  /// taken at [frameIndices] of a clip at [sourceFps].
  VisionVideoInput prepareFrames(
    List<Uint8List> frames, {
    required int height,
    required int width,
    required List<int> frameIndices,
    required double sourceFps,
  }) {
    if (frames.length < temporalPatchSize ||
        frames.length != frameIndices.length ||
        frames.any((f) => f.length != height * width * 3) ||
        !sourceFps.isFinite ||
        sourceFps <= 0) {
      throw ArgumentError(
        'expected sampled frames of one size and their indices',
      );
    }
    final (outH, outW) = resizedSize(frames.length, height, width);
    final resized = [
      for (final frame in frames) _resizeRgb(frame, height, width, outH, outW),
    ];
    while (resized.length % temporalPatchSize != 0) {
      resized.add(resized.last);
    }
    return VisionVideoInput._(
      List.unmodifiable([
        for (var i = 0; i < resized.length; i += temporalPatchSize)
          _patchify(
            resized.sublist(i, i + temporalPatchSize),
            outH,
            outW,
            patchSize,
            mergeSize,
          ),
      ]),
      List.unmodifiable(frameIndices),
      // Upstream rounds the container's frame rate to six decimals.
      double.parse(sourceFps.toStringAsFixed(6)),
    );
  }
}

int _roundEven(double value) {
  final lower = value.floor();
  if (value - lower == 0.5) return lower.isEven ? lower : lower + 1;
  return value.round();
}

double _cubic(double distance) {
  final x = distance.abs();
  if (x < 1) return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
  return 0;
}

final class _Axis {
  _Axis(int input, int output) {
    final scale = input / output;
    final filterScale = math.max(1.0, scale);
    final support = 2 * filterScale;
    final raw = <List<double>>[];
    var maxWeight = 0.0;
    for (var i = 0; i < output; i++) {
      final center = scale * (i + 0.5);
      final first = math.max(0, (center - support + 0.5).truncate());
      final end = math.min(input, (center + support + 0.5).truncate());
      starts.add(first);
      final row = [
        for (var j = first; j < end; j++)
          _cubic((j - center + 0.5) / filterScale),
      ];
      final total = row.fold(0.0, (a, b) => a + b);
      for (var j = 0; j < row.length; j++) {
        row[j] /= total;
        maxWeight = math.max(maxWeight, row[j]);
      }
      raw.add(row);
    }
    // Match PyTorch's signed 16-bit coefficient precision for uint8 resize.
    while (precision < 22 &&
        (0.5 + maxWeight * (1 << (precision + 1))).truncate() < (1 << 15)) {
      precision++;
    }
    for (final row in raw) {
      weights.add([
        for (final w in row)
          (w * (1 << precision) + (w < 0 ? -0.5 : 0.5)).truncate(),
      ]);
    }
  }
  final starts = <int>[];
  final weights = <List<int>>[];
  int precision = 0;
}

Uint8List _resizeRgb(Uint8List rgb, int h, int w, int outH, int outW) {
  var horizontal = rgb;
  if (w != outW) {
    final axis = _Axis(w, outW);
    horizontal = Uint8List(h * outW * 3);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < outW; x++) {
        for (var c = 0; c < 3; c++) {
          var sum = 1 << (axis.precision - 1);
          final weights = axis.weights[x];
          for (var k = 0; k < weights.length; k++) {
            sum += rgb[(y * w + axis.starts[x] + k) * 3 + c] * weights[k];
          }
          horizontal[(y * outW + x) * 3 + c] = (sum >> axis.precision).clamp(
            0,
            255,
          );
        }
      }
    }
  }
  if (h == outH) return horizontal;
  final axis = _Axis(h, outH);
  final result = Uint8List(outH * outW * 3);
  for (var y = 0; y < outH; y++) {
    for (var x = 0; x < outW; x++) {
      for (var c = 0; c < 3; c++) {
        var sum = 1 << (axis.precision - 1);
        final weights = axis.weights[y];
        for (var k = 0; k < weights.length; k++) {
          sum +=
              horizontal[((axis.starts[y] + k) * outW + x) * 3 + c] *
              weights[k];
        }
        result[(y * outW + x) * 3 + c] = (sum >> axis.precision).clamp(0, 255);
      }
    }
  }
  return result;
}
