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
              for (var t = 0; t < temporalPatchSize; t++) {
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
