/// CPU float32 reference encoder and projector for one LFM3-VL image.
/// Operator structure follows the pinned Liquid AI modeling_lfm3_vl.py.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'matmul.dart';
import 'ops.dart';
import 'safetensors.dart';
import 'vision_config.dart';
import 'vision_image.dart';

/// Correctness-first, synchronous CPU inference. Weights are widened to f32
/// on the Dart heap; this is a lab API, not the optimized on-device path.
/// Images are encoded independently; no cross-image attention or video state.
final class Lfm3VisionEncoder {
  Lfm3VisionEncoder._(this.config, this._weights);

  factory Lfm3VisionEncoder.load(String modelDir) {
    final c = Lfm3VlConfig.load(modelDir);
    final st = SafetensorsFile.open('$modelDir/model.safetensors');
    try {
      return Lfm3VisionEncoder._(c, {
        for (final e in lfm3VisionTensorShapes(c).entries)
          e.key: st.readF32(e.key, e.value),
      });
    } finally {
      st.close();
    }
  }

  final Lfm3VlConfig config;
  final Map<String, Float32List> _weights;
  static const _v = Lfm3VlConfig.visionPrefix;
  static const _p = Lfm3VlConfig.projectorPrefix;

  /// Returns [image.imageTokens, config.text.hiddenSize] projected rows,
  /// ready for Lfm3Model.forward's embeddingOverrides. [capture] gets owned
  /// snapshots after embedding norm, each layer and projection.
  Float32List encode(
    VisionImageInput image, {
    void Function(String stage, Float32List rows)? capture,
  }) {
    Float32List? result;
    for (final (stage, rows) in _stages(image)) {
      capture?.call(stage, Float32List.fromList(rows));
      result = rows;
    }
    return result!;
  }

  /// Same arithmetic as [encode], yielding to the event loop between stages.
  /// [checkCancelled] may throw to abort before the next stage is computed.
  /// A stage itself is synchronous CPU work; run inference in a worker isolate
  /// when integrating with a UI.
  Future<Float32List> encodeAsync(
    VisionImageInput image, {
    void Function(String stage, Float32List rows)? capture,
    void Function()? checkCancelled,
  }) async {
    await Future<void>.delayed(Duration.zero);
    checkCancelled?.call();
    Float32List? result;
    for (final (stage, rows) in _stages(image)) {
      capture?.call(stage, Float32List.fromList(rows));
      result = rows;
      await Future<void>.delayed(Duration.zero);
      checkCancelled?.call();
    }
    return result!;
  }

  Iterable<(String, Float32List)> _stages(VisionImageInput image) sync* {
    final c = config.vision;
    if (c.inChannels != 3 ||
        image.patchSize != c.patchSize ||
        image.temporalPatchSize != c.temporalPatchSize ||
        image.mergeSize != c.spatialMergeSize ||
        image.pixelValues.length != image.patchCount * image.patchDim ||
        image.pixelValues.any((v) => !v.isFinite)) {
      throw ArgumentError('image preprocessing does not match vision config');
    }
    final n = image.patchCount;
    final h = c.hiddenSize;
    final (_, gh, gw) = image.gridThw;
    final positions = <(int, int)>[
      for (var by = 0; by < gh; by += c.spatialMergeSize)
        for (var bx = 0; bx < gw; bx += c.spatialMergeSize)
          for (var dy = 0; dy < c.spatialMergeSize; dy++)
            for (var dx = 0; dx < c.spatialMergeSize; dx++) (by + dy, bx + dx),
    ];
    final rasterToToken = List<int>.filled(n, 0);
    for (var i = 0; i < n; i++) {
      rasterToToken[positions[i].$1 * gw + positions[i].$2] = i;
    }
    var x = _linear('${_v}patchifier', image.pixelValues, h, image.patchDim, n);
    x = _norm(x, '${_v}embedding_norm.weight', h, c.normEps);
    yield ('embedding', x);
    for (var i = 0; i < c.numLayers; i++) {
      final prefix = '${_v}layers.$i.';
      final normed = _norm(x, '${prefix}operator_norm.weight', h, c.normEps);
      final mixed = c.layerTypes[i] == VisionLayerType.conv
          ? _conv(normed, '${prefix}conv.', positions, rasterToToken, gh, gw)
          : _attention(normed, '${prefix}self_attn.', positions);
      _add(x, mixed);
      final f = '${prefix}feed_forward.';
      final a = _norm(x, '${prefix}ffn_norm.weight', h, c.normEps);
      final gate = _linear('${f}w1', a, c.intermediateSize, h, n);
      final up = _linear('${f}w3', a, c.intermediateSize, h, n);
      for (var j = 0; j < gate.length; j++) {
        gate[j] = gate[j] / (1 + math.exp(-gate[j]));
        gate[j] *= up[j];
      }
      _add(x, _linear('${f}w2', gate, h, c.intermediateSize, n));
      yield ('layer.$i', x);
    }
    // LayerNorm precedes merging; it normalizes each vision token alone.
    final normalized = Float32List(x.length);
    final normW = _weights['${_p}layer_norm.weight']!;
    final normB = _weights['${_p}layer_norm.bias']!;
    for (var t = 0; t < n; t++) {
      var mean = 0.0;
      for (var j = 0; j < h; j++) {
        mean += x[t * h + j];
      }
      mean /= h;
      var variance = 0.0;
      for (var j = 0; j < h; j++) {
        final d = x[t * h + j] - mean;
        variance += d * d;
      }
      final scale = 1 / math.sqrt(variance / h + config.projectorNormEps);
      for (var j = 0; j < h; j++) {
        normalized[t * h + j] =
            (x[t * h + j] - mean) * scale * normW[j] + normB[j];
      }
    }
    final projected = _linear(
      '${_p}linear_1',
      normalized,
      config.projectorHiddenSize,
      config.projectorInputSize,
      image.imageTokens,
      bias: true,
    );
    for (var i = 0; i < projected.length; i++) {
      final value = projected[i];
      projected[i] = 0.5 * value * (1 + _erf(value / math.sqrt2));
    }
    final result = _linear(
      '${_p}linear_2',
      projected,
      config.text.hiddenSize,
      config.projectorHiddenSize,
      image.imageTokens,
      bias: true,
    );
    yield ('projected', result);
  }

  Float32List _linear(
    String name,
    Float32List x,
    int rows,
    int cols,
    int n, {
    bool bias = false,
  }) {
    final result = Float32List(n * rows);
    final w = _weights['$name.weight']!;
    if (cols % 4 == 0) {
      gemm(w, x, result, rows, cols, n);
    } else {
      for (var t = 0; t < n; t++) {
        for (var r = 0; r < rows; r++) {
          var sum = 0.0;
          for (var j = 0; j < cols; j++) {
            sum += w[r * cols + j] * x[t * cols + j];
          }
          result[t * rows + r] = sum;
        }
      }
    }
    if (bias) {
      final b = _weights['$name.bias']!;
      for (var i = 0; i < result.length; i++) {
        result[i] += b[i % rows];
      }
    }
    return result;
  }

  Float32List _norm(Float32List x, String name, int width, double eps) {
    final result = Float32List(x.length);
    for (var offset = 0; offset < x.length; offset += width) {
      rmsNorm(x, _weights[name]!, result, offset, offset, width, eps);
    }
    return result;
  }

  Float32List _conv(
    Float32List x,
    String p,
    List<(int, int)> positions,
    List<int> rasterToToken,
    int gh,
    int gw,
  ) {
    final c = config.vision;
    final h = c.hiddenSize;
    final n = positions.length;
    final bc = _linear('${p}in_proj', x, 3 * h, h, n);
    final gated = Float32List(n * h);
    for (var t = 0; t < n; t++) {
      for (var j = 0; j < h; j++) {
        gated[t * h + j] = bc[t * 3 * h + j] * bc[t * 3 * h + 2 * h + j];
      }
    }
    final kernel = _weights['${p}spatial_conv.weight']!;
    final k = c.convKernelSize;
    final mixed = Float32List(n * h);
    for (var t = 0; t < n; t++) {
      final (y, x) = positions[t];
      for (var channel = 0; channel < h; channel++) {
        var sum = 0.0;
        for (var ky = 0; ky < k; ky++) {
          final row = y + ky - k ~/ 2;
          if (row < 0 || row >= gh) continue;
          for (var kx = 0; kx < k; kx++) {
            final col = x + kx - k ~/ 2;
            if (col < 0 || col >= gw) continue;
            sum +=
                gated[rasterToToken[row * gw + col] * h + channel] *
                kernel[(channel * k + ky) * k + kx];
          }
        }
        mixed[t * h + channel] = sum;
        mixed[t * h + channel] *= bc[t * 3 * h + h + channel];
      }
    }
    return _linear('${p}out_proj', mixed, h, h, n);
  }

  Float32List _attention(Float32List x, String p, List<(int, int)> positions) {
    final c = config.vision;
    final h = c.hiddenSize;
    final d = c.headDim;
    final kv = c.numKvHeads * d;
    final n = positions.length;
    var q = _linear('${p}q_proj', x, h, h, n);
    var k = _linear('${p}k_proj', x, kv, h, n);
    final v = _linear('${p}v_proj', x, kv, h, n);
    q = _norm(q, '${p}q_layernorm.weight', d, c.normEps);
    k = _norm(k, '${p}k_layernorm.weight', d, c.normEps);
    _rotate(q, c.numHeads, positions);
    _rotate(k, c.numKvHeads, positions);
    final output = Float32List(n * h);
    final scores = Float32List(n);
    final scale = 1 / math.sqrt(d);
    for (var head = 0; head < c.numHeads; head++) {
      final kvHead = head ~/ (c.numHeads ~/ c.numKvHeads);
      for (var t = 0; t < n; t++) {
        var maximum = double.negativeInfinity;
        for (var s = 0; s < n; s++) {
          var dot = 0.0;
          for (var j = 0; j < d; j++) {
            dot += q[t * h + head * d + j] * k[s * kv + kvHead * d + j];
          }
          scores[s] = dot;
          scores[s] *= scale;
          maximum = math.max(maximum, scores[s]);
        }
        var total = 0.0;
        for (var s = 0; s < n; s++) {
          scores[s] = math.exp(scores[s] - maximum);
          total += scores[s];
        }
        for (var s = 0; s < n; s++) {
          scores[s] /= total;
        }
        for (var j = 0; j < d; j++) {
          var sum = 0.0;
          for (var s = 0; s < n; s++) {
            sum += scores[s] * v[s * kv + kvHead * d + j];
          }
          output[t * h + head * d + j] = sum;
        }
      }
    }
    return _linear('${p}out_proj', output, h, h, n);
  }

  void _rotate(Float32List values, int heads, List<(int, int)> positions) {
    final d = config.vision.headDim;
    final half = d ~/ 2;
    final quarter = d ~/ 4;
    final f32 = Float32List(3);
    final frequencies = Float32List(quarter);
    for (var j = 0; j < quarter; j++) {
      f32[0] = 2 * j / half;
      f32[0] = math.pow(config.vision.ropeTheta, f32[0]).toDouble();
      frequencies[j] = 1 / f32[0];
    }
    for (var t = 0; t < positions.length; t++) {
      for (var j = 0; j < half; j++) {
        final axis = j < quarter ? positions[t].$1 : positions[t].$2;
        f32[0] = frequencies[j % quarter] * axis;
        f32[1] = math.cos(f32[0]);
        f32[2] = math.sin(f32[0]);
        final cos = f32[1], sin = f32[2];
        for (var head = 0; head < heads; head++) {
          final at = (t * heads + head) * d + j;
          final a = values[at], b = values[at + half];
          values[at] = a * cos - b * sin;
          values[at + half] = b * cos + a * sin;
        }
      }
    }
  }
}

void _add(Float32List x, Float32List y) {
  for (var i = 0; i < x.length; i++) {
    x[i] += y[i];
  }
}

// Abramowitz-Stegun 7.1.26, maximum absolute error about 1.5e-7.
// Used for the exact-erf GELU formula, rather than the tanh GELU variant.
double _erf(double value) {
  final x = value.abs();
  final t = 1 / (1 + 0.3275911 * x);
  final polynomial =
      (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
                  0.284496736) *
              t +
          0.254829592) *
      t;
  final result = 1 - polynomial * math.exp(-x * x);
  return value < 0 ? -result : result;
}

/// Required vision/projector checkpoint tensor names and shapes for I/O.
Map<String, List<int>> lfm3VisionTensorShapes(Lfm3VlConfig config) {
  final c = config.vision;
  final h = c.hiddenSize;
  const v = Lfm3VlConfig.visionPrefix, p = Lfm3VlConfig.projectorPrefix;
  final result = <String, List<int>>{
    '${v}patchifier.weight': [
      h,
      c.inChannels * c.temporalPatchSize * c.patchSize * c.patchSize,
    ],
    '${v}embedding_norm.weight': [h],
    '${p}layer_norm.weight': [h],
    '${p}layer_norm.bias': [h],
    '${p}linear_1.weight': [
      config.projectorHiddenSize,
      config.projectorInputSize,
    ],
    '${p}linear_1.bias': [config.projectorHiddenSize],
    '${p}linear_2.weight': [config.text.hiddenSize, config.projectorHiddenSize],
    '${p}linear_2.bias': [config.text.hiddenSize],
  };
  for (var i = 0; i < c.numLayers; i++) {
    final b = '${v}layers.$i.';
    result['${b}operator_norm.weight'] = [h];
    result['${b}ffn_norm.weight'] = [h];
    result['${b}feed_forward.w1.weight'] = [c.intermediateSize, h];
    result['${b}feed_forward.w3.weight'] = [c.intermediateSize, h];
    result['${b}feed_forward.w2.weight'] = [h, c.intermediateSize];
    if (c.layerTypes[i] == VisionLayerType.conv) {
      result['${b}conv.in_proj.weight'] = [3 * h, h];
      result['${b}conv.out_proj.weight'] = [h, h];
      result['${b}conv.spatial_conv.weight'] = [
        h,
        1,
        c.convKernelSize,
        c.convKernelSize,
      ];
    } else {
      final a = '${b}self_attn.';
      result['${a}q_proj.weight'] = [h, h];
      result['${a}k_proj.weight'] = [c.numKvHeads * c.headDim, h];
      result['${a}v_proj.weight'] = [c.numKvHeads * c.headDim, h];
      result['${a}out_proj.weight'] = [h, h];
      result['${a}q_layernorm.weight'] = [c.headDim];
      result['${a}k_layernorm.weight'] = [c.headDim];
    }
  }
  return result;
}
