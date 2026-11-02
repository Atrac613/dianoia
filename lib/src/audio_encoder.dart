/// LFM2 audio encoder and projector, consuming official Q4_0 GGUF blocks.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'audio_features.dart';
import 'audio_gguf.dart';
import 'config.dart';
import 'gguf.dart';
import 'matmul.dart';
import 'native.dart';
import 'ops.dart';

/// Synchronous CPU lab implementation for one complete clip.
///
/// Q4 weights retain their published blocks. Convolutions and norms stay F32.
/// Matmul uses float inputs by default to isolate weight quantization from
/// activation quantization. Run in a worker isolate for interactive apps.
/// This does not yet implement streaming encoder caches or audio generation.
final class Lfm3AudioEncoder {
  Lfm3AudioEncoder._(
    this.config,
    this._floats,
    this._matrices,
    this.int8Inputs,
  );

  factory Lfm3AudioEncoder.load(String mmproj, {bool int8Inputs = false}) {
    if (!NativeKernels.isLoaded) NativeKernels.load(null);
    final file = GgufFile.open(mmproj);
    final matrices = <String, NativeQ4Matrix>{};
    try {
      final c = Lfm3AudioEncoderConfig.fromGguf(file);
      final floats = <String, Float32List>{};
      for (final e in c.tensorShapes.entries) {
        final tensor = file.tensors[e.key]!;
        if (c.isQuantized(e.key)) {
          matrices[e.key] = NativeQ4Matrix.fromGgufBlocks(
            file.bytes(tensor),
            e.value[0],
            e.value[1],
          );
        } else {
          floats[e.key] = file.f32(tensor);
        }
      }
      return Lfm3AudioEncoder._(c, floats, matrices, int8Inputs);
    } catch (_) {
      for (final m in matrices.values) {
        m.free();
      }
      rethrow;
    } finally {
      file.close();
    }
  }

  final Lfm3AudioEncoderConfig config;
  final bool int8Inputs;
  final Map<String, Float32List> _floats;
  final Map<String, NativeQ4Matrix> _matrices;
  bool _closed = false;

  /// Owned [features.encoderTokens, config.projectionSize] projected rows.
  /// [capture] receives owned copies of intermediate rows for parity checks.
  Float32List encode(
    AudioFeatures features, {
    void Function(String stage, Float32List rows)? capture,
  }) {
    if (_closed) throw StateError('Audio encoder is closed');
    if (features.frames <= 0 ||
        features.melBins != config.melBins ||
        features.values.length != features.frames * config.melBins ||
        features.values.any((x) => !x.isFinite)) {
      throw ArgumentError('expected nonempty finite LFM3-Audio features');
    }
    void emit(String name, Float32List x) =>
        capture?.call(name, Float32List.fromList(x));
    final c = config;
    final h = c.hiddenSize;
    var x = _downsample(features);
    final n = features.encoderTokens;
    emit('downsample', x);
    for (var layer = 0; layer < c.numLayers; layer++) {
      final p = 'a.blk.$layer.';
      final a = _norm(x, '${p}ln1.weight', h);
      final mixer = c.convLayers[layer] ? _conv(a, p, n) : _attention(a, p, n);
      _add(x, mixer);
      final b = _norm(x, '${p}ln2.weight', h);
      final gate = _linear('${p}ffn_gate', b, n);
      final up = _linear('${p}ffn_up', b, n);
      for (var i = 0; i < gate.length; i++) {
        gate[i] = gate[i] / (1 + math.exp(-gate[i]));
        gate[i] *= up[i];
      }
      _add(x, _linear('${p}ffn_down', gate, n));
      emit('layer.$layer', x);
    }
    x = _norm(x, 'a.post_ln.weight', h);
    emit('embedding', x);
    x = _layerNorm(x, h);
    x = _linear('mm.a.mlp.1', x, n, bias: true);
    for (var i = 0; i < x.length; i++) {
      x[i] = 0.5 * x[i] * (1 + _erf(x[i] / math.sqrt2));
    }
    x = _linear('mm.a.mlp.3', x, n, bias: true);
    emit('projected', x);
    return x;
  }

  void close() {
    if (_closed) return;
    _closed = true;
    for (final m in _matrices.values) {
      m.free();
    }
    _matrices.clear();
    _floats.clear();
  }

  Float32List _linear(String name, Float32List x, int n, {bool bias = false}) {
    final matrix = _matrices['$name.weight']!;
    final out = Float32List(n * matrix.rows);
    matrix.apply(x, out, n, int8Inputs: int8Inputs);
    if (bias) {
      final b = _floats['$name.bias']!;
      for (var i = 0; i < out.length; i++) {
        out[i] += b[i % b.length];
      }
    }
    return out;
  }

  Float32List _norm(Float32List x, String name, int width) {
    final out = Float32List(x.length);
    for (var offset = 0; offset < x.length; offset += width) {
      rmsNorm(x, _floats[name]!, out, offset, offset, width, config.normEps);
    }
    return out;
  }

  Float32List _conv(Float32List a, String p, int n) {
    final h = config.hiddenSize;
    final bcx = _linear('${p}conv_pw1', a, n);
    final z = Float32List(n * h);
    for (var t = 0; t < n; t++) {
      for (var ch = 0; ch < h; ch++) {
        z[t * h + ch] = bcx[t * 3 * h + ch] * bcx[t * 3 * h + 2 * h + ch];
      }
    }
    final mixed = Float32List(n * h);
    final kernel = _floats['${p}conv_dw.weight']!;
    for (var t = 0; t < n; t++) {
      for (var ch = 0; ch < h; ch++) {
        var sum = 0.0;
        for (var k = 0; k < 3; k++) {
          final s = t + k - 2;
          if (s >= 0) {
            sum += z[s * h + ch] * kernel[ch * 3 + k];
          }
        }
        mixed[t * h + ch] = sum;
        mixed[t * h + ch] *= bcx[t * 3 * h + h + ch];
      }
    }
    return _linear('${p}conv_pw2', mixed, n);
  }

  Float32List _attention(Float32List a, String p, int n) {
    final c = config, h = c.hiddenSize, d = c.headDim;
    final kv = c.numKvHeads * d;
    final q = _norm(_linear('${p}attn_q', a, n), '${p}attn_q_norm.weight', d);
    final k = _norm(_linear('${p}attn_k', a, n), '${p}attn_k_norm.weight', d);
    final v = _linear('${p}attn_v', a, n);
    final rope = Rope(const RopeSpec(Lfm3AudioEncoderConfig.ropeTheta, 1), d);
    for (var t = 0; t < n; t++) {
      rope.setPosition(t);
      for (var head = 0; head < c.numHeads; head++) {
        rope.apply(q, t * h + head * d);
      }
      for (var head = 0; head < c.numKvHeads; head++) {
        rope.apply(k, t * kv + head * d);
      }
    }
    final output = Float32List(n * h), scores = Float32List(n);
    final scale = 1 / math.sqrt(d);
    for (var t = 0; t < n; t++) {
      final start = math.max(0, t - c.window + 1);
      for (var head = 0; head < c.numHeads; head++) {
        final kvHead = head ~/ (c.numHeads ~/ c.numKvHeads);
        var maximum = double.negativeInfinity;
        for (var s = start; s <= t; s++) {
          var dot = 0.0;
          for (var j = 0; j < d; j++) {
            dot += q[t * h + head * d + j] * k[s * kv + kvHead * d + j];
          }
          scores[s] = dot * scale;
          maximum = math.max(maximum, scores[s]);
        }
        var total = 0.0;
        for (var s = start; s <= t; s++) {
          scores[s] = math.exp(scores[s] - maximum);
          total += scores[s];
        }
        for (var s = start; s <= t; s++) {
          scores[s] /= total;
        }
        for (var j = 0; j < d; j++) {
          var sum = 0.0;
          for (var s = start; s <= t; s++) {
            sum += scores[s] * v[s * kv + kvHead * d + j];
          }
          output[t * h + head * d + j] = sum;
        }
      }
    }
    return _linear('${p}attn_out', output, n);
  }

  Float32List _downsample(AudioFeatures features) {
    var time = features.frames, freq = config.melBins;
    var x = features.values;
    for (final i in [0, 2, 5]) {
      final nt = (time + 1) ~/ 2, nf = (freq + 1) ~/ 2;
      final ch = config.convChannels;
      final w = _floats['a.conv1d.$i.weight']!,
          b = _floats['a.conv1d.$i.bias']!;
      final out = Float32List(ch * nf * nt);
      for (var c = 0; c < ch; c++) {
        for (var f = 0; f < nf; f++) {
          for (var t = 0; t < nt; t++) {
            var sum = 0.0;
            for (var y = 0; y < 3; y++) {
              final sy = 2 * f + y - 1;
              if (sy < 0 || sy >= freq) continue;
              for (var z = 0; z < 3; z++) {
                final st = 2 * t + z - 1;
                if (st >= 0 && st < time) {
                  sum +=
                      x[((i == 0 ? 0 : c) * freq + sy) * time + st] *
                      w[(c * 3 + y) * 3 + z];
                }
              }
            }
            out[(c * nf + f) * nt + t] = sum;
            out[(c * nf + f) * nt + t] += b[c];
          }
        }
      }
      x = i == 0 ? out : _pointwise(out, nf, nt, i == 2 ? 3 : 6);
      for (var j = 0; j < x.length; j++) {
        x[j] = x[j] / (1 + math.exp(-x[j]));
      }
      freq = nf;
      time = nt;
    }
    final width = config.convChannels * freq;
    final rows = Float32List(time * width);
    for (var t = 0; t < time; t++) {
      for (var j = 0; j < width; j++) {
        rows[t * width + j] = x[j * time + t];
      }
    }
    return _linear('a.pre_encode.out', rows, time, bias: true);
  }

  Float32List _pointwise(Float32List x, int freq, int time, int index) {
    final ch = config.convChannels, n = freq * time;
    final input = Float32List(n * ch), output = Float32List(n * ch);
    for (var r = 0; r < n; r++) {
      for (var c = 0; c < ch; c++) {
        input[r * ch + c] = x[c * n + r];
      }
    }
    gemm(_floats['a.conv1d.$index.weight']!, input, output, ch, ch, n);
    final bias = _floats['a.conv1d.$index.bias']!;
    final result = Float32List(x.length);
    for (var r = 0; r < n; r++) {
      for (var c = 0; c < ch; c++) {
        result[c * n + r] = output[r * ch + c] + bias[c];
      }
    }
    return result;
  }

  Float32List _layerNorm(Float32List x, int width) {
    final result = Float32List(x.length);
    final w = _floats['mm.a.mlp.0.weight']!, b = _floats['mm.a.mlp.0.bias']!;
    for (var offset = 0; offset < x.length; offset += width) {
      var mean = 0.0;
      for (var j = 0; j < width; j++) {
        mean += x[offset + j];
      }
      mean /= width;
      var variance = 0.0;
      for (var j = 0; j < width; j++) {
        final d = x[offset + j] - mean;
        variance += d * d;
      }
      final scale = 1 / math.sqrt(variance / width + 1e-5);
      for (var j = 0; j < width; j++) {
        result[offset + j] = (x[offset + j] - mean) * scale * w[j] + b[j];
      }
    }
    return result;
  }

  static void _add(Float32List x, Float32List y) {
    for (var i = 0; i < x.length; i++) {
      x[i] += y[i];
    }
  }
}

// Abramowitz-Stegun 7.1.26, as in the vision reference projector.
double _erf(double value) {
  final x = value.abs(), t = 1 / (1 + 0.3275911 * x);
  final polynomial =
      (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
                  0.284496736) *
              t +
          0.254829592) *
      t;
  final result = 1 - polynomial * math.exp(-x * x);
  return value < 0 ? -result : result;
}
