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

/// Synchronous CPU lab implementation for complete clips or streaming input.
///
/// Q4 weights retain their published blocks. Convolutions and norms stay F32.
/// Matmul uses float inputs by default to isolate weight quantization from
/// activation quantization. Run in a worker isolate for interactive apps.
/// [startStream] keeps bounded per-stream context. Audio generation is separate.
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
  final _streams = <Lfm3AudioEncoderStream>{};
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
    return _encodeRows(
      _downsample(features),
      features.encoderTokens,
      capture: capture,
    );
  }

  /// Independent causal state sharing this encoder's weights. The downsampler
  /// needs seven future feature frames; only [Lfm3AudioEncoderStream.finish]
  /// pads the tail. Closing this encoder also closes every active stream.
  Lfm3AudioEncoderStream startStream() {
    if (_closed) throw StateError('Audio encoder is closed');
    final stream = Lfm3AudioEncoderStream._(this);
    _streams.add(stream);
    return stream;
  }

  Float32List _encodeRows(
    Float32List x,
    int n, {
    List<_AudioLayerState>? states,
    int position = 0,
    void Function(String stage, Float32List rows)? capture,
  }) {
    void emit(String name, Float32List x) =>
        capture?.call(name, Float32List.fromList(x));
    final c = config;
    final h = c.hiddenSize;
    emit('downsample', x);
    for (var layer = 0; layer < c.numLayers; layer++) {
      final p = 'a.blk.$layer.';
      final a = _norm(x, '${p}ln1.weight', h);
      final mixer = c.convLayers[layer]
          ? _conv(a, p, n, states?[layer])
          : _attention(a, p, n, states?[layer], position);
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
    if (_streams.any((s) => s._busy)) {
      throw StateError('Audio encoder has a busy stream');
    }
    _closed = true;
    for (final stream in _streams.toList()) {
      stream.close();
    }
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

  Float32List _conv(Float32List a, String p, int n, _AudioLayerState? state) {
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
          } else if (state != null) {
            sum += state.conv[(s + 2) * h + ch] * kernel[ch * 3 + k];
          }
        }
        mixed[t * h + ch] = sum;
        mixed[t * h + ch] *= bcx[t * 3 * h + h + ch];
      }
    }
    final result = _linear('${p}conv_pw2', mixed, n);
    if (state != null) {
      final combined = Float32List(2 * h + z.length)
        ..setAll(0, state.conv)
        ..setAll(2 * h, z);
      state.conv = combined.sublist(combined.length - 2 * h);
    }
    return result;
  }

  Float32List _attention(
    Float32List a,
    String p,
    int n,
    _AudioLayerState? state,
    int position,
  ) {
    final c = config, h = c.hiddenSize, d = c.headDim;
    final kv = c.numKvHeads * d;
    final q = _norm(_linear('${p}attn_q', a, n), '${p}attn_q_norm.weight', d);
    final k = _norm(_linear('${p}attn_k', a, n), '${p}attn_k_norm.weight', d);
    final v = _linear('${p}attn_v', a, n);
    final rope = Rope(const RopeSpec(Lfm3AudioEncoderConfig.ropeTheta, 1), d);
    for (var t = 0; t < n; t++) {
      rope.setPosition(position + t);
      for (var head = 0; head < c.numHeads; head++) {
        rope.apply(q, t * h + head * d);
      }
      for (var head = 0; head < c.numKvHeads; head++) {
        rope.apply(k, t * kv + head * d);
      }
    }
    final prefix = (state?.keys.length ?? 0) ~/ kv;
    final keys = state == null
        ? k
        : (Float32List(state.keys.length + k.length)
            ..setAll(0, state.keys)
            ..setAll(state.keys.length, k));
    final values = state == null
        ? v
        : (Float32List(state.values.length + v.length)
            ..setAll(0, state.values)
            ..setAll(state.values.length, v));
    final output = Float32List(n * h), scores = Float32List(prefix + n);
    final scale = 1 / math.sqrt(d);
    for (var t = 0; t < n; t++) {
      final end = prefix + t;
      final start = math.max(0, end - c.window + 1);
      for (var head = 0; head < c.numHeads; head++) {
        final kvHead = head ~/ (c.numHeads ~/ c.numKvHeads);
        var maximum = double.negativeInfinity;
        for (var s = start; s <= end; s++) {
          var dot = 0.0;
          for (var j = 0; j < d; j++) {
            dot += q[t * h + head * d + j] * keys[s * kv + kvHead * d + j];
          }
          scores[s] = dot * scale;
          maximum = math.max(maximum, scores[s]);
        }
        var total = 0.0;
        for (var s = start; s <= end; s++) {
          scores[s] = math.exp(scores[s] - maximum);
          total += scores[s];
        }
        for (var s = start; s <= end; s++) {
          scores[s] /= total;
        }
        for (var j = 0; j < d; j++) {
          var sum = 0.0;
          for (var s = start; s <= end; s++) {
            sum += scores[s] * values[s * kv + kvHead * d + j];
          }
          output[t * h + head * d + j] = sum;
        }
      }
    }
    final result = _linear('${p}attn_out', output, n);
    if (state != null) {
      final keep = math.min(c.window - 1, prefix + n) * kv;
      state.keys = keys.sublist(keys.length - keep);
      state.values = values.sublist(values.length - keep);
    }
    return result;
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

final class _AudioLayerState {
  _AudioLayerState(int hidden, bool convolution)
    : conv = Float32List(convolution ? 2 * hidden : 0);
  Float32List conv;
  Float32List keys = Float32List(0), values = Float32List(0);
}

/// A single feature stream with bounded LFM2 convolution/KV state.
///
/// [add] returns newly completed projected rows in time order, possibly empty.
/// Across calls, retains at most 15 spectrogram frames, two gated-input rows
/// per convolution layer and window-1 KV rows per attention layer. Temporary
/// work scales with the current chunk, not elapsed recording duration.
/// Invalid input leaves state intact. A computation/capture failure terminates
/// the stream because some layer caches may already have advanced.
final class Lfm3AudioEncoderStream {
  Lfm3AudioEncoderStream._(this._encoder)
    : _states = [
        for (final conv in _encoder.config.convLayers)
          _AudioLayerState(_encoder.config.hiddenSize, conv),
      ];
  final Lfm3AudioEncoder _encoder;
  final List<_AudioLayerState> _states;
  Float32List _pending = Float32List(0);
  int _pendingFrames = 0, _offset = 0, _frames = 0, _tokens = 0;
  bool _finished = false, _busy = false;

  int get frames => _frames;
  int get tokens => _tokens;
  int get retainedFeatureFrames => _pendingFrames;
  bool get isFinished => _finished;

  /// Persistent signal/cache storage only; excludes shared weights and scratch.
  int get cachedStateBytes =>
      _pending.lengthInBytes +
      _states.fold(
        0,
        (n, s) =>
            n +
            s.conv.lengthInBytes +
            s.keys.lengthInBytes +
            s.values.lengthInBytes,
      );

  Float32List add(
    AudioFeatures features, {
    void Function(String stage, Float32List rows)? capture,
  }) => _add(features, false, capture);

  Float32List finish({
    void Function(String stage, Float32List rows)? capture,
  }) =>
      _add(AudioFeatures.fromLogMel(Float32List(0), frames: 0), true, capture);

  void close() {
    if (_busy) throw StateError('Audio encoder stream is busy');
    _release();
  }

  void _release() {
    _finished = true;
    _pending = Float32List(0);
    _pendingFrames = 0;
    _states.clear();
    _encoder._streams.remove(this);
  }

  Float32List _add(
    AudioFeatures features,
    bool finalChunk,
    void Function(String stage, Float32List rows)? capture,
  ) {
    if (_finished || _encoder._closed) {
      throw StateError('Audio encoder stream is closed');
    }
    if (_busy) throw StateError('Audio encoder stream is busy');
    if (features.frames < 0 ||
        features.values.length != 128 * features.frames ||
        features.values.any((x) => !x.isFinite)) {
      throw ArgumentError('expected finite [128, frames] audio features');
    }
    _busy = true;
    try {
      final totalFrames = _frames + features.frames;
      final totalTokens = (totalFrames + (finalChunk ? 7 : 0)) ~/ 8;
      final count = totalTokens - _tokens;
      final bufferFrames = _pendingFrames + features.frames;
      final buffer = Float32List(128 * bufferFrames);
      for (var m = 0; m < 128; m++) {
        buffer.setRange(
          m * bufferFrames,
          m * bufferFrames + _pendingFrames,
          _pending,
          m * _pendingFrames,
        );
        buffer.setRange(
          m * bufferFrames + _pendingFrames,
          (m + 1) * bufferFrames,
          features.values,
          m * features.frames,
        );
      }
      var projected = Float32List(0);
      if (count > 0) {
        final all = _encoder._downsample(
          AudioFeatures.fromLogMel(buffer, frames: bufferFrames),
        );
        final first = _tokens - _offset ~/ 8;
        final h = _encoder.config.hiddenSize;
        final rows = all.sublist(first * h, (first + count) * h);
        projected = _encoder._encodeRows(
          rows,
          count,
          states: _states,
          position: _tokens,
          capture: capture,
        );
      }
      final offset = math.max(0, (totalTokens - 1) * 8);
      final keep = finalChunk ? 0 : totalFrames - offset;
      final pending = Float32List(128 * keep);
      for (var m = 0; m < 128; m++) {
        pending.setRange(
          m * keep,
          (m + 1) * keep,
          buffer,
          m * bufferFrames + offset - _offset,
        );
      }
      _pending = pending;
      _pendingFrames = keep;
      _offset = offset;
      _frames = totalFrames;
      _tokens = totalTokens;
      if (finalChunk) _release();
      return projected;
    } catch (_) {
      _release();
      rethrow;
    } finally {
      _busy = false;
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
