/// Q4_0 Depthformer and LiquidCodec decoder arithmetic, orchestrated in Dart.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'audio_gguf.dart';
import 'config.dart';
import 'gguf.dart';
import 'matmul.dart';
import 'native.dart';
import 'ops.dart';

final class _Matrix {
  _Matrix(this.rows, this.cols, this.native, this.values);
  final int rows, cols;
  final NativeQ4Matrix? native;
  final Float32List? values;
  Float32List apply(Float32List x, int n) {
    final out = Float32List(n * rows);
    if (native != null) {
      native!.apply(x, out, n, int8Inputs: false);
    } else {
      gemm(values!, x, out, rows, cols, n);
    }
    return out;
  }

  double at(int row, int col) =>
      native?.weightAt(row, col) ?? values![row * cols + col];
}

final class _Graph {
  _Graph(
    this.prefix,
    this.h,
    this.f,
    this.heads,
    this.kvHeads,
    this.d,
    this.layers,
    this.kernel,
    this.sandwich,
    this.window,
  );
  final String prefix;
  final int h, f, heads, kvHeads, d, layers, kernel, window;
  final bool sandwich;
  bool conv(int i) => prefix.contains('.wav.') && ![2, 4, 6].contains(i);
  Map<String, List<int>> get shapes {
    final out = <String, List<int>>{
      '${prefix}output_norm.weight': [h],
    };
    for (var i = 0; i < layers; i++) {
      final p = '${prefix}blk.$i.';
      out.addAll({
        '${p}ln1.weight': [h],
        '${p}ln2.weight': [h],
        '${p}ffn_gate.weight': [f, h],
        '${p}ffn_up.weight': [f, h],
        '${p}ffn_down.weight': [h, f],
        if (sandwich) '${p}attn_post_norm.weight': [h],
        if (sandwich) '${p}ffn_post_norm.weight': [h],
      });
      if (conv(i)) {
        out.addAll({
          '${p}conv_dw.weight': [h, kernel],
          '${p}conv_pw1.weight': [3 * h, h],
          '${p}conv_pw2.weight': [h, h],
        });
      } else {
        out.addAll({
          '${p}attn_q.weight': [heads * d, h],
          '${p}attn_k.weight': [kvHeads * d, h],
          '${p}attn_v.weight': [kvHeads * d, h],
          '${p}attn_out.weight': [h, heads * d],
          '${p}attn_gate.weight': [heads, h],
          '${p}attn_q_norm.weight': [d],
          '${p}attn_k_norm.weight': [d],
        });
      }
    }
    return out;
  }
}

final class _LayerState {
  _LayerState(int history) : conv = Float32List(history);
  Float32List conv, keys = Float32List(0), values = Float32List(0);
}

final class _GraphState {
  _GraphState(_Graph g)
    : layers = [
        for (var i = 0; i < g.layers; i++)
          _LayerState(g.conv(i) ? (g.kernel - 1) * g.h : 0),
      ];
  final List<_LayerState> layers;
  int position = 0;
}

/// Published generation weights only; does not load the ASR encoder.
/// All matrix blocks are retained exactly, with float activation inputs.
final class Lfm3AudioGenerator {
  Lfm3AudioGenerator._(
    this._depth,
    this._codec,
    this.codebooks,
    this.vocabSize,
    this.hiddenSize,
    this._quantDim,
    this.fftSize,
    this._floats,
    this._matrices,
    this._heads,
    this.variant,
    this.sampling,
  );

  factory Lfm3AudioGenerator.load(String mmproj) {
    if (!NativeKernels.isLoaded) NativeKernels.load(null);
    final file = GgufFile.open(mmproj);
    final matrices = <String, _Matrix>{};
    final floats = <String, Float32List>{};
    final heads = <_Matrix>[];
    try {
      if (file.metadata['clip.gen.audio.projector_type'] != 'lfm3a_gen') {
        throw const FormatException('not an LFM3 audio generator');
      }
      final variant = Lfm3SpeechVariant.of(file);
      final sampling = Lfm3SpeechSampling.fromGguf(file);
      List<int> dims(String name) =>
          file.tensors[name]?.dims.reversed.toList() ??
          (throw FormatException('missing $name'));
      int meta(String key) {
        final n = file.metadata['clip.gen.audio.$key'];
        if (n is! int || n <= 0) {
          throw FormatException('invalid generation $key');
        }
        return n;
      }

      final emb = dims('a.gen.code.embd.weight');
      if (emb.length != 3) {
        throw const FormatException('invalid codebook embedding');
      }
      final cb = emb[0],
          vocab = emb[1],
          dh = emb[2],
          hidden = meta('projection_dim');
      final depth = _Graph(
        'a.gen.code.',
        dh,
        meta('feed_forward_length'),
        meta('attention.head_count'),
        meta('attention.head_count_kv'),
        dims('a.gen.code.blk.0.attn_q_norm.weight')[0],
        meta('block_count'),
        7,
        false,
        0,
      );
      final ch = dims('a.gen.wav.tfm.output_norm.weight')[0];
      final cd = dims('a.gen.wav.tfm.blk.2.attn_q_norm.weight')[0];
      final ck = dims('a.gen.wav.tfm.blk.2.attn_k.weight')[0];
      final codec = _Graph(
        'a.gen.wav.tfm.',
        ch,
        dims('a.gen.wav.tfm.blk.0.ffn_gate.weight')[0],
        ch ~/ cd,
        ck ~/ cd,
        cd,
        8,
        7,
        true,
        30,
      );
      final qdim = dims('a.gen.wav.quant.first.out_proj.bias')[0];
      final bins = dims('a.gen.wav.tfm.out_proj.bias')[0] ~/ 2;
      final fft = 2 * (bins - 1);
      if (cb != 12 ||
          vocab < 2 ||
          dh != meta('embedding_length') ||
          dh % 32 != 0 ||
          hidden % 32 != 0 ||
          depth.heads * depth.d != dh ||
          depth.heads % depth.kvHeads != 0 ||
          depth.d.isOdd ||
          depth.layers != 3 ||
          ch % 32 != 0 ||
          codec.heads % codec.kvHeads != 0 ||
          cd.isOdd ||
          qdim % 32 != 0 ||
          fft < 20 ||
          fft % 5 != 0 ||
          ((fft ~/ 5) & ((fft ~/ 5) - 1)) != 0) {
        throw const FormatException('unsupported generation dimensions');
      }
      final shapes = <String, List<int>>{
        ...depth.shapes,
        ...codec.shapes,
        'a.gen.code.embd.weight': [cb, vocab, dh],
        'a.gen.code.head_norm.weight': [cb, dh],
        'a.gen.code.mask_embd': [dh],
        'a.gen.code.proj_in.weight': [cb * dh, hidden],
        'a.gen.code.proj_in.bias': [cb * dh],
        'a.gen.code.out_embd.weight': [cb * vocab, hidden],
        'a.gen.wav.quant.first.codebook.weight': [vocab - 1, 4],
        'a.gen.wav.quant.first.out_proj.weight': [qdim, cb * 4],
        'a.gen.wav.quant.first.out_proj.bias': [qdim],
        'a.gen.wav.tfm.in_proj.weight': [ch, qdim],
        'a.gen.wav.tfm.in_proj.bias': [ch],
        'a.gen.wav.tfm.out_proj.weight': [2 * bins, ch],
        'a.gen.wav.tfm.out_proj.bias': [2 * bins],
      };
      bool vector(String name) =>
          name.endsWith('.bias') ||
          name.contains('norm') ||
          name.contains('.ln') ||
          name.endsWith('mask_embd') ||
          name.endsWith('conv_dw.weight') ||
          name.endsWith('first.codebook.weight');
      for (final e in shapes.entries) {
        final t = file.tensors[e.key];
        if (t == null ||
            t.dims.reversed.join(',') != e.value.join(',') ||
            (vector(e.key)
                ? t.type != GgufType.f32
                : ![GgufType.f32, GgufType.q4_0].contains(t.type))) {
          throw FormatException('unsupported generation tensor ${e.key}');
        }
      }
      if (file.tensors.keys.any(
        (n) => n.startsWith('a.gen.') && !shapes.containsKey(n),
      )) {
        throw const FormatException('unexpected generation tensor');
      }
      _Matrix matrix(GgufTensor t, int rows, int cols, [int byteOffset = 0]) {
        if (t.type == GgufType.q4_0) {
          final bytes = rows * cols ~/ 32 * 18;
          return _Matrix(
            rows,
            cols,
            NativeQ4Matrix.fromGgufBlocks(
              Uint8List.sublistView(
                file.bytes(t),
                byteOffset,
                byteOffset + bytes,
              ),
              rows,
              cols,
            ),
            null,
          );
        }
        final data = file.f32(t);
        return _Matrix(
          rows,
          cols,
          null,
          Float32List.fromList(
            data.sublist(byteOffset ~/ 4, byteOffset ~/ 4 + rows * cols),
          ),
        );
      }

      for (final e in shapes.entries) {
        final t = file.tensors[e.key]!;
        if (vector(e.key)) {
          floats[e.key] = file.f32(t);
        } else if (e.key == 'a.gen.code.embd.weight') {
          final per = vocab * dh ~/ t.type.blockValues * t.type.blockBytes;
          for (var i = 0; i < cb; i++) {
            heads.add(matrix(t, vocab, dh, i * per));
          }
        } else {
          matrices[e.key] = matrix(t, e.value[0], e.value[1]);
        }
      }
      return Lfm3AudioGenerator._(
        depth,
        codec,
        cb,
        vocab,
        hidden,
        qdim,
        fft,
        floats,
        matrices,
        heads,
        variant,
        sampling,
      );
    } catch (_) {
      for (final m in [...matrices.values, ...heads]) {
        m.native?.free();
      }
      rethrow;
    } finally {
      file.close();
    }
  }

  final _Graph _depth, _codec;
  final int codebooks, vocabSize, hiddenSize, _quantDim, fftSize;

  /// Which checkpoint's prompt conventions the generated speech expects.
  final Lfm3SpeechVariant variant;

  /// The checkpoint's published speech sampling defaults.
  final Lfm3SpeechSampling sampling;
  int get eosCode => vocabSize - 1;
  int get sampleRate => 24000;
  int get frameSamples => fftSize ~/ 4 * 6;
  final Map<String, Float32List> _floats;
  final Map<String, _Matrix> _matrices;
  final List<_Matrix> _heads;
  final _streams = <LiquidCodecStream>{};
  bool _closed = false;
  void _open() {
    if (_closed) throw StateError('Audio generator is closed');
  }

  void close() {
    if (_closed) return;
    _closed = true;
    for (final stream in _streams.toList()) {
      stream.close();
    }
    for (final m in [..._matrices.values, ..._heads]) {
      m.native?.free();
    }
    _matrices.clear();
    _floats.clear();
    _heads.clear();
  }

  Float32List _linear(String name, Float32List x, int n, {bool bias = false}) {
    final m = _matrices['$name.weight']!;
    final out = m.apply(x, n);
    if (bias) {
      final b = _floats['$name.bias']!;
      for (var i = 0; i < out.length; i++) {
        out[i] += b[i % b.length];
      }
    }
    return out;
  }

  Float32List _norm(Float32List x, Float32List w, int h) {
    final out = Float32List(x.length);
    for (var at = 0; at < x.length; at += h) {
      rmsNorm(x, w, out, at, at, h, 1e-5);
    }
    return out;
  }

  static void _add(Float32List x, Float32List y) {
    for (var i = 0; i < x.length; i++) {
      x[i] += y[i];
    }
  }

  Float32List _graph(
    _Graph g,
    Float32List rows,
    int n, {
    _GraphState? state,
    void Function(String, Float32List)? capture,
  }) {
    var x = Float32List.fromList(rows);
    for (var layer = 0; layer < g.layers; layer++) {
      final p = '${g.prefix}blk.$layer.';
      final a = _norm(x, _floats['${p}ln1.weight']!, g.h);
      final s = state?.layers[layer];
      Float32List mixed;
      if (g.conv(layer)) {
        final bcx = _linear('${p}conv_pw1', a, n);
        final history = s?.conv ?? Float32List((g.kernel - 1) * g.h);
        final z = Float32List(history.length + n * g.h)..setAll(0, history);
        for (var t = 0; t < n; t++) {
          for (var c = 0; c < g.h; c++) {
            z[history.length + t * g.h + c] =
                bcx[t * 3 * g.h + c] * bcx[t * 3 * g.h + 2 * g.h + c];
          }
        }
        mixed = Float32List(n * g.h);
        final kernel = _floats['${p}conv_dw.weight']!;
        for (var t = 0; t < n; t++) {
          for (var c = 0; c < g.h; c++) {
            var sum = 0.0;
            for (var k = 0; k < g.kernel; k++) {
              sum += z[(t + k) * g.h + c] * kernel[c * g.kernel + k];
            }
            mixed[t * g.h + c] = sum;
            mixed[t * g.h + c] *= bcx[t * 3 * g.h + g.h + c];
          }
        }
        mixed = _linear('${p}conv_pw2', mixed, n);
        if (s != null) s.conv = z.sublist(z.length - history.length);
      } else {
        final q = _norm(
          _linear('${p}attn_q', a, n),
          _floats['${p}attn_q_norm.weight']!,
          g.d,
        );
        final k = _norm(
          _linear('${p}attn_k', a, n),
          _floats['${p}attn_k_norm.weight']!,
          g.d,
        );
        final v = _linear('${p}attn_v', a, n);
        final kv = g.kvHeads * g.d;
        final rope = Rope(const RopeSpec(1000000, 1), g.d);
        for (var t = 0; t < n; t++) {
          rope.setPosition((state?.position ?? 0) + t);
          for (var h = 0; h < g.heads; h++) {
            rope.apply(q, t * g.h + h * g.d);
          }
          for (var h = 0; h < g.kvHeads; h++) {
            rope.apply(k, t * kv + h * g.d);
          }
        }
        final prefix = (s?.keys.length ?? 0) ~/ kv;
        final keys = Float32List(prefix * kv + k.length)
          ..setAll(0, s?.keys ?? const [])
          ..setAll(prefix * kv, k);
        final vals = Float32List(prefix * kv + v.length)
          ..setAll(0, s?.values ?? const [])
          ..setAll(prefix * kv, v);
        final gate = _linear('${p}attn_gate', a, n);
        final attention = Float32List(n * g.h),
            scores = Float32List(prefix + n);
        for (var t = 0; t < n; t++) {
          final end = g.window == 0 ? n - 1 : prefix + t;
          final begin = g.window == 0 ? 0 : math.max(0, end - g.window + 1);
          for (var h = 0; h < g.heads; h++) {
            final kh = h ~/ (g.heads ~/ g.kvHeads);
            var maximum = double.negativeInfinity;
            for (var pos = begin; pos <= end; pos++) {
              var dot = 0.0;
              for (var j = 0; j < g.d; j++) {
                dot += q[t * g.h + h * g.d + j] * keys[pos * kv + kh * g.d + j];
              }
              scores[pos] = dot / math.sqrt(g.d);
              maximum = math.max(maximum, scores[pos]);
            }
            var total = 0.0;
            for (var pos = begin; pos <= end; pos++) {
              scores[pos] = math.exp(scores[pos] - maximum);
              total += scores[pos];
            }
            for (var pos = begin; pos <= end; pos++) {
              scores[pos] /= total;
            }
            final gain = 1 / (1 + math.exp(-gate[t * g.heads + h]));
            for (var j = 0; j < g.d; j++) {
              var sum = 0.0;
              for (var pos = begin; pos <= end; pos++) {
                sum += scores[pos] * vals[pos * kv + kh * g.d + j];
              }
              attention[t * g.h + h * g.d + j] = sum;
              attention[t * g.h + h * g.d + j] *= gain;
            }
          }
        }
        mixed = _linear('${p}attn_out', attention, n);
        if (s != null) {
          final keep = math.min(g.window - 1, prefix + n) * kv;
          s.keys = keys.sublist(keys.length - keep);
          s.values = vals.sublist(vals.length - keep);
        }
      }
      if (g.sandwich) {
        mixed = _norm(mixed, _floats['${p}attn_post_norm.weight']!, g.h);
      }
      _add(x, mixed);
      final b = _norm(x, _floats['${p}ln2.weight']!, g.h);
      final gate = _linear('${p}ffn_gate', b, n),
          up = _linear('${p}ffn_up', b, n);
      for (var i = 0; i < gate.length; i++) {
        gate[i] = gate[i] / (1 + math.exp(-gate[i]));
        gate[i] *= up[i];
      }
      var ff = _linear('${p}ffn_down', gate, n);
      if (g.sandwich) ff = _norm(ff, _floats['${p}ffn_post_norm.weight']!, g.h);
      _add(x, ff);
      capture?.call('layer.$layer', Float32List.fromList(x));
    }
    if (state != null) state.position += n;
    return _norm(x, _floats['${g.prefix}output_norm.weight']!, g.h);
  }

  /// One MaskGIT pass: -1 marks a masked codebook, others are known codes.
  /// Returns [codebooks, vocabSize] logits. No random sampling occurs here.
  Float32List depthLogits(
    Float32List hidden,
    List<int> codes, {
    void Function(String, Float32List)? capture,
  }) {
    _open();
    if (hidden.length != hiddenSize ||
        hidden.any((x) => !x.isFinite) ||
        codes.length != codebooks ||
        codes.any((c) => c < -1 || c >= vocabSize)) {
      throw ArgumentError('invalid depth input');
    }
    final h = _depth.h;
    final input = _linear('a.gen.code.proj_in', hidden, 1, bias: true);
    for (var cb = 0; cb < codebooks; cb++) {
      for (var j = 0; j < h; j++) {
        input[cb * h + j] += codes[cb] < 0
            ? _floats['a.gen.code.mask_embd']![j]
            : _heads[cb].at(codes[cb], j);
      }
    }
    capture?.call('input', Float32List.fromList(input));
    final output = _graph(_depth, input, codebooks, capture: capture);
    capture?.call('embedding', Float32List.fromList(output));
    final logits = Float32List(codebooks * vocabSize),
        norms = _floats['a.gen.code.head_norm.weight']!;
    for (var cb = 0; cb < codebooks; cb++) {
      final row = Float32List.fromList(output.sublist(cb * h, (cb + 1) * h));
      final norm = _norm(
        row,
        Float32List.sublistView(norms, cb * h, (cb + 1) * h),
        h,
      );
      logits.setAll(cb * vocabSize, _heads[cb].apply(norm, 1));
    }
    capture?.call('logits', Float32List.fromList(logits));
    return logits;
  }

  /// Sum of published per-codebook feedback embeddings for the language model.
  Float32List feedback(List<int> codes) {
    _open();
    if (codes.length != codebooks ||
        codes.any((c) => c < 0 || c >= vocabSize)) {
      throw ArgumentError('invalid speech frame');
    }
    final out = Float32List(hiddenSize),
        m = _matrices['a.gen.code.out_embd.weight']!;
    for (var cb = 0; cb < codebooks; cb++) {
      for (var j = 0; j < hiddenSize; j++) {
        out[j] += m.at(cb * vocabSize + codes[cb], j);
      }
    }
    return out;
  }

  /// Three passes reveal four randomly ordered codebooks each. The penalty
  /// covers the codes of the last [repetitionWindow] frames of [history].
  /// Omitted options use [sampling]; a null top-k there samples every code.
  /// Random seeds are Dart-specific.
  List<int> sampleFrame(
    Float32List hidden,
    math.Random random, {
    double? temperature,
    int? topK,
    double? repetitionPenalty,
    int? repetitionWindow,
    List<List<int>> history = const [],
  }) {
    return _sampleFrame(
      hidden,
      random,
      temperature ?? sampling.temperature,
      topK ?? sampling.topK ?? vocabSize,
      repetitionPenalty ?? sampling.repetitionPenalty,
      repetitionWindow ?? sampling.repetitionWindow,
      history,
    );
  }

  List<int> _sampleFrame(
    Float32List hidden,
    math.Random random,
    double temperature,
    int topK,
    double repetitionPenalty,
    int repetitionWindow,
    List<List<int>> history,
  ) {
    if (repetitionWindow < 0) {
      throw ArgumentError('invalid audio sampling options');
    }
    if (history.length > repetitionWindow) {
      history = history.sublist(history.length - repetitionWindow);
    }
    if (history.any(
          (frame) =>
              frame.length != codebooks ||
              frame.any((c) => c < 0 || c >= vocabSize),
        ) ||
        !temperature.isFinite ||
        temperature < 0 ||
        topK < 1 ||
        !repetitionPenalty.isFinite ||
        repetitionPenalty < 1) {
      throw ArgumentError('invalid audio sampling options');
    }
    final order = List.generate(codebooks, (i) => i)..shuffle(random);
    final codes = List.filled(codebooks, -1);
    for (var pass = 0; pass < 3; pass++) {
      final logits = depthLogits(hidden, codes);
      for (final cb in order.skip(pass * 4).take(4)) {
        final seen = {for (final frame in history) frame[cb]};
        final scores = List<double>.generate(vocabSize, (v) {
          var score = logits[cb * vocabSize + v];
          if (seen.contains(v)) {
            score = score < 0
                ? score * repetitionPenalty
                : score / repetitionPenalty;
          }
          return score;
        });
        if (scores.any((v) => !v.isFinite)) {
          throw StateError('nonfinite depth logits');
        }
        final sorted = List.generate(vocabSize, (i) => i)
          ..sort((a, b) {
            final cmp = scores[b].compareTo(scores[a]);
            return cmp == 0 ? a.compareTo(b) : cmp;
          });
        if (temperature == 0) {
          codes[cb] = sorted.first;
          continue;
        }
        final limit = math.min(topK, vocabSize),
            maxScore = scores[sorted.first];
        final probs = [
          for (final i in sorted.take(limit))
            math.exp((scores[i] - maxScore) / temperature),
        ];
        var pick = random.nextDouble() * probs.fold(0.0, (a, b) => a + b);
        codes[cb] = sorted[limit - 1];
        for (var i = 0; i < limit; i++) {
          pick -= probs[i];
          if (pick <= 0) {
            codes[cb] = sorted[i];
            break;
          }
        }
      }
    }
    return codes;
  }

  LiquidCodecStream startCodecStream() {
    _open();
    final stream = LiquidCodecStream._(this);
    _streams.add(stream);
    return stream;
  }
}

/// Stateful code-to-waveform decoder. Causal LFM3 caches and the 960-sample
/// overlap-add tail are retained; emitted audio is owned by the caller.
final class LiquidCodecStream {
  LiquidCodecStream._(this._owner)
    : _state = _GraphState(_owner._codec),
      _ifft = _InverseRealFft(_owner.fftSize);
  final Lfm3AudioGenerator _owner;
  final _GraphState _state;
  final _InverseRealFft _ifft;
  Float32List _audio = Float32List(0), _envelope = Float32List(0);
  bool _started = false, _closed = false;
  void close() {
    _closed = true;
    _audio = Float32List(0);
    _envelope = Float32List(0);
    _state.layers.clear();
    _owner._streams.remove(this);
  }

  Float32List add(
    List<int> frameMajorCodes, {
    void Function(String, Float32List)? capture,
  }) {
    if (_closed || _owner._closed) throw StateError('Codec stream is closed');
    if (frameMajorCodes.length % _owner.codebooks != 0 ||
        frameMajorCodes.any((c) => c < 0 || c >= _owner.eosCode)) {
      throw ArgumentError('invalid LiquidCodec codes (EOS is not audio)');
    }
    if (frameMajorCodes.isEmpty) return Float32List(0);
    try {
      final n = frameMajorCodes.length ~/ _owner.codebooks,
          cb = _owner.codebooks;
      final latents = Float32List(n * cb * 4),
          table = _owner._floats['a.gen.wav.quant.first.codebook.weight']!;
      for (var i = 0; i < frameMajorCodes.length; i++) {
        latents.setRange(i * 4, i * 4 + 4, table, frameMajorCodes[i] * 4);
      }
      var x = _owner._linear(
        'a.gen.wav.quant.first.out_proj',
        latents,
        n,
        bias: true,
      );
      if (x.length != n * _owner._quantDim) {
        throw StateError('invalid codec projection');
      }
      x = _owner._linear('a.gen.wav.tfm.in_proj', x, n, bias: true);
      final h = _owner._codec.h;
      final expanded = Float32List(n * 6 * h);
      for (var t = 0; t < n * 6; t++) {
        expanded.setRange(t * h, (t + 1) * h, x, (t ~/ 6) * h);
      }
      capture?.call('input', Float32List.fromList(expanded));
      x = _owner._graph(
        _owner._codec,
        expanded,
        n * 6,
        state: _state,
        capture: capture,
      );
      capture?.call('embedding', Float32List.fromList(x));
      final polar = _owner._linear(
        'a.gen.wav.tfm.out_proj',
        x,
        n * 6,
        bias: true,
      );
      capture?.call('polar', Float32List.fromList(polar));
      return _overlap(polar, n * 6, false);
    } catch (_) {
      close();
      rethrow;
    }
  }

  Float32List finish() {
    if (_closed || _owner._closed) throw StateError('Codec stream is closed');
    try {
      return _started ? _overlap(Float32List(0), 0, true) : Float32List(0);
    } finally {
      close();
    }
  }

  Float32List _overlap(Float32List polar, int frames, bool finalChunk) {
    final fft = _owner.fftSize,
        hop = fft ~/ 4,
        trim = (fft - hop) ~/ 2,
        tail = fft - hop,
        bins = fft ~/ 2 + 1;
    final length = frames == 0 ? tail : (frames - 1) * hop + fft;
    final audio = Float32List(length), envelope = Float32List(length);
    for (var t = 0; t < frames; t++) {
      final row = _ifft.polarToFrame(
        Float32List.sublistView(polar, t * 2 * bins, (t + 1) * 2 * bins),
      );
      for (var j = 0; j < fft; j++) {
        audio[t * hop + j] += row[j];
        envelope[t * hop + j] += _ifft.window[j] * _ifft.window[j];
      }
    }
    for (var i = 0; i < _audio.length; i++) {
      audio[i] += _audio[i];
      envelope[i] += _envelope[i];
    }
    final begin = _started ? 0 : trim;
    _started = true;
    final end = finalChunk ? length - trim : length - tail;
    final out = Float32List(end - begin);
    for (var i = begin; i < end; i++) {
      out[i - begin] = audio[i] / envelope[i];
      if (!out[i - begin].isFinite) {
        throw StateError('nonfinite codec waveform');
      }
    }
    if (!finalChunk) {
      _audio = audio.sublist(length - tail);
      _envelope = envelope.sublist(length - tail);
    }
    return out;
  }
}

/// Mixed radix inverse FFT (radix two and a final radix five), no native FFT
/// dependency. The supported published transform is 1280 = 256 * 5.
final class _InverseRealFft {
  _InverseRealFft(this.n)
    : window = Float32List.fromList([
        for (var i = 0; i < n; i++) 0.5 - 0.5 * math.cos(2 * math.pi * i / n),
      ]),
      cos = [for (var i = 0; i < n; i++) math.cos(2 * math.pi * i / n)],
      sin = [for (var i = 0; i < n; i++) math.sin(2 * math.pi * i / n)];
  final int n;
  final Float32List window;
  final List<double> cos, sin;
  Float32List polarToFrame(Float32List polar) {
    final bins = n ~/ 2 + 1;
    final r = Float64List(n),
        im = Float64List(n),
        outR = Float64List(n),
        outI = Float64List(n);
    final f = Float32List(3);
    for (var k = 0; k < bins; k++) {
      f[0] = math.exp(polar[k]);
      f[1] = f[0] * math.cos(polar[bins + k]);
      f[2] = f[0] * math.sin(polar[bins + k]);
      r[k] = f[1];
      im[k] = f[2];
      if (k > 0 && k < n ~/ 2) {
        r[n - k] = f[1];
        im[n - k] = -f[2];
      }
    }
    // DC and Nyquist imaginary components do not contribute to irfft.
    im[0] = 0;
    im[n ~/ 2] = 0;
    void fft(int offset, int stride, int size, int dst) {
      if (size == 1) {
        outR[dst] = r[offset];
        outI[dst] = im[offset];
        return;
      }
      if (size.isEven) {
        final half = size ~/ 2;
        fft(offset, stride * 2, half, dst);
        fft(offset + stride, stride * 2, half, dst + half);
        for (var k = 0; k < half; k++) {
          final tw = k * (n ~/ size), ar = outR[dst + k], ai = outI[dst + k];
          final br =
              outR[dst + half + k] * cos[tw] - outI[dst + half + k] * sin[tw];
          final bi =
              outR[dst + half + k] * sin[tw] + outI[dst + half + k] * cos[tw];
          outR[dst + k] = ar + br;
          outI[dst + k] = ai + bi;
          outR[dst + half + k] = ar - br;
          outI[dst + half + k] = ai - bi;
        }
      } else {
        if (size != 5) throw const FormatException('unsupported FFT size');
        for (var k = 0; k < 5; k++) {
          var real = 0.0, imag = 0.0;
          for (var j = 0; j < 5; j++) {
            final tw = (k * j * (n ~/ 5)) % n;
            real +=
                r[offset + j * stride] * cos[tw] -
                im[offset + j * stride] * sin[tw];
            imag +=
                r[offset + j * stride] * sin[tw] +
                im[offset + j * stride] * cos[tw];
          }
          outR[dst + k] = real;
          outI[dst + k] = imag;
        }
      }
    }

    fft(0, 1, n, 0);
    return Float32List.fromList([
      for (var i = 0; i < n; i++) outR[i] / n * window[i],
    ]);
  }
}
