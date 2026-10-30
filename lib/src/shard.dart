/// One shard of LFM3's layers: a slice of every mixer and FFN, with the
/// state that slice needs.
///
/// Tensor parallelism in the Megatron style. Shard s of S owns
///
/// * conv layers: channels [c0, c1): the B, C and X rows of in_proj, the
///   depthwise kernel and conv history of those channels, and the matching
///   columns of out_proj;
/// * attention layers: query heads [h0, h1) with the KV heads they read
///   (shared KV heads are computed by every shard that needs them), their
///   rows of q/k/v/gate, their KV cache, and the matching columns of o_proj;
/// * FFN: intermediate units [f0, f1): those rows of w1 and w3 and the
///   matching columns of w2;
/// * the LM head: vocabulary rows [v0, v1).
///
/// Every mixer and FFN call returns a partial [n, H] output. The partials of
/// all shards add up to the layer output, so one round trip per mixer and
/// one per FFN suffice; the main isolate keeps only the residual stream and
/// the norms. With a single shard owning everything, the arithmetic is
/// exactly that of an unsharded model.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'config.dart';
import 'linear.dart';
import 'native.dart';
import 'ops.dart';
import 'q8_cache.dart';
import 'quant.dart';
import 'safetensors.dart';

final class _ConvShard {
  _ConvShard(
    this.c0,
    this.c1,
    this.inProj,
    this.kernel,
    this.outProj,
    this.history,
  );

  final int c0, c1;

  /// [3 * width, H]: rows B, then C, then X of this shard's channels.
  final Linear inProj;

  /// [width, K] depthwise causal kernel; column K-1 weights the newest input.
  final Float32List kernel;

  /// [H, width]: out_proj columns of this shard's channels.
  final Linear outProj;

  /// The last K-1 gated inputs z of these channels, oldest first.
  final Float32List history;

  int get width => c1 - c0;
}

final class _AttentionShard {
  _AttentionShard({
    required this.h0,
    required this.h1,
    required this.kv0,
    required this.kv1,
    required this.q,
    required this.k,
    required this.v,
    required this.gate,
    required this.o,
    required this.qNorm,
    required this.kNorm,
    required this.keys,
    required this.values,
    required this.sliding,
  });

  /// Query heads [h0, h1) and the KV heads [kv0, kv1) they read.
  final int h0, h1, kv0, kv1;
  final Linear q, k, v, gate, o;
  final Float32List qNorm, kNorm;

  /// [maxSeq, (kv1 - kv0) * headDim].
  final Float32List keys, values;
  final bool sliding;
}

final class _FfnShard {
  _FfnShard(this.w1, this.w3, this.w2);

  /// [width, H], [width, H] and [H, width].
  final Linear w1, w3, w2;
}

final class LayerShard {
  LayerShard._(
    this.config,
    this.maxSeq,
    this.maxChunk,
    this._conv,
    this._attention,
    this._ffn,
    this._lmHead,
    this.vocabStart,
  ) : _fullRope = Rope(config.fullRope, config.headDim),
      _slidingRope = Rope(config.slidingRope, config.headDim);

  /// Loads shard [index] of [shards] from `model.safetensors`, quantizing
  /// to [format] every matrix whose name contains no entry of [keepF32].
  /// With Q8, matrices found in [q8Cache] are mapped instead of read and
  /// quantized, and newly quantized ones are collected in [q8Built].
  factory LayerShard.load(
    String modelDir, {
    required int index,
    required int shards,
    required int maxSeq,
    required int maxChunk,
    WeightFormat format = WeightFormat.f32,
    List<String> keepF32 = const [],
    Q8Cache? q8Cache,
    Map<String, NativeMatrix>? q8Built,
  }) {
    if ((q8Cache != null || q8Built != null) && shards != 1) {
      throw ArgumentError('the Q8 cache holds whole matrices: one shard only');
    }
    final c = Lfm3Config.load(modelDir);
    final h = c.hiddenSize;
    final hd = c.headDim;
    final group = c.numHeads ~/ c.numKvHeads;
    final st = SafetensorsFile.open('$modelDir/model.safetensors');
    // Reads lazily so a cached Q8 matrix skips the safetensors read.
    Linear linear(
      String name,
      int rows,
      int cols,
      Float32List Function() read,
    ) {
      // The tied LM head stays Q8 under Q4 (see WeightFormat.q4).
      final f = keepF32.any(name.contains)
          ? WeightFormat.f32
          : format == WeightFormat.q4 && name == 'model.embed_tokens.weight'
          ? WeightFormat.q8
          : format;
      if (f.isNative) {
        final cached = q8Cache?.matrix(name, rows, cols);
        if (cached != null) return Linear.native(cached);
      }
      final made = Linear.from(read(), rows, cols, f);
      if (made.native != null) q8Built?[name] = made.native!;
      return made;
    }

    try {
      final convs = <_ConvShard?>[];
      final attentions = <_AttentionShard?>[];
      final ffns = <_FfnShard>[];
      final (c0, c1) = shareOf(h, index, shards, align: 4);
      final (h0, h1) = shareOf(c.numHeads, index, shards);
      final kv0 = h0 ~/ group;
      final kv1 = h1 > h0 ? (h1 - 1) ~/ group + 1 : kv0;
      final (f0, f1) = shareOf(c.intermediateSize, index, shards, align: 4);
      for (var i = 0; i < c.numLayers; i++) {
        final p = 'model.layers.$i.';
        final type = c.layerTypes[i];
        if (type == LayerType.conv) {
          final name = '${p}conv.in_proj.weight';
          final kernelAll = st.readF32('${p}conv.conv.weight', [
            h,
            1,
            c.convKernel,
          ]);
          convs.add(
            _ConvShard(
              c0,
              c1,
              linear(
                name,
                3 * (c1 - c0),
                h,
                () => readSlice(
                  st,
                  name,
                  3 * h,
                  h,
                  rowRanges: [
                    (c0, c1),
                    (h + c0, h + c1),
                    (2 * h + c0, 2 * h + c1),
                  ],
                ),
              ),
              Float32List.sublistView(
                kernelAll,
                c0 * c.convKernel,
                c1 * c.convKernel,
              ),
              linear(
                '${p}conv.out_proj.weight',
                h,
                c1 - c0,
                () => readSlice(
                  st,
                  '${p}conv.out_proj.weight',
                  h,
                  h,
                  colStart: c0,
                  colEnd: c1,
                ),
              ),
              Float32List((c.convKernel - 1) * (c1 - c0)),
            ),
          );
          attentions.add(null);
        } else {
          final a = '${p}self_attn.';
          final qDim = c.numHeads * hd;
          attentions.add(
            _AttentionShard(
              h0: h0,
              h1: h1,
              kv0: kv0,
              kv1: kv1,
              q: linear(
                '${a}q_proj',
                (h1 - h0) * hd,
                h,
                () => readSlice(
                  st,
                  '${a}q_proj.weight',
                  qDim,
                  h,
                  rowRanges: [(h0 * hd, h1 * hd)],
                ),
              ),
              k: linear(
                '${a}k_proj',
                (kv1 - kv0) * hd,
                h,
                () => readSlice(
                  st,
                  '${a}k_proj.weight',
                  c.kvDim,
                  h,
                  rowRanges: [(kv0 * hd, kv1 * hd)],
                ),
              ),
              v: linear(
                '${a}v_proj',
                (kv1 - kv0) * hd,
                h,
                () => readSlice(
                  st,
                  '${a}v_proj.weight',
                  c.kvDim,
                  h,
                  rowRanges: [(kv0 * hd, kv1 * hd)],
                ),
              ),
              gate: linear(
                '${a}gate_proj',
                h1 - h0,
                h,
                () => readSlice(
                  st,
                  '${a}gate_proj.weight',
                  c.numHeads,
                  h,
                  rowRanges: [(h0, h1)],
                ),
              ),
              o: linear(
                '${a}o_proj',
                h,
                (h1 - h0) * hd,
                () => readSlice(
                  st,
                  '${a}o_proj.weight',
                  h,
                  qDim,
                  colStart: h0 * hd,
                  colEnd: h1 * hd,
                ),
              ),
              qNorm: st.readF32('${a}q_norm.weight', [hd]),
              kNorm: st.readF32('${a}k_norm.weight', [hd]),
              keys: Float32List(maxSeq * (kv1 - kv0) * hd),
              values: Float32List(maxSeq * (kv1 - kv0) * hd),
              sliding: type == LayerType.slidingAttention,
            ),
          );
          convs.add(null);
        }
        final f = '${p}feed_forward.';
        final fi = c.intermediateSize;
        ffns.add(
          _FfnShard(
            linear(
              '${f}w1',
              f1 - f0,
              h,
              () =>
                  readSlice(st, '${f}w1.weight', fi, h, rowRanges: [(f0, f1)]),
            ),
            linear(
              '${f}w3',
              f1 - f0,
              h,
              () =>
                  readSlice(st, '${f}w3.weight', fi, h, rowRanges: [(f0, f1)]),
            ),
            linear(
              '${f}w2',
              h,
              f1 - f0,
              () => readSlice(
                st,
                '${f}w2.weight',
                h,
                fi,
                colStart: f0,
                colEnd: f1,
              ),
            ),
          ),
        );
      }
      const embed = 'model.embed_tokens.weight';
      final (v0, v1) = shareOf(c.vocabSize, index, shards, align: 2);
      final lmHead = linear(
        embed,
        v1 - v0,
        h,
        () => readSlice(st, embed, c.vocabSize, h, rowRanges: [(v0, v1)]),
      );
      return LayerShard._(
        c,
        maxSeq,
        maxChunk,
        convs,
        attentions,
        ffns,
        lmHead,
        v0,
      );
    } finally {
      st.close();
    }
  }

  final Lfm3Config config;
  final int maxSeq;
  final int maxChunk;
  final List<_ConvShard?> _conv;
  final List<_AttentionShard?> _attention;
  final List<_FfnShard> _ffn;
  final Linear _lmHead;
  final Rope _fullRope, _slidingRope;

  /// First vocabulary row of this shard's LM-head slice.
  final int vocabStart;
  int get vocabRows => _lmHead.rows;

  // Scratch, sized for the widest slice of a full chunk.
  late final Float32List _out = Float32List(maxChunk * config.hiddenSize);
  late final Float32List _bcx = Float32List(maxChunk * 3 * config.hiddenSize);
  late final Float32List _z = Float32List(maxChunk * config.hiddenSize);
  late final Float32List _y = Float32List(maxChunk * config.hiddenSize);
  late final Float32List _q = Float32List(
    maxChunk * config.numHeads * config.headDim,
  );
  late final Float32List _k = Float32List(maxChunk * config.kvDim);
  late final Float32List _v = Float32List(maxChunk * config.kvDim);
  late final Float32List _attnOut = Float32List(
    maxChunk * config.numHeads * config.headDim,
  );
  late final Float32List _gate = Float32List(maxChunk * config.numHeads);
  late final Float64List _scores = Float64List(maxSeq);
  late final Float32List _g = Float32List(maxChunk * config.intermediateSize);
  late final Float32List _u = Float32List(maxChunk * config.intermediateSize);
  late final Float32List _logits = Float32List(_lmHead.rows);

  /// Clears the conv histories so the next call starts a new sequence. KV
  /// cache rows are overwritten before they are read, so they need no reset.
  void reset() {
    for (final conv in _conv) {
      conv?.history.fillRange(0, conv.history.length, 0);
    }
  }

  /// A copy of the conv histories: with the KV cache rows before the
  /// current position, which later tokens never overwrite, they are all the
  /// state a position carries.
  List<Float32List> saveState() => [
    for (final conv in _conv)
      if (conv != null) Float32List.fromList(conv.history),
  ];

  /// Puts back conv histories from [saveState].
  void restoreState(List<Float32List> state) {
    var i = 0;
    for (final conv in _conv) {
      conv?.history.setAll(0, state[i++]);
    }
  }

  /// Views of everything the first [position] tokens left behind: the conv
  /// histories, then each attention layer's keys and values rows before
  /// [position]. Writing them out and reading them back into another model
  /// on the same backend restores that position (see Lfm3Model.saveState).
  List<Float32List> stateBlocks(int position) => [
    for (final conv in _conv)
      if (conv != null) conv.history,
    for (final att in _attention)
      if (att != null) ...[
        Float32List.sublistView(
          att.keys,
          0,
          position * (att.kv1 - att.kv0) * config.headDim,
        ),
        Float32List.sublistView(
          att.values,
          0,
          position * (att.kv1 - att.kv0) * config.headDim,
        ),
      ],
  ];

  /// This shard's partial of layer [layer]'s mixer for the [n] normalized
  /// inputs [a] ([n, H]) at positions [position], [position] + 1, ...
  /// Returns a view of [n, H] valid until the next call.
  Float32List mixer(int layer, Float32List a, int n, int position) {
    final conv = _conv[layer];
    if (conv != null) {
      _convMixer(conv, a, n);
    } else {
      _attentionMixer(_attention[layer]!, a, n, position);
    }
    return Float32List.sublistView(_out, 0, n * config.hiddenSize);
  }

  /// This shard's partial of layer [layer]'s SwiGLU FFN.
  Float32List ffn(int layer, Float32List a, int n) {
    final ffn = _ffn[layer];
    final width = ffn.w1.rows;
    final h = config.hiddenSize;
    if (width == 0) {
      _out.fillRange(0, n * h, 0);
    } else {
      ffn.w1.apply(a, _g, n);
      ffn.w3.apply(a, _u, n);
      for (var j = 0; j < n * width; j++) {
        final g = _g[j];
        _g[j] = g / (1 + math.exp(-g));
        _g[j] = _g[j] * _u[j];
      }
      ffn.w2.apply(_g, _out, n);
    }
    return Float32List.sublistView(_out, 0, n * h);
  }

  /// Logits of vocabulary rows [vocabStart, vocabStart + vocabRows).
  Float32List lmHead(Float32List normed) {
    _lmHead.apply(normed, _logits, 1);
    return _logits;
  }

  /// Gated short convolution of this shard's channels: z = B * X, then a
  /// causal depthwise conv over the last K values of z, then
  /// out_proj[:, channels](C * conv).
  void _convMixer(_ConvShard conv, Float32List a, int n) {
    final h = config.hiddenSize;
    final k = config.convKernel;
    final width = conv.width;
    if (width == 0) {
      _out.fillRange(0, n * h, 0);
      return;
    }
    final history = conv.history;
    conv.inProj.apply(a, _bcx, n);
    for (var t = 0; t < n; t++) {
      final row = t * 3 * width;
      for (var ch = 0; ch < width; ch++) {
        _z[t * width + ch] = _bcx[row + ch] * _bcx[row + 2 * width + ch];
      }
    }
    // Input s steps before token t is z[t - s] inside the chunk, or a
    // history row (oldest first, the newest at row k - 2) before it.
    for (var t = 0; t < n; t++) {
      final row = t * 3 * width;
      for (var ch = 0; ch < width; ch++) {
        final w = ch * k;
        var acc = 0.0;
        for (var s = 0; s < k; s++) {
          final src = t - s;
          final z = src >= 0
              ? _z[src * width + ch]
              : history[(k - 1 + src) * width + ch];
          acc += conv.kernel[w + k - 1 - s] * z;
        }
        _y[t * width + ch] = _bcx[row + width + ch] * acc;
      }
    }
    // Keep the last k - 1 gated inputs of history followed by this chunk.
    final keep = k - 1;
    if (n >= keep) {
      history.setRange(0, keep * width, _z, (n - keep) * width);
    } else {
      history.setRange(0, (keep - n) * width, history, n * width);
      history.setRange((keep - n) * width, keep * width, _z);
    }
    conv.outProj.apply(_y, _out, n);
  }

  /// Grouped-query attention for this shard's query heads, with QK-norm,
  /// RoPE and the per-head output gate, then o_proj[:, heads].
  void _attentionMixer(
    _AttentionShard att,
    Float32List a,
    int n,
    int position,
  ) {
    final c = config;
    final h = c.hiddenSize;
    final hd = c.headDim;
    final heads = att.h1 - att.h0;
    if (heads == 0) {
      _out.fillRange(0, n * h, 0);
      return;
    }
    final qDim = heads * hd;
    final kvDim = (att.kv1 - att.kv0) * hd;
    final rope = att.sliding ? _slidingRope : _fullRope;
    final keys = att.keys;
    final values = att.values;

    att.q.apply(a, _q, n);
    att.k.apply(a, _k, n);
    att.v.apply(a, _v, n);
    att.gate.apply(a, _gate, n);
    for (var t = 0; t < n; t++) {
      final pos = position + t;
      rope.setPosition(pos);
      for (var head = 0; head < heads; head++) {
        final o = t * qDim + head * hd;
        rmsNorm(_q, att.qNorm, _q, o, o, hd, c.normEps);
        rope.apply(_q, o);
      }
      for (var head = 0; head < att.kv1 - att.kv0; head++) {
        final o = t * kvDim + head * hd;
        rmsNorm(_k, att.kNorm, _k, o, o, hd, c.normEps);
        rope.apply(_k, o);
      }
      keys.setRange(pos * kvDim, (pos + 1) * kvDim, _k, t * kvDim);
      values.setRange(pos * kvDim, (pos + 1) * kvDim, _v, t * kvDim);
    }

    // With the native kernels loaded, a shard holding every head hands the
    // attention itself to C, which splits (token, head) pairs over threads.
    if (NativeKernels.isLoaded &&
        NativeKernels.instance.attention &&
        att.h0 == 0 &&
        heads == c.numHeads) {
      nativeAttention(
        _q,
        keys,
        values,
        _gate,
        _attnOut,
        n: n,
        position: position,
        heads: heads,
        kvHeads: att.kv1 - att.kv0,
        headDim: hd,
        window: att.sliding ? c.slidingWindow : 0,
      );
      att.o.apply(_attnOut, _out, n);
      return;
    }

    final group = c.numHeads ~/ c.numKvHeads;
    final scale = 1 / math.sqrt(hd);
    for (var t = 0; t < n; t++) {
      final pos = position + t;
      // The reference mask keeps kv positions with kv > q - window, i.e.
      // the current token and the window - 1 before it.
      final start = att.sliding ? math.max(0, pos - c.slidingWindow + 1) : 0;
      for (var head = 0; head < heads; head++) {
        final qo = t * qDim + head * hd;
        final kvo = ((att.h0 + head) ~/ group - att.kv0) * hd;
        var maxScore = double.negativeInfinity;
        for (var s = start; s <= pos; s++) {
          final ko = s * kvDim + kvo;
          var dot = 0.0;
          for (var d = 0; d < hd; d++) {
            dot += _q[qo + d] * keys[ko + d];
          }
          final score = dot * scale;
          _scores[s] = score;
          if (score > maxScore) maxScore = score;
        }
        var sum = 0.0;
        for (var s = start; s <= pos; s++) {
          final e = math.exp(_scores[s] - maxScore);
          _scores[s] = e;
          sum += e;
        }
        final g = 1 / (1 + math.exp(-_gate[t * heads + head]));
        for (var d = 0; d < hd; d++) {
          var acc = 0.0;
          for (var s = start; s <= pos; s++) {
            acc += _scores[s] * values[s * kvDim + kvo + d];
          }
          _attnOut[qo + d] = acc / sum;
          _attnOut[qo + d] = _attnOut[qo + d] * g;
        }
      }
    }
    att.o.apply(_attnOut, _out, n);
  }
}
