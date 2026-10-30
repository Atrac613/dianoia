/// Matrix products for row-major float32 weights.
///
/// [gemv] multiplies one vector. Its kernel is `gemvV4x8NochkF32` from
/// stage1_gemv: eight Float32x4 accumulators, compiled without bounds checks,
/// the fastest pure-Dart variant measured in stage 2 (about 30 GB/s on one
/// M2 Max core).
///
/// [gemm] multiplies a batch of vectors, such as the tokens of a prompt. It
/// reuses each loaded weight vector for four tokens, so Dart's per-load
/// overhead (index arithmetic and a data-pointer reload) is paid once per
/// four multiply-adds. On 7680 x 2048 weights with 60 tokens this measured
/// 13.7 GMAC/s against 7.2 for repeated gemv. Blocks of 4 x 4 were slower,
/// probably from register spills; 2 x 6 gained under 10 % more.
library;

import 'dart:typed_data';

/// y[0:rows] = w[rows, cols] * x[0:cols].
///
/// Shapes are validated here because the inner loop runs without bounds
/// checks. All lists must start on a 16-byte boundary of their buffer.
void gemv(Float32List w, Float32List x, Float32List y, int rows, int cols) {
  if (cols % 4 != 0 ||
      w.length < rows * cols ||
      x.length < cols ||
      y.length < rows ||
      w.offsetInBytes % 16 != 0 ||
      x.offsetInBytes % 16 != 0) {
    throw ArgumentError(
      'gemv: bad shapes rows=$rows cols=$cols w=${w.length} '
      'x=${x.length} y=${y.length}',
    );
  }
  _gemvUnchecked(
    Float32x4List.sublistView(w, 0, rows * cols),
    Float32x4List.sublistView(x, 0, cols),
    y,
    rows,
    cols ~/ 4,
  );
}

@pragma('vm:unsafe:no-bounds-checks')
void _gemvUnchecked(
  Float32x4List wv,
  Float32x4List xv,
  Float32List y,
  int rows,
  int kv,
) {
  final mainV = kv - kv % 8;
  for (var i = 0; i < rows; i++) {
    final base = i * kv;
    var v0 = Float32x4.zero();
    var v1 = Float32x4.zero();
    var v2 = Float32x4.zero();
    var v3 = Float32x4.zero();
    var v4 = Float32x4.zero();
    var v5 = Float32x4.zero();
    var v6 = Float32x4.zero();
    var v7 = Float32x4.zero();
    var j = 0;
    for (; j < mainV; j += 8) {
      v0 += wv[base + j] * xv[j];
      v1 += wv[base + j + 1] * xv[j + 1];
      v2 += wv[base + j + 2] * xv[j + 2];
      v3 += wv[base + j + 3] * xv[j + 3];
      v4 += wv[base + j + 4] * xv[j + 4];
      v5 += wv[base + j + 5] * xv[j + 5];
      v6 += wv[base + j + 6] * xv[j + 6];
      v7 += wv[base + j + 7] * xv[j + 7];
    }
    for (; j < kv; j++) {
      v0 += wv[base + j] * xv[j];
    }
    final v = ((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7));
    final t = v + v.shuffle(Float32x4.yxwz);
    y[i] = t.x + t.z;
  }
}

/// ys[t] = w[rows, cols] * xs[t] for t < n, with xs packed as [n, cols] and
/// ys as [n, rows]. Falls back to [gemv] per vector for the last n % 4
/// vectors and when rows is odd.
void gemm(
  Float32List w,
  Float32List xs,
  Float32List ys,
  int rows,
  int cols,
  int n,
) {
  if (cols % 4 != 0 ||
      w.length < rows * cols ||
      xs.length < n * cols ||
      ys.length < n * rows ||
      w.offsetInBytes % 16 != 0 ||
      xs.offsetInBytes % 16 != 0) {
    throw ArgumentError(
      'gemm: bad shapes rows=$rows cols=$cols n=$n w=${w.length} '
      'xs=${xs.length} ys=${ys.length}',
    );
  }
  final blocked = rows.isEven ? n - n % 4 : 0;
  if (blocked > 0) {
    _gemmUnchecked(
      Float32x4List.sublistView(w, 0, rows * cols),
      Float32x4List.sublistView(xs, 0, blocked * cols),
      ys,
      rows,
      cols ~/ 4,
      blocked,
    );
  }
  for (var t = blocked; t < n; t++) {
    gemv(
      w,
      Float32List.sublistView(xs, t * cols, (t + 1) * cols),
      Float32List.sublistView(ys, t * rows, (t + 1) * rows),
      rows,
      cols,
    );
  }
}

/// Two weight rows times four vectors per block: eight accumulators.
@pragma('vm:unsafe:no-bounds-checks')
void _gemmUnchecked(
  Float32x4List wv,
  Float32x4List xv,
  Float32List ys,
  int rows,
  int kv,
  int n,
) {
  for (var t0 = 0; t0 < n; t0 += 4) {
    final x0 = t0 * kv;
    final x1 = x0 + kv;
    final x2 = x1 + kv;
    final x3 = x2 + kv;
    for (var r = 0; r < rows; r += 2) {
      final w0 = r * kv;
      final w1 = w0 + kv;
      var a0 = Float32x4.zero();
      var a1 = Float32x4.zero();
      var a2 = Float32x4.zero();
      var a3 = Float32x4.zero();
      var b0 = Float32x4.zero();
      var b1 = Float32x4.zero();
      var b2 = Float32x4.zero();
      var b3 = Float32x4.zero();
      for (var j = 0; j < kv; j++) {
        final wa = wv[w0 + j];
        final wb = wv[w1 + j];
        final xa = xv[x0 + j];
        final xb = xv[x1 + j];
        final xc = xv[x2 + j];
        final xd = xv[x3 + j];
        a0 += wa * xa;
        a1 += wa * xb;
        a2 += wa * xc;
        a3 += wa * xd;
        b0 += wb * xa;
        b1 += wb * xb;
        b2 += wb * xc;
        b3 += wb * xd;
      }
      ys[t0 * rows + r] = _laneSum(a0);
      ys[(t0 + 1) * rows + r] = _laneSum(a1);
      ys[(t0 + 2) * rows + r] = _laneSum(a2);
      ys[(t0 + 3) * rows + r] = _laneSum(a3);
      ys[t0 * rows + r + 1] = _laneSum(b0);
      ys[(t0 + 1) * rows + r + 1] = _laneSum(b1);
      ys[(t0 + 2) * rows + r + 1] = _laneSum(b2);
      ys[(t0 + 3) * rows + r + 1] = _laneSum(b3);
    }
  }
}

@pragma('vm:prefer-inline')
double _laneSum(Float32x4 v) {
  final t = v + v.shuffle(Float32x4.yxwz);
  return t.x + t.z;
}
