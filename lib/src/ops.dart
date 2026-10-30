/// Small numeric pieces shared by the main isolate and the layer shards.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'config.dart';

/// RMSNorm of x[xOffset, xOffset + n) into out[outOffset, ...). The
/// reference rounds x * rsqrt(...) to float32 before multiplying by the
/// weight, so this does too.
void rmsNorm(
  Float32List x,
  Float32List weight,
  Float32List out,
  int xOffset,
  int outOffset,
  int n,
  double eps,
) {
  var sumSquares = 0.0;
  for (var j = 0; j < n; j++) {
    final v = x[xOffset + j];
    sumSquares += v * v;
  }
  final r = 1 / math.sqrt(sumSquares / n + eps);
  for (var j = 0; j < n; j++) {
    out[outOffset + j] = x[xOffset + j] * r;
    out[outOffset + j] = weight[j] * out[outOffset + j];
  }
}

/// Rotation table for one RoPE flavour at one position. Values are rounded
/// to float32 at the same points as the reference, which computes
/// `inv_freq * position` and its cosine in float32.
final class Rope {
  Rope(RopeSpec spec, int headDim)
    : half = headDim ~/ 2,
      invFreq = Float32List(headDim ~/ 2),
      cos = Float32List(headDim ~/ 2),
      sin = Float32List(headDim ~/ 2) {
    final pairs = (spec.partialRotaryFactor * headDim).toInt() ~/ 2;
    final f = Float32List(1);
    for (var j = 0; j < pairs; j++) {
      f[0] = 2 * j / headDim;
      f[0] = math.pow(spec.theta, f[0]).toDouble();
      invFreq[j] = 1.0 / f[0];
    }
    // Pairs past `pairs` keep frequency zero: they pass through unrotated.
  }

  final int half;
  final Float32List invFreq, cos, sin;
  final Float32List _angle = Float32List(1);

  void setPosition(int position) {
    for (var j = 0; j < half; j++) {
      _angle[0] = position * invFreq[j];
      cos[j] = math.cos(_angle[0]);
      sin[j] = math.sin(_angle[0]);
    }
  }

  /// Half-split rotation: dimension j pairs with j + half.
  void apply(Float32List x, int offset) {
    for (var j = 0; j < half; j++) {
      final x1 = x[offset + j];
      final x2 = x[offset + j + half];
      x[offset + j] = x1 * cos[j] - x2 * sin[j];
      x[offset + j + half] = x2 * cos[j] + x1 * sin[j];
    }
  }
}

/// The contiguous share [start, end) of [count] items that shard [index] of
/// [shards] owns, with inner boundaries rounded down to a multiple of
/// [align].
(int, int) shareOf(int count, int index, int shards, {int align = 1}) {
  int boundary(int w) =>
      w == shards ? count : (count * w ~/ shards) ~/ align * align;
  return (boundary(index), boundary(index + 1));
}
