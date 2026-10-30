// The build hook's kernels asset against double-precision references: this
// is the test that the hook compiled and bundled the kernels at all.
import 'dart:ffi';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

/// Where hook/build.dart compiles the kernels.
final _supported =
    Abi.current() == Abi.macosArm64 ||
    Abi.current() == Abi.iosArm64 ||
    Abi.current() == Abi.androidArm64 ||
    Abi.current() == Abi.windowsX64;

void main() {
  setUpAll(() {
    if (_supported) NativeKernels.load(null, threads: 2);
  });

  test('the kernels asset exists exactly where the hook builds it', () {
    expect(NativeKernels.isAvailable, _supported);
  });

  for (final (format, quantize) in [
    ('Q8', NativeQ8Matrix.quantize),
    ('Q4', NativeQ4Matrix.quantize),
  ]) {
    for (final int8 in [false, true]) {
      test('$format products with ${int8 ? 'int8' : 'float'} inputs', () {
        const rows = 37, cols = 2048, n = 6;
        final rng = math.Random(11);
        final w = Float32List(rows * cols);
        for (var i = 0; i < w.length; i++) {
          w[i] = rng.nextDouble() - 0.5;
        }
        final xs = Float32List(n * cols);
        for (var i = 0; i < xs.length; i++) {
          xs[i] = rng.nextDouble() * 2 - 1;
        }
        final NativeMatrix q = quantize(w, rows, cols);
        final ys = Float32List(n * rows);
        q.apply(xs, ys, n, int8Inputs: int8);

        final x = int8 ? _quantizedInputs(xs) : xs;
        var worst = 0.0;
        for (var t = 0; t < n; t++) {
          for (var r = 0; r < rows; r++) {
            var ref = 0.0, mag = 0.0;
            for (var c = 0; c < cols; c++) {
              final d = q.weightAt(r, c) * x[t * cols + c];
              ref += d;
              mag += d.abs();
            }
            worst = math.max(worst, (ys[t * rows + r] - ref).abs() / mag);
          }
        }
        expect(worst, lessThan(1e-5));
      }, skip: _supported ? false : 'no native kernels for ${Abi.current()}');

      // The x86-64 kernels promise NEON's float operations in NEON's order,
      // so both must equal this replay of it bit for bit.
      test('$format products with ${int8 ? 'int8' : 'float'} inputs follow '
          "NEON's order bit for bit", () {
        const rows = 37, cols = 2048, n = 3;
        final rng = math.Random(23);
        final w = Float32List(rows * cols);
        for (var i = 0; i < w.length; i++) {
          w[i] = rng.nextDouble() - 0.5;
        }
        final xs = Float32List(n * cols);
        for (var i = 0; i < xs.length; i++) {
          xs[i] = rng.nextDouble() * 2 - 1;
        }
        final NativeMatrix q = quantize(w, rows, cols);
        final ys = Float32List(n * rows);
        q.apply(xs, ys, n, int8Inputs: int8);
        expect(ys, _neonProducts(q, xs, n, int8: int8));
      }, skip: _supported ? false : 'no native kernels for ${Abi.current()}');
    }
  }
}

double _f32(double x) => (Float32List(1)..[0] = x)[0];

/// A fused multiply-add rounded once to float32. a * b is exact in double
/// for the operands here (at most 20 and 24 bits), so only the add rounds
/// before the float32 rounding, which can differ from one rounding only
/// on a float32 tie.
double _fma(double a, double b, double c) => _f32(a * b + c);

/// lab_native.c's products replayed in Dart: four lanes, one block at a
/// time, lanes summed as (l0 + l1) + (l2 + l3).
Float32List _neonProducts(
  NativeMatrix q,
  Float32List xs,
  int n, {
  required bool int8,
}) {
  final rows = q.rows, cols = q.cols, blocks = cols ~/ 32;
  final q4 = q.format == 'q4';
  final qs = q.qsBytes;
  final scales = q4
      ? [
          for (final h in q.scaleBytes.buffer.asUint16List(
            q.scaleBytes.offsetInBytes,
            rows * blocks,
          ))
            halfToDouble(h),
        ]
      : q.scaleBytes.buffer.asFloat32List(
          q.scaleBytes.offsetInBytes,
          rows * blocks,
        );
  int weight(int r, int j) {
    if (!q4) return qs.buffer.asInt8List(qs.offsetInBytes)[r * cols + j];
    final b = j ~/ 32, k = j % 32;
    final byte = qs[(r * cols ~/ 2) + b * 16 + k % 16];
    return (k < 16 ? byte & 15 : byte >> 4) - 8;
  }

  final (xq, xsc) = _quantizeExact(xs);
  final ys = Float32List(n * rows);
  for (var t = 0; t < n; t++) {
    for (var r = 0; r < rows; r++) {
      final acc = List.filled(4, 0.0);
      for (var b = 0; b < blocks; b++) {
        final base = b * 32;
        final scale = scales[r * blocks + b];
        for (var l = 0; l < 4; l++) {
          if (int8) {
            // Two sdots: lane l takes bytes 4l..4l+3 of each half.
            var d = 0;
            for (final j in [4 * l, 4 * l + 16]) {
              for (var i = 0; i < 4; i++) {
                d += weight(r, base + j + i) * xq[t * cols + base + j + i];
              }
            }
            final s = _f32(scale * xsc[t * blocks + b]);
            acc[l] = _fma(d.toDouble(), s, acc[l]);
          } else {
            var s = 0.0;
            for (var k = 0; k < 8; k++) {
              final j = base + 4 * k + l;
              s = _fma(weight(r, j).toDouble(), xs[t * cols + j], s);
            }
            acc[l] = _fma(s, scale, acc[l]);
          }
        }
      }
      ys[t * rows + r] = _f32(_f32(acc[0] + acc[1]) + _f32(acc[2] + acc[3]));
    }
  }
  return ys;
}

/// lab_quantize_q8 on the inputs, in float32 as C does it: the int8 values
/// and the per-block scales.
(Int8List, Float32List) _quantizeExact(Float32List xs) {
  final q = Int8List(xs.length);
  final scales = Float32List(xs.length ~/ 32);
  for (var b = 0; b < scales.length; b++) {
    var amax = 0.0;
    for (var k = 0; k < 32; k++) {
      amax = math.max(amax, xs[b * 32 + k].abs());
    }
    final scale = scales[b] = _f32(amax / 127);
    final inv = scale > 0 ? _f32(1 / scale) : 0.0;
    for (var k = 0; k < 32; k++) {
      final v = _f32(xs[b * 32 + k] * inv);
      final r = _f32(v < 0 ? v - 0.5 : v + 0.5).truncate();
      q[b * 32 + k] = r.clamp(-127, 127);
    }
  }
  return (q, scales);
}

/// The inputs as the int8 path sees them: per block of 32, scaled by
/// amax / 127 and rounded half away from zero, as lab_native.c does.
List<double> _quantizedInputs(Float32List xs) {
  final x = Float64List(xs.length);
  for (var base = 0; base < xs.length; base += 32) {
    var amax = 0.0;
    for (var k = 0; k < 32; k++) {
      amax = math.max(amax, xs[base + k].abs());
    }
    final scale = Float32List.fromList([amax / 127])[0];
    final inv = scale > 0 ? Float32List.fromList([1 / scale])[0] : 0.0;
    for (var k = 0; k < 32; k++) {
      final v = Float32List.fromList([xs[base + k] * inv])[0];
      final r = (v < 0 ? v - 0.5 : v + 0.5).truncate().clamp(-127, 127);
      x[base + k] = r * scale;
    }
  }
  return x;
}
