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
    Abi.current() == Abi.androidArm64;

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
    }
  }
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
