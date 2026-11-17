import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/gpu_host.dart';

void main() {
  group(
    'GpuLinear',
    () {
      late GpuLinear gpu;
      final random = math.Random(5);
      Float32List values(int count) => Float32List.fromList([
        for (var i = 0; i < count; i++) random.nextDouble() * 2 - 1,
      ]);

      setUpAll(() {
        NativeKernels.load(null, threads: 4);
        gpu = GpuLinear.start()!;
      });
      tearDownAll(() => gpu.dispose());

      // Row and token counts that do not divide into the GEMM's tiles.
      for (final (format, rows, cols, n) in [
        ('q4', 64, 64, 1),
        ('q4', 70, 96, 13),
        ('q8', 36, 64, 9),
        ('q8', 128, 128, 40),
      ]) {
        test('matches the CPU kernels: $format $rows x $cols, n=$n', () {
          final w = values(rows * cols);
          final matrix = format == 'q4'
              ? NativeQ4Matrix.quantize(w, rows, cols)
              : NativeQ8Matrix.quantize(w, rows, cols);
          addTearDown(matrix.free);
          gpu.upload('m', matrix);
          final xs = values(n * cols);
          final cpu = Float32List(n * rows), onGpu = Float32List(n * rows);
          matrix.apply(xs, cpu, n, int8Inputs: false);
          gpu.apply('m', xs, onGpu, n);
          for (var i = 0; i < cpu.length; i++) {
            expect(onGpu[i], closeTo(cpu[i], 1e-4 + 1e-5 * cpu[i].abs()));
          }
        });
      }

      test('grows its buffers and keeps earlier matrices', () {
        final small = NativeQ4Matrix.quantize(values(32 * 32), 32, 32);
        final large = NativeQ4Matrix.quantize(values(64 * 256), 64, 256);
        addTearDown(small.free);
        addTearDown(large.free);
        gpu
          ..upload('small', small)
          ..upload('large', large);
        for (final (name, m, n) in [
          ('small', small, 3),
          ('large', large, 50),
          ('small', small, 7),
        ]) {
          final xs = values(n * m.cols);
          final cpu = Float32List(n * m.rows), onGpu = Float32List(n * m.rows);
          m.apply(xs, cpu, n, int8Inputs: false);
          gpu.apply(name, xs, onGpu, n);
          for (var i = 0; i < cpu.length; i++) {
            expect(onGpu[i], closeTo(cpu[i], 1e-4 + 1e-5 * cpu[i].abs()));
          }
        }
      });

      test('rejects unknown matrices and short lists', () {
        final m = NativeQ4Matrix.quantize(values(32 * 32), 32, 32);
        addTearDown(m.free);
        gpu.upload('m', m);
        expect(
          () => gpu.apply('missing', Float32List(32), Float32List(32), 1),
          throwsArgumentError,
        );
        expect(
          () => gpu.apply('m', Float32List(32), Float32List(32), 2),
          throwsArgumentError,
        );
      });

      test('frees its buffers once and then refuses work', () {
        final live = GpuForward.liveBuffers;
        final other = GpuLinear.start()!;
        final m = NativeQ8Matrix.quantize(values(32 * 32), 32, 32);
        addTearDown(m.free);
        other.upload('m', m);
        other.apply('m', values(32), Float32List(32), 1);
        expect(GpuForward.liveBuffers, live + 3);
        other
          ..dispose()
          ..dispose();
        expect(GpuForward.liveBuffers, live);
        expect(
          () => other.apply('m', values(32), Float32List(32), 1),
          throwsStateError,
        );
      });
    },
    skip: !NativeKernels.isAvailable
        ? 'requires native kernels'
        : !hostHasGpu
        ? noGpuSkip
        : gpuStarts
        ? false
        : gpuDoesNotStartSkip,
  );
}
