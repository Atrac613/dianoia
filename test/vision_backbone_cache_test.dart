import 'dart:io';
import 'dart:typed_data';
import 'package:dianoia/lab.dart';
import 'package:test/test.dart';
import 'support/vision_model_fixture.dart';

void main() {
  for (final weights in [WeightFormat.q8, WeightFormat.q4]) {
    test(
      'streamed ${weights.name} cache preserves every matrix payload',
      () async {
        final root = Directory.systemTemp.createTempSync(
          'dianoia-stream-cache-',
        );
        addTearDown(() => root.deleteSync(recursive: true));
        await prepareVisionFixture(root);
        final backbone = '${root.path}/backbone';
        final referencePath = '$backbone/reference.cache';
        final model = await Lfm3Model.load(
          backbone,
          weights: weights,
          maxSeq: 1,
          maxChunk: 1,
          q8CachePath: referencePath,
        );
        await model.close();
        buildVlBackboneCache(backbone, weights: weights);
        final key = Q8Cache.keyFor(backbone, weights: weights.name);
        final reference = Q8Cache.open(referencePath, key)!;
        final streamed = Q8Cache.open(
          '$backbone/lfm3-${weights.name}.cache',
          key,
        )!;
        try {
          expect(streamed.shapes.toSet(), reference.shapes.toSet());
          for (final (name, rows, cols) in reference.shapes) {
            final a = reference.matrix(name, rows, cols)!;
            final b = streamed.matrix(name, rows, cols)!;
            expect(b.format, a.format, reason: name);
            expect(b.qsBytes, a.qsBytes, reason: name);
            expect(b.scaleBytes, a.scaleBytes, reason: name);
            // Mapping ownership remains with the cache, not its matrices.
            expect(b.free, throwsStateError);
          }
        } finally {
          reference.close();
          streamed.close();
        }
      },
      skip: NativeKernels.isAvailable ? false : 'requires native kernels',
    );
  }

  test(
    'row-chunk quantization preserves Q8 payloads across chunk boundaries',
    () {
      NativeKernels.load(null);
      const rows = 37, cols = 32;
      final values = Float32List.fromList(
        List.generate(rows * cols, (i) => (i % 31 - 15) / 7),
      );
      final whole = NativeQ8Matrix.quantize(values, rows, cols);
      final chunks = NativeQ8Matrix.quantizeRows(rows, cols, (
        first,
        count,
        out,
      ) {
        out.setRange(0, count * cols, values, first * cols);
      }, chunk: 5);
      try {
        expect(chunks.qsBytes, whole.qsBytes);
        expect(chunks.scaleBytes, whole.scaleBytes);
      } finally {
        whole.free();
        chunks.free();
      }
      expect(
        () => NativeQ8Matrix.quantizeRows(rows, cols, (_, _, _) {
          throw const FormatException('row read failed');
        }),
        throwsFormatException,
      );
    },
    skip: NativeKernels.isAvailable ? false : 'requires native kernels',
  );

  test('a failed cache write still releases its produced matrix', () {
    NativeKernels.load(null);
    final root = Directory.systemTemp.createTempSync('dianoia-cache-write-');
    addTearDown(() => root.deleteSync(recursive: true));
    var released = 0;
    expect(
      () => Q8Cache.writeEach(
        '${root.path}/cache',
        'key',
        [('matrix', 'q4', 2, 32)],
        (_) => NativeQ4Matrix.quantize(Float32List(32), 1, 32),
        done: (matrix) {
          (matrix as NativeMatrix).free();
          matrix.free(); // Owned allocations can be released idempotently.
          released++;
        },
      ),
      throwsStateError,
    );
    expect(released, 1);
    expect(File('${root.path}/cache').existsSync(), isFalse);
  }, skip: NativeKernels.isAvailable ? false : 'requires native kernels');
}
