import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

final _modelDir = Platform.environment['DIANOIA_LFM3_DIR'];

void main() {
  for (final gpu in [false, true]) {
    group(gpu ? 'GPU embedding overrides' : 'CPU embedding overrides', () {
      late Lfm3Model model;
      setUpAll(() async {
        NativeKernels.load(null, threads: 2);
        model = await Lfm3Model.load(
          _modelDir!,
          maxSeq: 32,
          maxChunk: 2,
          weights: WeightFormat.q4,
          q8CachePath: '$_modelDir/lfm3-q4.cache',
          gpu: gpu,
        );
        addTearDown(model.close);
      });
      setUp(() => model.reset());

      test(
        'projected rows cross chunk boundaries and preserve arithmetic',
        () async {
          final rows = <int, Float32List>{};
          const ids = [10, 20, 30, 40, 50];
          await model.forward(
            ids,
            capture: (slot, first, data, count) {
              if (slot != 0) return;
              final h = model.config.hiddenSize;
              for (var t = 0; t < count; t++) {
                rows[first + t] = Float32List.fromList(
                  data.sublist(t * h, (t + 1) * h),
                );
              }
            },
          );
          final expected = Float32List.fromList(model.logits);
          await model.reset();
          // The placeholder IDs differ, but the supplied vectors are identical.
          await model.forward(
            [10, 99, 99, 99, 50],
            embeddingOverrides: {
              for (final i in [1, 2, 3]) i: rows[i]!,
            },
          );
          expect(model.logits, orderedEquals(expected));
          expect(model.history, [10, -1, -1, -1, 50]);
          await model.checkpoint();
          expect(await model.prefill(ids), 0);
          expect(model.logits, orderedEquals(expected));
        },
      );

      test('invalid overrides fail before sequence state changes', () async {
        for (final overrides in <Map<int, Float32List>>[
          {-1: Float32List(model.config.hiddenSize)},
          {1: Float32List(model.config.hiddenSize)},
          {0: Float32List(1)},
          {0: Float32List(model.config.hiddenSize)..[0] = double.nan},
        ]) {
          await expectLater(
            model.forward([10], embeddingOverrides: overrides),
            throwsArgumentError,
          );
          expect(model.length, 0);
          expect(model.history, isEmpty);
        }
      });
    }, skip: _modelDir == null ? 'set DIANOIA_LFM3_DIR to a Q4 bundle' : false);
  }
}
