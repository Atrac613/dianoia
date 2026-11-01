// Regression for Q8 row lookup after CPU warmup and cache reload on the GPU.
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/gpu_host.dart';

void main() {
  final directory = Platform.environment['DIANOIA_LFM3_DIR'];
  final skip = directory == null
      ? 'set DIANOIA_LFM3_DIR to a Q4 bundle'
      : !hostHasGpu
      ? noGpuSkip
      : false;
  test('Q8 embeddings survive hot CPU lookup then GPU cache reload', () async {
    NativeKernels.load(null, threads: 2);
    final ids = List.generate(64, (i) => 10 + i);
    Float32List? expected;
    for (final gpu in [false, true]) {
      final model = await Lfm3Model.load(
        directory!,
        weights: WeightFormat.q4,
        q8CachePath: '$directory/lfm3-q4.cache',
        maxSeq: 128,
        maxChunk: 2,
        gpu: gpu,
      );
      try {
        for (var pass = 0; pass < (gpu ? 1 : 2); pass++) {
          await model.reset();
          final embeddings = Float32List(ids.length * model.config.hiddenSize);
          await model.forward(
            ids,
            computeLogits: false,
            capture: (slot, position, rows, count) {
              if (slot == 0) {
                embeddings.setRange(
                  position * model.config.hiddenSize,
                  (position + count) * model.config.hiddenSize,
                  rows,
                );
              }
            },
          );
          if (expected == null) {
            expected = embeddings;
          } else {
            expect(embeddings, orderedEquals(expected));
          }
        }
      } finally {
        await model.close();
      }
    }
  }, skip: skip);
}
