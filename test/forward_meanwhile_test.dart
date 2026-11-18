import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/gpu_host.dart';
import 'support/vision_model_fixture.dart';

void main() {
  if (!NativeKernels.isAvailable) {
    test('forward runs meanwhile', () {}, skip: 'requires native kernels');
    return;
  }

  late Directory root;
  setUpAll(() async {
    root = Directory.systemTemp.createTempSync('dianoia-meanwhile-');
    await prepareVisionFixture(root);
  });
  tearDownAll(() => root.deleteSync(recursive: true));

  for (final gpu in [false, true]) {
    test(
      'forward runs meanwhile once and keeps its results '
      '(${gpu ? 'GPU' : 'CPU'})',
      () async {
        NativeKernels.load(null, threads: 2);
        final backbone = '${root.path}/backbone';
        final model = await Lfm3Model.load(
          backbone,
          maxSeq: 64,
          maxChunk: 4,
          weights: WeightFormat.q8,
          q8CachePath: '$backbone/lfm3-q8.cache',
          gpu: gpu,
        );
        addTearDown(model.close);
        const ids = [40, 50, 60, 70, 80, 90];
        await model.forward(ids);
        final plain = Float32List.fromList(model.logits);

        await model.reset();
        var runs = 0;
        await model.forward(ids, meanwhile: () => runs++);
        expect(runs, 1);
        expect(model.logits, orderedEquals(plain));

        // No tokens: it still runs once.
        await model.forward(
          const [],
          computeLogits: false,
          meanwhile: () => runs++,
        );
        expect(runs, 2);

        // Its error comes after the pass, which keeps its state.
        await model.reset();
        await expectLater(
          model.forward(ids, meanwhile: () => throw const FormatException('x')),
          throwsFormatException,
        );
        expect(model.length, ids.length);
        expect(model.logits, orderedEquals(plain));
        await model.step(7);
        expect(model.length, ids.length + 1);
      },
      skip: gpu && !hostHasGpu
          ? noGpuSkip
          : gpu && !gpuStarts
          ? gpuDoesNotStartSkip
          : false,
    );
  }
}
