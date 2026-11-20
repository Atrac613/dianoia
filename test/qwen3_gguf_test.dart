// Importing a Qwen3 Q8_0 GGUF: a tiny synthetic checkpoint written as
// llama.cpp lays it out, imported without requantizing and run.
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/synthetic_qwen3.dart';

void main() {
  test('Q8_0 blocks split into values and widened scales', () {
    final values = Float32List.fromList([
      for (var i = 0; i < 64; i++) (i - 20) / 16,
    ]);
    final blocks = quantizeQ8_0(values);
    final m = splitQ8_0(blocks, 2, 32);
    expect(m.format, 'q8');
    final qs = Int8List.sublistView(m.qsBytes);
    final scales = Float32List.sublistView(m.scaleBytes);
    for (var b = 0; b < 2; b++) {
      expect(scales[b], halfToDouble(blocks[34 * b] | blocks[34 * b + 1] << 8));
      for (var k = 0; k < 32; k++) {
        expect(qs[32 * b + k], Int8List.sublistView(blocks)[34 * b + 2 + k]);
        expect(qs[32 * b + k] * scales[b], closeTo(values[32 * b + k], 0.02));
      }
    }
  });

  group('import', () {
    late Directory root;
    setUpAll(() {
      root = Directory.systemTemp.createTempSync('dianoia-qwen3-gguf-');
      writeSyntheticQwen3('${root.path}/hf');
      writeSyntheticQwen3Gguf('${root.path}/hf', '${root.path}/model.gguf');
      NativeKernels.load(null, threads: 2);
    });
    tearDownAll(() => root.deleteSync(recursive: true));

    test('keeps the GGUF values and opens with LocalModel', () async {
      final bundle = '${root.path}/bundle';
      final lines = importGguf(
        '${root.path}/model.gguf',
        '${root.path}/hf',
        bundle,
      );
      // Vectors are exact; matrices differ from bf16 by Q8 rounding only.
      expect(lines, hasLength(13));
      for (final line in lines) {
        final [diff, family] = line.trim().split(RegExp(r'\s+'));
        expect(
          double.parse(diff),
          family.contains('norm') ? 0 : lessThan(0.02),
          reason: family,
        );
      }
      expect(
        Directory(
          bundle,
        ).listSync().map((e) => e.uri.pathSegments.last).toSet(),
        {
          'config.json',
          'tokenizer.json',
          'model.safetensors',
          'qwen3-q8.cache',
        },
      );

      final cache = Q8Cache.open(
        '$bundle/qwen3-q8.cache',
        Q8Cache.keyFor(bundle),
      )!;
      final gguf = GgufFile.open('${root.path}/model.gguf');
      addTearDown(gguf.close);
      final blocks = gguf.bytes(gguf.tensors['blk.1.ffn_down.weight']!);
      final down = cache.matrix('model.layers.1.feed_forward.w2', 32, 64)!;
      expect(down.qsBytes, splitQ8_0(blocks, 32, 64).qsBytes);
      expect(down.scaleBytes, splitQ8_0(blocks, 32, 64).scaleBytes);

      final model = await LocalModel.load(
        bundle,
        backend: InferenceBackend.cpu,
        contextLength: 64,
        threads: 2,
      );
      addTearDown(model.close);
      final turn = await model.chat(maxNewTokens: 2).send({
        'role': 'user',
        'content': 'hi',
      });
      expect(turn.generatedTokens, greaterThan(0));
    });

    test('rejects a GGUF of another Qwen3', () {
      final other = '${root.path}/other';
      Directory(other).createSync();
      for (final name in ['config.json', 'tokenizer.json']) {
        File('${root.path}/hf/$name').copySync('$other/$name');
      }
      final config = File('$other/config.json');
      config.writeAsStringSync(
        config.readAsStringSync().replaceFirst(
          '"num_hidden_layers":2',
          '"num_hidden_layers":3',
        ),
      );
      expect(
        () => importGguf('${root.path}/model.gguf', other, '${root.path}/x'),
        throwsA(isA<StateError>()),
      );
    });
  }, skip: NativeKernels.isAvailable ? false : 'requires native kernels');
}
