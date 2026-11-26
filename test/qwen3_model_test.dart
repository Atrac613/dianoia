// A tiny synthetic Qwen3 checkpoint, imported and run in float32: prompt
// reuse across a diverging conversation gives the logits of a fresh
// prefill, and keeps every common token since the model has no conv state.
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/synthetic_qwen3.dart';

void main() {
  late Directory root;
  late String dir;
  late TextModel model;

  setUpAll(() async {
    root = Directory.systemTemp.createTempSync('dianoia-qwen3-model-');
    writeSyntheticQwen3('${root.path}/hf');
    dir = '${root.path}/model';
    importHfCheckpoint('${root.path}/hf', dir);
    model = await TextModel.load(dir, maxSeq: 64, maxChunk: 4);
  });
  tearDownAll(() async {
    await model.close();
    root.deleteSync(recursive: true);
  });

  test('loads as ungated attention without sandwich norms', () {
    expect(model.config.architecture, ModelArchitecture.qwen3);
    expect(model.config.attentionOutputGate, isFalse);
    expect(model.config.sandwichNorm, isFalse);
  });

  test('a diverging prompt reuses every common token', () async {
    final first = [5, 9, 12, 7, 30, 31, 2, 8, 40, 41, 42];
    final second = [5, 9, 12, 7, 30, 31, 3, 8, 40, 50];
    await model.reset();
    await model.prefill(first, checkpoints: [2]);
    expect(await model.prefill(second, checkpoints: [2]), 6);
    final reused = Float32List.fromList(model.logits);

    // The same chunks from scratch: the first prefill's up to the
    // divergence, then the rest.
    await model.reset();
    await model.forward(second.sublist(0, 2), computeLogits: false);
    await model.forward(second.sublist(2, 6), computeLogits: false);
    await model.forward(second.sublist(6));
    expect(model.logits, orderedEquals(reused));
    expect(reused.any((v) => v != 0), isTrue);
  });

  test('builds a Q8 bundle that LocalModel opens', () async {
    final bundle = '${root.path}/bundle';
    await buildTextBundle(
      '${root.path}/hf',
      bundle,
      stagingDir: '${root.path}/staging',
      weights: WeightFormat.q8,
    );
    expect(Directory('${root.path}/staging').existsSync(), isFalse);
    expect(File('$bundle/qwen3-q8.cache').existsSync(), isTrue);
    expect(File('$bundle/lfm3-q8.cache').existsSync(), isFalse);
    final local = await LocalModel.load(
      bundle,
      backend: InferenceBackend.cpu,
      contextLength: 64,
      threads: 2,
    );
    addTearDown(local.close);
    final turn = await local.chat(maxNewTokens: 3).send({
      'role': 'user',
      'content': 'hi',
    });
    expect(turn.generatedTokens, greaterThan(0));
    // Thinking replies may run to the end of the context by default.
    expect(local.chat().maxNew, 512);
    expect(local.chat(enableThinking: true).maxNew, 64);
    expect(local.chat(enableThinking: true, maxNewTokens: 9).maxNew, 9);

    // Without a context length, Qwen3 gets 4096 tokens.
    final defaults = await LocalModel.load(
      bundle,
      backend: InferenceBackend.cpu,
      threads: 2,
    );
    addTearDown(defaults.close);
    expect(defaults.contextLength, 4096);
  }, skip: NativeKernels.isAvailable ? false : 'requires native kernels');
}
