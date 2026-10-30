// Closing a model gives back everything it took. Needs a Q4 bundle in
// DIANOIA_LFM3_DIR and the GPU of an Apple silicon Mac; skipped otherwise.
import 'dart:ffi';
import 'dart:io';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

final _modelDir = Platform.environment['DIANOIA_LFM3_DIR'];

Future<LocalModel> _loadAndChat() async {
  final model = await LocalModel.load(_modelDir!);
  final turn = await model.chat(maxNewTokens: 4).send({
    'role': 'user',
    'content': 'Hi.',
  });
  expect(turn.generatedTokens, greaterThan(0));
  return model;
}

void main() {
  if (_modelDir == null || Abi.current() != Abi.macosArm64) {
    test('close', () {}, skip: 'set DIANOIA_LFM3_DIR on an Apple silicon Mac');
    return;
  }
  test('close gives back every GPU buffer and mapping', () async {
    // A first cycle settles what the process keeps for any model.
    await (await _loadAndChat()).close();
    final buffers = GpuForward.liveBuffers;
    final footprint = physFootprint();
    for (var i = 0; i < 4; i++) {
      final model = await _loadAndChat();
      expect(model.backend, InferenceBackend.gpu);
      expect(GpuForward.liveBuffers, greaterThan(buffers));
      await model.close();
      expect(GpuForward.liveBuffers, buffers, reason: 'cycle $i');
      expect(MappedFile.isMapped('$_modelDir/lfm3-q4.cache'), isFalse);
      expect(MappedFile.isMapped('$_modelDir/model.safetensors'), isFalse);
    }
    // Measured over these four cycles at the default context: 13-15 MB of
    // heap noise when every buffer is released, 409 MB when the GPU
    // release is skipped (the activations and KV caches of each model).
    final growth = (physFootprint() - footprint) / (1 << 20);
    expect(growth, lessThan(128), reason: 'grew $growth MB in four cycles');
  });

  test('a closed model refuses to run', () async {
    final model = await LocalModel.load(_modelDir!, contextLength: 512);
    final chat = model.chat(maxNewTokens: 4);
    await model.close();
    await model.close();
    expect(model.isClosed, isTrue);
    await expectLater(
      chat.send({'role': 'user', 'content': 'Hi.'}),
      throwsStateError,
    );
    expect(chat.messages, isEmpty);
  });

  test('two models share the cache mapping until both close', () async {
    final a = await LocalModel.load(_modelDir!, contextLength: 512);
    final b = await LocalModel.load(_modelDir!, contextLength: 512);
    await a.close();
    expect(MappedFile.isMapped('$_modelDir/lfm3-q4.cache'), isTrue);
    final turn = await b.chat(maxNewTokens: 4).send({
      'role': 'user',
      'content': 'Hi.',
    });
    expect(turn.generatedTokens, greaterThan(0));
    await b.close();
    expect(MappedFile.isMapped('$_modelDir/lfm3-q4.cache'), isFalse);
  });
}
