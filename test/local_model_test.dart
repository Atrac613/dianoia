// LocalModel: which bundles it accepts and where it runs them. The loading
// tests need a Q4 bundle in DIANOIA_LFM3_DIR and are skipped without one.
import 'dart:ffi';
import 'dart:io';

import 'package:dianoia/dianoia.dart';
import 'package:test/test.dart';

final _modelDir = Platform.environment['DIANOIA_LFM3_DIR'];

void main() {
  test('a directory without quantized weights is rejected', () async {
    final empty = Directory.systemTemp.createTempSync('dianoia_empty');
    addTearDown(() => empty.deleteSync());
    await expectLater(
      LocalModel.load(empty.path),
      throwsA(
        isA<ArgumentError>().having(
          (e) => e.message,
          'message',
          contains('import-gguf'),
        ),
      ),
    );
  });

  group('with a model', () {
    test('auto runs on the GPU of an Apple silicon Mac', () async {
      final model = await LocalModel.load(_modelDir!, contextLength: 256);
      addTearDown(model.close);
      expect(model.backend, InferenceBackend.gpu);
      expect(model.contextLength, 256);
    }, skip: Abi.current() == Abi.macosArm64 ? null : 'needs a Mac GPU');

    test('cpu runs on the CPU kernels and answers', () async {
      final model = await LocalModel.load(
        _modelDir!,
        backend: InferenceBackend.cpu,
        contextLength: 256,
      );
      addTearDown(model.close);
      expect(model.backend, InferenceBackend.cpu);
      final turn = await model.chat(maxNewTokens: 8).send({
        'role': 'user',
        'content': 'Say hi.',
      });
      expect(turn.generatedTokens, greaterThan(0));
    });
  }, skip: _modelDir == null ? 'set DIANOIA_LFM3_DIR to a Q4 bundle' : false);
}
