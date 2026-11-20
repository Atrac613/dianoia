// ModelStore installing a Qwen3 ModelSpec end to end: a tiny synthetic
// Qwen3 Q8_0 GGUF and its Hugging Face files served over HTTP, imported on
// install and opened with LocalModel.
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:dianoia/dianoia.dart';
import 'package:test/test.dart';

import 'support/synthetic_qwen3.dart';

Future<HttpServer> _serve(String dir) async {
  final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
  server.listen((request) async {
    final file = File('$dir/${request.uri.pathSegments.last}');
    if (!file.existsSync()) {
      request.response.statusCode = HttpStatus.notFound;
    } else {
      request.response.contentLength = file.lengthSync();
      await request.response.addStream(file.openRead());
    }
    await request.response.close();
  });
  return server;
}

void main() {
  test('the Qwen3 specs pin Qwen\'s Q8_0 GGUFs', () {
    const small = ModelSpec.qwen3_0_6bQ8, large = ModelSpec.qwen3_1_7bQ8;
    expect(small.files.map((f) => f.name).toSet(), {
      'config.json',
      'generation_config.json',
      'tokenizer.json',
      'LICENSE',
      'Qwen3-0.6B-Q8_0.gguf',
    });
    expect(small.downloadBytes, 650881650);
    expect(large.downloadBytes, 1845860978);
    expect(
      small.gguf.url,
      contains('23749fefcc72300e3a2ad315e1317431b06b590a'),
    );
    expect(
      large.gguf.url,
      contains('90862c4b9d2787eaed51d12237eafdfe7c5f6077'),
    );
    expect(
      large.config.url,
      contains('70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
    );
    expect(
      small.fromMirror('https://mirror.invalid/').generationConfig?.url,
      'https://mirror.invalid/generation_config.json',
    );
  });

  test('rejects GGUF specs whose metadata files are misnamed', () {
    const file = RemoteFile(
      'generation.json',
      'https://example.invalid/generation.json',
      size: 1,
      sha256: '',
    );
    final spec = ModelSpec(
      id: 'misnamed',
      name: 'Misnamed',
      license: 'test',
      licenseUrl: 'https://example.invalid/',
      gguf: ModelSpec.qwen3_0_6bQ8.gguf,
      config: ModelSpec.qwen3_0_6bQ8.config,
      tokenizer: ModelSpec.qwen3_0_6bQ8.tokenizer,
      licenseFile: ModelSpec.qwen3_0_6bQ8.licenseFile,
      generationConfig: file,
    );
    final root = Directory.systemTemp.createTempSync('dianoia-misnamed-');
    addTearDown(() => root.deleteSync(recursive: true));
    expect(() => ModelStore(root.path).install(spec), throwsArgumentError);
  });

  test('installs and loads a Qwen3 GGUF', () async {
    final root = Directory.systemTemp.createTempSync('dianoia-text-store-');
    addTearDown(() => root.deleteSync(recursive: true));
    final source = '${root.path}/source';
    writeSyntheticQwen3(source);
    final server = await _serve(source);
    addTearDown(() => server.close(force: true));

    writeSyntheticQwen3Gguf(source, '$source/model.gguf');
    RemoteFile pinned(String name) => RemoteFile(
      name,
      'http://unused.invalid/$name',
      size: File('$source/$name').lengthSync(),
      sha256: sha256
          .convert(File('$source/$name').readAsBytesSync())
          .toString(),
    );
    final spec = ModelSpec(
      id: 'tiny-qwen3-q8',
      name: 'Tiny Qwen3',
      license: 'test',
      licenseUrl: 'https://example.invalid/',
      gguf: pinned('model.gguf'),
      config: pinned('config.json'),
      tokenizer: pinned('tokenizer.json'),
      licenseFile: pinned('LICENSE'),
    ).fromMirror('http://${server.address.address}:${server.port}/');

    final store = ModelStore('${root.path}/store');
    expect(store.isInstalled(spec), isFalse);
    final stages = await store.install(spec).map((p) => p.stage).toSet();
    expect(stages, containsAll([InstallStage.building, InstallStage.done]));
    expect(store.isInstalled(spec), isTrue);

    final bundle = store.bundlePath(spec);
    expect(
      Directory(bundle).listSync().map((e) => e.uri.pathSegments.last).toSet(),
      {
        'config.json',
        'tokenizer.json',
        'LICENSE',
        'model.safetensors',
        'qwen3-q8.cache',
      },
    );
    for (final gone in ['downloads', 'bundle.partial']) {
      expect(
        Directory('${root.path}/store/${spec.id}/$gone').existsSync(),
        isFalse,
      );
    }

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
  }, skip: LocalModel.isSupported ? false : 'requires native kernels');
}
