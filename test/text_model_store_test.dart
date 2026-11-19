// ModelStore installing a TextCheckpointSpec end to end: a tiny synthetic
// Qwen3 checkpoint served over HTTP, quantized on install and opened with
// LocalModel.
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
  test('the Qwen3-0.6B spec pins five files', () {
    const spec = TextCheckpointSpec.qwen3_0_6bQ8;
    expect(spec.files.map((f) => f.name).toSet(), {
      'config.json',
      'generation_config.json',
      'tokenizer.json',
      'LICENSE',
      'model.safetensors',
    });
    expect(spec.downloadBytes, 1514735290);
    expect(spec.quantization, TextQuantization.q8);
    for (final f in spec.files) {
      expect(f.url, contains('c1899de289a04d12100db370d81485cdf75e47ca'));
    }
  });

  test('the Qwen3-1.7B spec pins its shards and index', () {
    const spec = TextCheckpointSpec.qwen3_1_7bQ8;
    expect(spec.downloadBytes, 4074976159);
    expect(
      spec.files.map((f) => f.name),
      containsAll([
        'model.safetensors.index.json',
        'model-00001-of-00002.safetensors',
        'model-00002-of-00002.safetensors',
      ]),
    );
    for (final f in spec.files) {
      expect(f.url, contains('70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'));
    }
  });

  test('installs, quantizes and loads a Qwen3 checkpoint', () async {
    final root = Directory.systemTemp.createTempSync('dianoia-text-store-');
    addTearDown(() => root.deleteSync(recursive: true));
    final source = '${root.path}/source';
    writeSyntheticQwen3(source);
    final server = await _serve(source);
    addTearDown(() => server.close(force: true));

    final spec = TextCheckpointSpec(
      id: 'tiny-qwen3-q8',
      name: 'Tiny Qwen3',
      license: 'test',
      licenseUrl: 'https://example.invalid/',
      files: [
        for (final name in [
          'config.json',
          'tokenizer.json',
          'LICENSE',
          'model.safetensors',
        ])
          RemoteFile(
            name,
            'http://unused.invalid/$name',
            size: File('$source/$name').lengthSync(),
            sha256: sha256
                .convert(File('$source/$name').readAsBytesSync())
                .toString(),
          ),
      ],
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
    for (final gone in ['downloads', 'staging', 'bundle.partial']) {
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
