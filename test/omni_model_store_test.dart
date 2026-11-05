// Tiny generated GGUFs served over loopback; no external model downloads.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/gguf_fixture.dart';
import 'support/vision_model_fixture.dart';

void main() {
  if (!NativeKernels.isAvailable) {
    test('omni installation', () {}, skip: 'requires native kernels');
    return;
  }
  late Directory fixture, root;
  late Map<String, Uint8List> files;
  late HttpServer server;

  OmniGgufSpec catalog() {
    RemoteFile f(String name) => RemoteFile(
      name,
      'http://127.0.0.1:${server.port}/$name',
      size: files[name]!.length,
      sha256: sha256.convert(files[name]!).toString(),
    );
    return OmniGgufSpec(
      id: 'tiny-omni',
      name: 'Synthetic omni',
      language: f('language.gguf'),
      mmproj: f('omni.gguf'),
      config: f('omni_config.json'),
      processor: f('processor_config.json'),
      tokenizer: f('tokenizer.json'),
      licenseFile: f('LICENSE'),
    );
  }

  setUpAll(() async {
    fixture = Directory.systemTemp.createTempSync('dianoia-omni-source-');
    await prepareVisionFixture(fixture, hiddenSize: 256);
    final model = '${fixture.path}/model';
    writeVisionGgufFixture(
      model,
      '${fixture.path}/language.gguf',
      '${fixture.path}/vision.gguf',
    );
    Map oracle(String name) =>
        jsonDecode(File('test/fixtures/audio/$name').readAsStringSync()) as Map;
    final encoder = oracle('encoder_oracle.json');
    final generator = oracle('generation_oracle.json');
    final adapted = <String, List<int>>{
      'mm.a.mlp.1.weight': [256, 32],
      'mm.a.mlp.1.bias': [256],
      'mm.a.mlp.3.weight': [256, 256],
      'mm.a.mlp.3.bias': [256],
      'a.gen.code.proj_in.weight': [12 * 32, 256],
      'a.gen.code.out_embd.weight': [12 * 33, 256],
    };
    final vision = GgufFile.open('${fixture.path}/vision.gguf');
    try {
      writeTestGguf(
        '${fixture.path}/omni.gguf',
        {
          ...(encoder['metadata'] as Map).cast<String, Object>(),
          ...(generator['metadata'] as Map).cast<String, Object>(),
          'clip.audio.projection_dim': 256,
          'clip.gen.audio.projection_dim': 256,
          'clip.vision.projector_type': 'lfm3vl',
          ...lfm3AudioSamplingMetadata,
        },
        [
          for (final t in [
            ...encoder['tensors'] as List,
            ...generator['tensors'] as List,
          ])
            () {
              final type = GgufType.values.byName(t['type'] as String);
              final shape = adapted[t['name']];
              return (
                t['name'] as String,
                shape ?? (t['shape'] as List).cast<int>(),
                type,
                shape != null
                    ? Uint8List(
                        shape.fold(1, (int a, b) => a * b) ~/
                            type.blockValues *
                            type.blockBytes,
                      )
                    : base64Decode(t['bytes'] as String),
              );
            }(),
          for (final t in vision.tensors.values)
            (
              t.name,
              t.dims.reversed.toList(),
              t.type,
              Uint8List.fromList(vision.bytes(t)),
            ),
        ],
      );
    } finally {
      vision.close();
    }
    final config =
        jsonDecode(File('$model/config.json').readAsStringSync())
            as Map<String, Object?>;
    files = {
      'omni_config.json': utf8.encode(
        jsonEncode(
          {...config, 'model_type': 'lfm3_omni'}..removeWhere(
            (k, _) => k.startsWith('vision_s') || k == 'vision_end_token_id',
          ),
        ),
      ),
      'processor_config.json': File(
        '$model/processor_config.json',
      ).readAsBytesSync(),
      for (final n in ['tokenizer.json', 'LICENSE'])
        n: File('$model/$n').readAsBytesSync(),
      for (final n in ['language.gguf', 'omni.gguf'])
        n: File('${fixture.path}/$n').readAsBytesSync(),
    };
  });
  tearDownAll(() => fixture.deleteSync(recursive: true));
  setUp(() async {
    root = Directory.systemTemp.createTempSync('dianoia-omni-install-');
    server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    server.listen((r) async {
      final bytes = files[r.uri.pathSegments.last];
      if (bytes == null) {
        r.response.statusCode = 404;
      } else {
        r.response.contentLength = bytes.length;
        r.response.add(bytes);
      }
      await r.response.close();
    });
  });
  tearDown(() async {
    await server.close(force: true);
    root.deleteSync(recursive: true);
  });

  test('installs a research-layout bundle and verifies its payloads', () async {
    final spec = catalog();
    final store = ModelStore('${root.path}/store');
    final progress = await store.install(spec).toList();
    expect(progress.last.stage, InstallStage.done);
    expect(store.isInstalled(spec), isTrue);
    final bundle = store.bundlePath(spec);
    await validateOmniBundle(bundle, spec.files, spec.mmproj.name);
    for (final name in [
      'omni_config.json',
      'processor_config.json',
      'omni.gguf',
      'language/lfm3-q4.cache',
      'language/tokenizer.json',
      'language/omni_source.json',
    ]) {
      expect(File('$bundle/$name').existsSync(), isTrue, reason: name);
    }
    expect(
      Directory('${store.directory}/${spec.id}/downloads').existsSync(),
      isFalse,
    );
    final tokenizer = Lfm3Tokenizer.load('$bundle/language');
    final config = loadOmniVisionConfig('$bundle/omni_config.json', tokenizer);
    loadVisionEncoderGguf('$bundle/omni.gguf', config).close();
    final cache = File('$bundle/language/lfm3-q4.cache');
    final data = cache.readAsBytesSync();
    data[data.length - 1] ^= 1;
    cache.writeAsBytesSync(data);
    await expectLater(
      validateOmniBundle(bundle, spec.files, spec.mmproj.name),
      throwsFormatException,
    );
  });

  test('an invalid mmproj never publishes a bundle', () async {
    final original = files['omni.gguf']!;
    try {
      writeTestGguf('${root.path}/bad.gguf', {}, []);
      files['omni.gguf'] = File('${root.path}/bad.gguf').readAsBytesSync();
      final spec = catalog();
      final store = ModelStore('${root.path}/store');
      await expectLater(
        store.install(spec).drain<void>(),
        throwsFormatException,
      );
      expect(Directory(store.bundlePath(spec)).existsSync(), isFalse);
      expect(
        Directory('${store.bundlePath(spec)}.partial').existsSync(),
        isFalse,
      );
    } finally {
      files['omni.gguf'] = original;
    }
  });

  test('official catalog pins and mirror preserve identity', () {
    const official = OmniGgufSpec.lfm3OmniQ4;
    expect(official.downloadBytes, 1338121958);
    final mirrored = official.fromMirror('https://example.com/omni');
    expect(
      [for (final f in mirrored.files) (f.name, f.size, f.sha256)],
      [for (final f in official.files) (f.name, f.size, f.sha256)],
    );
    expect(
      mirrored.mmproj.url,
      'https://example.com/omni/${official.mmproj.name}',
    );
    expect(() => official.fromMirror('file:///tmp/model'), throwsArgumentError);
  });
}
