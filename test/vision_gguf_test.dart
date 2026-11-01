import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:dianoia/lab.dart';
import 'package:test/test.dart';
import 'support/vision_model_fixture.dart';
import 'support/gguf_fixture.dart';

void main() {
  if (!NativeKernels.isAvailable) {
    test('QAD GGUF import', () {}, skip: 'requires native kernels');
    return;
  }
  late Directory root;
  late Map<String, List<int>> files;
  late HttpServer server;
  late VisionGgufSpec spec;
  late ModelStore store;
  setUp(() async {
    root = Directory.systemTemp.createTempSync('dianoia-qad-test-');
    await prepareVisionFixture(root, hiddenSize: 256);
    final model = '${root.path}/model';
    writeVisionGgufFixture(
      model,
      '${root.path}/language.gguf',
      '${root.path}/vision.gguf',
    );
    files = {
      for (final n in [
        'config.json',
        'processor_config.json',
        'tokenizer.json',
        'LICENSE',
      ])
        n: File('$model/$n').readAsBytesSync(),
      'language.gguf': File('${root.path}/language.gguf').readAsBytesSync(),
      'vision.gguf': File('${root.path}/vision.gguf').readAsBytesSync(),
    };
    server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    server.listen((r) async {
      final data = files[r.uri.pathSegments.last];
      if (data == null) {
        r.response.statusCode = 404;
      } else {
        r.response.contentLength = data.length;
        r.response.add(data);
      }
      await r.response.close();
    });
    RemoteFile f(String n) => RemoteFile(
      n,
      'http://127.0.0.1:${server.port}/$n',
      size: files[n]!.length,
      sha256: sha256.convert(files[n]!).toString(),
    );
    spec = VisionGgufSpec(
      id: 'tiny-qad',
      name: 'Tiny QAD',
      license: 'Synthetic',
      licenseUrl: 'https://example.com',
      language: f('language.gguf'),
      vision: f('vision.gguf'),
      config: f('config.json'),
      processor: f('processor_config.json'),
      tokenizer: f('tokenizer.json'),
      licenseFile: f('LICENSE'),
    );
    store = ModelStore('${root.path}/store');
  });
  tearDown(() async {
    await server.close(force: true);
    root.deleteSync(recursive: true);
  });

  test(
    'downloads paired GGUFs and runs a cache-embedding vision bundle',
    () async {
      await store.install(spec).drain<void>();
      expect(store.isInstalled(spec), isTrue);
      final path = store.bundlePath(spec);
      final manifest = VlBundleManifest.parse(
        File('$path/vl_bundle.json').readAsStringSync(),
      );
      expect(manifest.inputEmbedding, 'q8-cache');
      expect(manifest.weights, 'q4');
      expect(
        Directory('${root.path}/store/${spec.id}/downloads').existsSync(),
        isFalse,
      );
      expect(
        Directory(
          '${root.path}/store/${spec.id}',
        ).listSync().where((e) => e.path.contains('.dianoia-vl-qad-')),
        isEmpty,
      );
      final st = SafetensorsFile.open('$path/language/model.safetensors');
      try {
        expect(st.tensors.containsKey('model.embed_tokens.weight'), isFalse);
      } finally {
        st.close();
      }
      final model = await LocalVisionModel.loadBundle(
        path,
        backend: InferenceBackend.cpu,
        contextLength: 512,
      );
      try {
        final turn = await model
            .chat(maxNewTokens: 3)
            .send(
              'Describe.',
              images: [
                ChatImage.rgb(Uint8List(8 * 8 * 3), width: 8, height: 8),
              ],
            );
        expect(turn.generatedTokens, greaterThan(0));
      } finally {
        await model.close();
      }
      final mirror = spec.fromMirror('https://example.com/qad');
      expect(mirror.language.sha256, spec.language.sha256);
      expect(mirror.vision.url, 'https://example.com/qad/vision.gguf');
      final json =
          jsonDecode(File('$path/vl_bundle.json').readAsStringSync()) as Map;
      final marked = Map<String, dynamic>.from(json);
      json.remove('input_embedding');
      File('$path/vl_bundle.json').writeAsStringSync(jsonEncode(json));
      await expectLater(validateVlBundle(path), throwsFormatException);
      json.addAll(marked);
      json['input_embedding'] = 'unknown';
      File('$path/vl_bundle.json').writeAsStringSync(jsonEncode(json));
      expect(store.isInstalled(spec), isFalse);
      await expectLater(validateVlBundle(path), throwsFormatException);
    },
  );

  test('cancel setup leaves no published bundle and can resume', () async {
    final stopped = Completer<void>();
    late StreamSubscription<InstallProgress> sub;
    sub = store.install(spec).listen((p) {
      if (p.stage == InstallStage.building) {
        sub.cancel().then(
          (_) => stopped.complete(),
          onError: stopped.completeError,
        );
      }
    }, onError: stopped.completeError);
    await stopped.future;
    expect(store.isInstalled(spec), isFalse);
    expect(
      Directory('${store.bundlePath(spec)}.partial').existsSync(),
      isFalse,
    );
    await store.install(spec).drain<void>();
    expect(store.isInstalled(spec), isTrue);
  });

  test('vision payload ordering and malformed input clean up mappings', () {
    final config = Lfm3VlConfig.load('${root.path}/model');
    final output = '${root.path}/vision.safetensors';
    importVlVisionGguf('${root.path}/vision.gguf', config, output);
    final st = SafetensorsFile.open(output);
    try {
      final patch = st.readF32(
        'model.vision_tower.patchifier.weight',
        lfm3VisionTensorShapes(config)['model.vision_tower.patchifier.weight']!,
      );
      final area = config.vision.patchSize * config.vision.patchSize;
      for (var row = 0; row < config.vision.hiddenSize; row++) {
        for (var channel = 0; channel < 3; channel++) {
          final at = (row * 3 + channel) * 2 * area;
          expect(patch.sublist(at, at + area), everyElement(1.5));
          expect(patch.sublist(at + area, at + 2 * area), everyElement(-2.5));
        }
      }
    } finally {
      st.close();
    }
    expect(MappedFile.isMapped('${root.path}/vision.gguf'), isFalse);
    expect(
      () => importVlVisionGguf(
        '${root.path}/language.gguf',
        config,
        '${root.path}/bad',
      ),
      throwsFormatException,
    );
    expect(MappedFile.isMapped('${root.path}/language.gguf'), isFalse);
  });
  test('QAD catalog pins both GGUFs and metadata separately from PTQ', () {
    final catalog = VisionGgufSpec.lfm3VlQadQ4;
    expect(catalog.downloadBytes, 1446534952);
    expect(catalog.id, isNot(VisionCheckpointSpec.lfm3VlQ4.id));
    expect(
      catalog.language.url,
      contains('b36554cfd35854b06429900131730cd80f092cdb'),
    );
    expect(
      catalog.vision.url,
      contains('b36554cfd35854b06429900131730cd80f092cdb'),
    );
    expect(() => catalog.fromMirror('file:///tmp/model'), throwsArgumentError);
    final invalid = File('${root.path}/invalid.gguf')
      ..writeAsStringSync('GGUF');
    expect(() => GgufFile.open(invalid.path), throwsFormatException);
    expect(MappedFile.isMapped(invalid.path), isFalse);
  });
}
