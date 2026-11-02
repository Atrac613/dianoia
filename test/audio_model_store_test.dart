// Tiny generated GGUFs served over loopback; no external model downloads.
import 'dart:async';
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
    test('audio installation', () {}, skip: 'requires native kernels');
    return;
  }
  late Directory fixture, root;
  late Map<String, Uint8List> original, files;
  late HttpServer server;
  late ModelStore store;
  late AudioGgufSpec spec;
  final requests = <(String, String?)>[];
  AudioGgufSpec catalog() {
    RemoteFile f(String name) => RemoteFile(
      name,
      'http://127.0.0.1:${server.port}/$name',
      size: files[name]!.length,
      sha256: sha256.convert(files[name]!).toString(),
    );
    return AudioGgufSpec(
      id: 'tiny-audio',
      name: 'Synthetic audio',
      language: f('language.gguf'),
      audio: f('audio.gguf'),
      config: f('audio_config.json'),
      tokenizer: f('tokenizer.json'),
      licenseFile: f('LICENSE'),
    );
  }

  setUpAll(() async {
    fixture = Directory.systemTemp.createTempSync('dianoia-audio-source-');
    await prepareVisionFixture(fixture, hiddenSize: 256);
    final model = '${fixture.path}/model';
    writeVisionGgufFixture(
      model,
      '${fixture.path}/language.gguf',
      '${fixture.path}/unused.gguf',
    );
    final encoder =
        jsonDecode(
              File(
                'test/fixtures/audio/encoder_oracle.json',
              ).readAsStringSync(),
            )
            as Map;
    final generator =
        jsonDecode(
              File(
                'test/fixtures/audio/generation_oracle.json',
              ).readAsStringSync(),
            )
            as Map;
    final adapted = <String, List<int>>{
      'mm.a.mlp.1.weight': [256, 32],
      'mm.a.mlp.1.bias': [256],
      'mm.a.mlp.3.weight': [256, 256],
      'mm.a.mlp.3.bias': [256],
      'a.gen.code.proj_in.weight': [12 * 32, 256],
      'a.gen.code.out_embd.weight': [12 * 33, 256],
    };
    writeTestGguf(
      '${fixture.path}/audio.gguf',
      {
        ...(encoder['metadata'] as Map).cast<String, Object>(),
        ...(generator['metadata'] as Map).cast<String, Object>(),
        'clip.audio.projection_dim': 256,
        'clip.gen.audio.projection_dim': 256,
      },
      [
        for (final t in [
          ...encoder['tensors'] as List,
          ...generator['tensors'] as List,
        ])
          (
            t['name'] as String,
            adapted[t['name']] ?? (t['shape'] as List).cast<int>(),
            GgufType.values.byName(t['type'] as String),
            adapted.containsKey(t['name'])
                ? Uint8List(
                    adapted[t['name']]!.fold(1, (int a, b) => a * b) ~/
                        GgufType.values
                            .byName(t['type'] as String)
                            .blockValues *
                        GgufType.values.byName(t['type'] as String).blockBytes,
                  )
                : base64Decode(t['bytes'] as String),
          ),
      ],
    );
    final config =
        jsonDecode(File('$model/config.json').readAsStringSync()) as Map;
    original = {
      'audio_config.json': Uint8List.fromList(
        utf8.encode(
          jsonEncode({
            'model_type': 'lfm3_audio',
            'format_version': 1,
            'lfm': config['text_config'],
          }),
        ),
      ),
      for (final n in ['tokenizer.json', 'LICENSE'])
        n: File('$model/$n').readAsBytesSync(),
      for (final n in ['language.gguf', 'audio.gguf'])
        n: File('${fixture.path}/$n').readAsBytesSync(),
    };
  });
  tearDownAll(() => fixture.deleteSync(recursive: true));
  setUp(() async {
    root = Directory.systemTemp.createTempSync('dianoia-audio-install-');
    files = Map.from(original);
    requests.clear();
    server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    server.listen((r) async {
      final name = r.uri.pathSegments.last;
      final bytes = files[name];
      final range = r.headers.value(HttpHeaders.rangeHeader);
      requests.add((name, range));
      if (bytes == null) {
        r.response.statusCode = 404;
        await r.response.close();
        return;
      }
      final from = range == null
          ? 0
          : int.parse(RegExp(r'bytes=(\d+)-').firstMatch(range)![1]!);
      if (range != null) {
        r.response.statusCode = 206;
        r.response.headers.set(
          HttpHeaders.contentRangeHeader,
          'bytes $from-${bytes.length - 1}/${bytes.length}',
        );
      }
      r.response.contentLength = bytes.length - from;
      try {
        for (var at = from; at < bytes.length; at += 65536) {
          r.response.add(
            bytes.sublist(at, (at + 65536).clamp(0, bytes.length)),
          );
          await r.response.flush();
          await Future<void>.delayed(const Duration(milliseconds: 1));
        }
        await r.response.close();
      } on IOException {
        /* Expected on cancel. */
      }
    });
    spec = catalog();
    store = ModelStore('${root.path}/store');
  });
  tearDown(() async {
    await server.close(force: true);
    root.deleteSync(recursive: true);
  });

  test(
    'resumes cancelled HTTP, imports Q4 and verifies installed payloads',
    () async {
      final reached = Completer<void>();
      final subscription = store.install(spec).listen((p) {
        if (p.stage == InstallStage.downloading &&
            p.bytes > 1 << 20 &&
            !reached.isCompleted) {
          reached.complete();
        }
      });
      expect(() => store.install(spec), throwsStateError);
      await reached.future;
      await subscription.cancel();
      expect(store.isInstalled(spec), isFalse);
      final downloads = '${store.directory}/${spec.id}/downloads';
      final part = File('$downloads/language.gguf.part');
      expect(part.lengthSync(), greaterThan(0));
      final progress = await store.install(spec).toList();
      expect(progress.last.stage, InstallStage.done);
      expect(
        requests.any((r) => r.$1 == 'language.gguf' && r.$2 != null),
        isTrue,
      );
      expect(store.isInstalled(spec), isTrue);
      final bundle = store.bundlePath(spec);
      await validateAudioBundle(bundle, spec.files, spec.audio.name);
      expect(Directory(downloads).existsSync(), isFalse);
      expect(File('$bundle/language.gguf').existsSync(), isFalse);
      expect(Directory('$bundle.partial').existsSync(), isFalse);
      final model = await Lfm3Model.load(
        '$bundle/language',
        maxSeq: 64,
        weights: WeightFormat.q4,
        q8CachePath: '$bundle/language/lfm3-q4.cache',
      );
      await model.step(1);
      expect(model.logits.every((x) => x.isFinite), isTrue);
      await model.close();
      final count = requests.length;
      await store.install(spec).drain<void>();
      expect(requests.length, count);
      final cache = File('$bundle/language/lfm3-q4.cache');
      final data = cache.readAsBytesSync();
      data[data.length - 1] ^= 1;
      cache.writeAsBytesSync(data);
      await expectLater(
        validateAudioBundle(bundle, spec.files, spec.audio.name),
        throwsFormatException,
      );
    },
  );

  test(
    'conversion cancellation does not publish and a retry removes stale staging',
    () async {
      final building = Completer<void>();
      final subscription = store.install(spec).listen((p) {
        if (p.stage == InstallStage.building && !building.isCompleted) {
          building.complete();
        }
      });
      await building.future;
      await subscription.cancel();
      expect(store.isInstalled(spec), isFalse);
      expect(
        Directory('${store.bundlePath(spec)}.partial').existsSync(),
        isFalse,
      );
      expect(
        Directory('${store.directory}/${spec.id}/downloads').existsSync(),
        isTrue,
      );
      final stale = Directory('${store.bundlePath(spec)}.partial/.metadata')
        ..createSync(recursive: true);
      File(
        '${stale.path}/stale',
      ).writeAsStringSync('interrupted previous process');
      await store.install(spec).drain<void>();
      expect(store.isInstalled(spec), isTrue);
      expect(stale.existsSync(), isFalse);
    },
  );

  test('hash mismatch and invalid graphs never publish a bundle', () async {
    files['audio.gguf'] = Uint8List.fromList(files['audio.gguf']!)..[0] ^= 1;
    await expectLater(
      store.install(spec).drain<void>(),
      throwsA(isA<ModelDownloadException>()),
    );
    expect(store.isInstalled(spec), isFalse);
    writeTestGguf('${root.path}/bad.gguf', {}, []);
    files['audio.gguf'] = File('${root.path}/bad.gguf').readAsBytesSync();
    spec = catalog();
    await expectLater(store.install(spec).drain<void>(), throwsFormatException);
    expect(Directory(store.bundlePath(spec)).existsSync(), isFalse);
    expect(
      Directory('${store.bundlePath(spec)}.partial').existsSync(),
      isFalse,
    );
  });

  test('official catalog pins and mirror preserve identity', () {
    const official = AudioGgufSpec.lfm3AudioQ4;
    expect(official.downloadBytes, 1228932715);
    final mirrored = official.fromMirror('https://example.com/audio');
    expect(
      audioBundleSource(mirrored.files),
      audioBundleSource(official.files),
    );
    expect(
      mirrored.audio.url,
      'https://example.com/audio/${official.audio.name}',
    );
    expect(() => official.fromMirror('file:///tmp/model'), throwsArgumentError);
  });
}
