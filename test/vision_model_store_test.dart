// Real HTTP installation with tiny runtime-generated VL tensors; no downloads
// from external hosts and no committed model weights.
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:dianoia/dianoia.dart';
import 'package:dianoia/lab.dart' show packVlBundle;
import 'package:test/test.dart';

import 'support/vision_model_fixture.dart';

final class _Server {
  _Server(this.files);
  final Map<String, List<int>> files;
  final requests = <(String, String?)>[];
  String? cutOnce;
  late HttpServer server;
  String get base => 'http://127.0.0.1:${server.port}/models/vl';

  Future<void> start() async {
    server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    server.listen((request) async {
      final name = request.uri.path.substring('/models/vl/'.length);
      final range = request.headers.value(HttpHeaders.rangeHeader);
      requests.add((name, range));
      final data = files[name];
      final response = request.response;
      if (data == null) {
        response.statusCode = HttpStatus.notFound;
        await response.close();
        return;
      }
      final from = range == null
          ? 0
          : int.parse(RegExp(r'bytes=(\d+)-').firstMatch(range)![1]!);
      if (range != null) {
        response
          ..statusCode = HttpStatus.partialContent
          ..headers.set(
            HttpHeaders.contentRangeHeader,
            'bytes $from-${data.length - 1}/${data.length}',
          );
      }
      response.contentLength = data.length - from;
      try {
        if (cutOnce == name) {
          cutOnce = null;
          final socket = await response.detachSocket();
          socket.add(data.sublist(from, from + (data.length - from) ~/ 2));
          await socket.flush();
          socket.destroy();
        } else {
          response.add(data.sublist(from));
          await response.close();
        }
      } on IOException {
        // Cancellation can close the client before the response finishes.
      }
    });
  }

  Future<void> close() => server.close(force: true);
}

void main() {
  if (!LocalModel.isSupported) {
    test('VL model store', () {}, skip: 'requires native kernels');
    return;
  }
  late Directory fixture;
  late Map<String, List<int>> bundle;
  setUpAll(() async {
    fixture = Directory.systemTemp.createTempSync('dianoia_vl_install_fixture');
    await prepareVisionFixture(fixture);
    final path = '${fixture.path}/packed';
    await packVlBundle(
      '${fixture.path}/model',
      '${fixture.path}/backbone',
      path,
    );
    bundle = {
      for (final file in Directory(
        path,
      ).listSync(recursive: true).whereType<File>())
        file.path.substring(path.length + 1): file.readAsBytesSync(),
    };
  });
  tearDownAll(() => fixture.deleteSync(recursive: true));

  late Directory root;
  late _Server server;
  late ModelStore store;
  late VisionModelSpec spec;
  VisionModelSpec makeSpec({String? json, String id = 'tiny-vl-q8'}) =>
      VisionModelSpec.fromManifest(
        id: id,
        name: 'Synthetic VL Q8',
        license: 'Synthetic test weights',
        licenseUrl: '${server.base}/LICENSE',
        baseUrl: server.base,
        manifestJson: json ?? utf8.decode(bundle['vl_bundle.json']!),
      );

  setUp(() async {
    root = Directory.systemTemp.createTempSync('dianoia_vl_store');
    server = _Server(Map.of(bundle));
    await server.start();
    store = ModelStore(root.path);
    spec = makeSpec();
  });
  tearDown(() async {
    await server.close();
    root.deleteSync(recursive: true);
  });

  VisionCheckpointSpec checkpointSpec({
    VisionQuantization quantization = VisionQuantization.q8,
  }) {
    server.files
      ..clear()
      ..addAll({
        for (final name in [
          'config.json',
          'processor_config.json',
          'tokenizer.json',
          'LICENSE',
          'model.safetensors',
        ])
          name: File('${fixture.path}/model/$name').readAsBytesSync(),
      });
    return VisionCheckpointSpec(
      id: 'tiny-checkpoint',
      quantization: quantization,
      name: 'Synthetic VL checkpoint',
      license: 'Synthetic test weights',
      licenseUrl: '${server.base}/LICENSE',
      files: [
        for (final e in server.files.entries)
          RemoteFile(
            e.key,
            '${server.base}/${e.key}',
            size: e.value.length,
            sha256: sha256.convert(e.value).toString(),
          ),
      ],
    );
  }

  for (final quantization in VisionQuantization.values) {
    test(
      'downloads a checkpoint, builds ${quantization.name} and loads image chat',
      () async {
        final checkpoint = checkpointSpec(quantization: quantization);
        server.cutOnce = 'model.safetensors';
        final progress = await store.install(checkpoint, retries: 1).toList();
        expect(progress.last.stage, InstallStage.done);
        expect(store.isInstalled(checkpoint), isTrue);
        expect(
          Directory('${root.path}/${checkpoint.id}/downloads').existsSync(),
          isFalse,
        );
        expect(
          Directory(
            '${store.bundlePath(checkpoint)}.partial.backbone',
          ).existsSync(),
          isFalse,
        );
        final manifest =
            jsonDecode(
                  File(
                    '${store.bundlePath(checkpoint)}/vl_bundle.json',
                  ).readAsStringSync(),
                )
                as Map;
        expect(manifest['weights'], quantization.name);
        expect(
          File(
            '${store.bundlePath(checkpoint)}/language/lfm3-${quantization.name}.cache',
          ).existsSync(),
          isTrue,
        );
        final other = quantization == VisionQuantization.q4 ? 'q8' : 'q4';
        expect(
          File(
            '${store.bundlePath(checkpoint)}/language/lfm3-$other.cache',
          ).existsSync(),
          isFalse,
        );
        final model = await LocalVisionModel.loadBundle(
          store.bundlePath(checkpoint),
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
        final count = server.requests.length;
        await store.install(checkpoint).drain<void>();
        expect(server.requests.length, count);
        final mirror = checkpoint.fromMirror('https://example.com/vl');
        expect(mirror.quantization, quantization);
        expect(
          mirror.files.last.url,
          'https://example.com/vl/model.safetensors',
        );
        expect(
          mirror.files.map((f) => f.sha256),
          checkpoint.files.map((f) => f.sha256),
        );
      },
    );
  }

  test(
    'cancel during checkpoint conversion keeps downloads unpublished',
    () async {
      final checkpoint = checkpointSpec();
      final stopped = Completer<void>();
      late StreamSubscription<InstallProgress> subscription;
      subscription = store.install(checkpoint).listen((p) {
        if (p.stage == InstallStage.building) {
          subscription.cancel().then(
            (_) => stopped.complete(),
            onError: stopped.completeError,
          );
        }
      }, onError: stopped.completeError);
      await stopped.future;
      expect(store.isInstalled(checkpoint), isFalse);
      expect(
        Directory('${root.path}/${checkpoint.id}/downloads').existsSync(),
        isTrue,
      );
      expect(
        Directory('${store.bundlePath(checkpoint)}.partial').existsSync(),
        isFalse,
      );
      expect(
        Directory(
          '${store.bundlePath(checkpoint)}.partial.backbone',
        ).existsSync(),
        isFalse,
      );
      await store.install(checkpoint).drain<void>();
      expect(store.isInstalled(checkpoint), isTrue);
    },
  );

  test('checkpoint conversion failure cleans temporary output', () async {
    final checkpoint = checkpointSpec();
    server.files['config.json'] = utf8.encode('{}');
    final invalid = VisionCheckpointSpec(
      id: checkpoint.id,
      name: checkpoint.name,
      license: checkpoint.license,
      licenseUrl: checkpoint.licenseUrl,
      files: [
        for (final e in server.files.entries)
          RemoteFile(
            e.key,
            '${server.base}/${e.key}',
            size: e.value.length,
            sha256: sha256.convert(e.value).toString(),
          ),
      ],
    );
    await expectLater(store.install(invalid).drain<void>(), throwsA(anything));
    expect(store.isInstalled(invalid), isFalse);
    expect(
      Directory('${store.bundlePath(invalid)}.partial').existsSync(),
      isFalse,
    );
    expect(
      Directory('${store.bundlePath(invalid)}.partial.backbone').existsSync(),
      isFalse,
    );
    // What a killed process's build leaves: its finally blocks never ran.
    final stale = Directory(
      '${store.directory}/${checkpoint.id}/.dianoia-vl-bundle-killed',
    )..createSync(recursive: true);
    File('${stale.path}/model.safetensors').writeAsStringSync('stale');
    await store.install(checkpointSpec()).drain<void>();
    expect(store.isInstalled(checkpoint), isTrue);
    expect(stale.existsSync(), isFalse);
  });

  test('upstream checkpoint catalog and path validation', () {
    final pinned = VisionCheckpointSpec.lfm3VlQ8;
    final q4 = VisionCheckpointSpec.lfm3VlQ4;
    expect(q4.quantization, VisionQuantization.q4);
    expect(q4.id, isNot(pinned.id));
    expect(q4.files, same(pinned.files));
    expect(q4.downloadBytes, pinned.downloadBytes);
    expect(pinned.downloadBytes, 3854341192);
    expect(
      pinned.files.every(
        (f) => f.url.contains('ba44bd51db20a2753f28c273a1571ccaa210d954'),
      ),
      isTrue,
    );
    expect(() => pinned.fromMirror('file:///tmp/vl'), throwsArgumentError);
    final unsafe = VisionCheckpointSpec(
      id: 'unsafe',
      name: 'unsafe',
      license: '',
      licenseUrl: '',
      files: [
        const RemoteFile(
          '../outside',
          'https://example.com/weights',
          size: 0,
          sha256: '',
        ),
      ],
    );
    expect(() => store.install(unsafe), throwsArgumentError);
  });

  test(
    'installs nested files atomically and loads through the public API',
    () async {
      expect(
        spec.downloadBytes,
        bundle.values.fold<int>(0, (n, b) => n + b.length),
      );
      expect(spec.files.map((f) => f.name).toSet(), bundle.keys.toSet());
      expect(() => spec.files.clear(), throwsUnsupportedError);
      final progress = await store.install(spec).toList();
      expect(progress.last.stage, InstallStage.done);
      expect(progress.last.bytes, spec.downloadBytes);
      expect(progress.last.fraction, 1);
      expect(progress.any((p) => p.stage == InstallStage.building), isTrue);
      expect(store.isInstalled(spec), isTrue);
      expect(
        Directory('${root.path}/${spec.id}/downloads').existsSync(),
        isFalse,
      );
      for (final entry in bundle.entries) {
        expect(
          File('${store.bundlePath(spec)}/${entry.key}').readAsBytesSync(),
          entry.value,
        );
      }
      final model = await LocalVisionModel.loadBundle(
        store.bundlePath(spec),
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
      final count = server.requests.length;
      expect(await store.install(spec).map((p) => p.stage).toList(), [
        InstallStage.done,
      ]);
      expect(server.requests.length, count);
      await store.remove(spec);
      expect(store.isInstalled(spec), isFalse);
    },
  );

  test(
    'cancel keeps downloads; resume uses nested Range and excludes another install',
    () async {
      final cache = 'language/lfm3-q8.cache';
      final part = File('${root.path}/${spec.id}/downloads/$cache.part');
      part.parent.createSync(recursive: true);
      part.writeAsBytesSync(bundle[cache]!.sublist(0, 100));
      final reached = Completer<void>();
      final sub = store.install(spec).listen((p) {
        if (p.bytes > 0 && !reached.isCompleted) reached.complete();
      }, onError: reached.completeError);
      await reached.future;
      expect(() => ModelStore(root.path).install(spec), throwsStateError);
      await expectLater(store.remove(spec), throwsStateError);
      await sub.cancel();
      expect(store.isInstalled(spec), isFalse);
      await store.install(spec).drain<void>();
      expect(store.isInstalled(spec), isTrue);
      expect(server.requests, contains((cache, 'bytes=100-')));
    },
  );

  test('cancel during validation leaves the bundle unpublished', () async {
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
    expect(Directory('${root.path}/${spec.id}/downloads').existsSync(), isTrue);
    await store.install(spec).drain<void>();
    expect(store.isInstalled(spec), isTrue);
  });

  test('retries a cut response and resumes instead of restarting', () async {
    server.cutOnce = 'language/lfm3-q8.cache';
    await store.install(spec, retries: 1).drain<void>();
    final requests = server.requests
        .where((r) => r.$1 == 'language/lfm3-q8.cache')
        .toList();
    expect(requests, hasLength(2));
    expect(requests.first.$2, isNull);
    expect(requests.last.$2, startsWith('bytes='));
    expect(requests.last.$2, isNot('bytes=0-'));
    expect(store.isInstalled(spec), isTrue);
  });

  test(
    'rejects changed remote payload and manifest without publishing',
    () async {
      for (final name in ['LICENSE', 'vl_bundle.json']) {
        final cached = File('${root.path}/${spec.id}/downloads/$name');
        if (cached.existsSync()) cached.deleteSync();
        server.files[name] = List.of(bundle[name]!)..[0] ^= 1;
        await expectLater(
          store.install(spec, retries: 0).drain<void>(),
          throwsA(isA<ModelDownloadException>()),
        );
        expect(store.isInstalled(spec), isFalse);
        expect(
          File('${root.path}/${spec.id}/downloads/$name.part').existsSync(),
          isFalse,
        );
        server.files[name] = bundle[name]!;
      }
      await store.install(spec).drain<void>();
      expect(store.isInstalled(spec), isTrue);
    },
  );

  test(
    'revalidates completed downloads after cancellation or a changed spec',
    () async {
      final downloads = '${root.path}/${spec.id}/downloads';
      for (final entry in bundle.entries) {
        final file = File('$downloads/${entry.key}');
        file.parent.createSync(recursive: true);
        file.writeAsBytesSync(entry.value);
      }
      final license = File('$downloads/LICENSE');
      license.writeAsBytesSync(List.of(bundle['LICENSE']!)..[0] ^= 1);
      await store.install(spec).drain<void>();
      expect(server.requests.map((r) => r.$1), ['LICENSE']);
      expect(store.isInstalled(spec), isTrue);
    },
  );

  test(
    'rejects internally inconsistent bundles even when payload hashes match',
    () async {
      final name = 'language/vl_backbone.json';
      final identity = jsonDecode(utf8.decode(bundle[name]!)) as Map;
      identity['source_key'] = 'wrong-source';
      final changed = utf8.encode(jsonEncode(identity));
      final manifest =
          jsonDecode(utf8.decode(bundle['vl_bundle.json']!)) as Map;
      (manifest['files'] as Map)[name] = {
        'bytes': changed.length,
        'sha256': sha256.convert(changed).toString(),
      };
      final json = jsonEncode(manifest);
      server.files[name] = changed;
      server.files['vl_bundle.json'] = utf8.encode(json);
      final inconsistent = makeSpec(json: json);
      await expectLater(
        store.install(inconsistent).drain<void>(),
        throwsArgumentError,
      );
      expect(store.isInstalled(inconsistent), isFalse);
    },
  );

  test(
    'a different pinned manifest cannot replace an installed bundle',
    () async {
      await store.install(spec).drain<void>();
      final different = makeSpec(
        json: '${utf8.decode(bundle['vl_bundle.json']!)}\n',
      );
      expect(store.isInstalled(different), isFalse);
      await expectLater(
        store.install(different).drain<void>(),
        throwsStateError,
      );
      expect(store.isInstalled(spec), isTrue);
    },
  );

  test('rejects unsafe model IDs, file lists and unsupported manifests', () {
    final badId = makeSpec(id: '../outside');
    expect(() => store.install(badId), throwsArgumentError);
    expect(() => store.bundlePath(badId), throwsArgumentError);
    for (final change in <void Function(Map)>[
      (m) => m['version'] = 2,
      (m) => (m['files'] as Map).remove('LICENSE'),
      (m) =>
          (m['files'] as Map)['../outside'] = {'bytes': 1, 'sha256': '0' * 64},
      (m) => ((m['files'] as Map)['LICENSE'] as Map)['sha256'] = 'invalid',
    ]) {
      final manifest =
          jsonDecode(utf8.decode(bundle['vl_bundle.json']!)) as Map;
      change(manifest);
      expect(() => makeSpec(json: jsonEncode(manifest)), throwsFormatException);
    }
    expect(() => spec.fromMirror('file:///tmp/models'), throwsArgumentError);
    final mirror = spec.fromMirror('https://example.com/pinned');
    expect(
      mirror.files.last.url,
      'https://example.com/pinned/language/vl_backbone.json',
    );
    expect(mirror.downloadBytes, spec.downloadBytes);
    expect(mirror.files.map((f) => f.sha256), spec.files.map((f) => f.sha256));
  });
}
