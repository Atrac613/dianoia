// Installs an actual pack-vl bundle over loopback HTTP, then uses the public
// vision API. Optional local assets only; never fetches external weights.
import 'dart:io';

import 'package:dianoia/dianoia.dart';
import 'package:test/test.dart';

void main() {
  final source = Platform.environment['DIANOIA_VL_BUNDLE'];
  final validation = Platform.environment['DIANOIA_VL_VALIDATION_DIR'];
  test(
    'installs the real VL bundle and reads Japanese OCR on CPU and Metal',
    () async {
      final root = Directory.systemTemp.createTempSync('dianoia_real_vl_store');
      addTearDown(() => root.deleteSync(recursive: true));
      final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      addTearDown(() => server.close(force: true));
      final requests = <String>[];
      server.listen((request) async {
        final name = request.uri.path.substring(1);
        requests.add(name);
        final file = File('$source/$name');
        final response = request.response;
        if (!file.existsSync()) {
          response.statusCode = HttpStatus.notFound;
          await response.close();
          return;
        }
        response.contentLength = file.lengthSync();
        try {
          await response.addStream(file.openRead());
          await response.close();
        } on IOException {
          // The install may have been cancelled by a failed test.
        }
      });
      final spec = VisionModelSpec.fromManifest(
        id: 'lfm3-vl-2b-q8-local-test',
        name: 'LFM3-VL-2B (Q8)',
        license: 'LFM Open License v1.0',
        licenseUrl: 'http://127.0.0.1:${server.port}/LICENSE',
        baseUrl: 'http://127.0.0.1:${server.port}',
        manifestJson: File('$source/vl_bundle.json').readAsStringSync(),
      );
      final store = ModelStore(root.path);
      final progress = await store.install(spec).toList();
      expect(progress.last.stage, InstallStage.done);
      expect(progress.last.bytes, spec.downloadBytes);
      expect(requests.toSet(), spec.files.map((f) => f.name).toSet());
      expect(store.isInstalled(spec), isTrue);
      expect(
        Directory('${root.path}/${spec.id}/downloads').existsSync(),
        isFalse,
      );
      for (final backend in [InferenceBackend.cpu, InferenceBackend.gpu]) {
        final model = await LocalVisionModel.loadBundle(
          store.bundlePath(spec),
          backend: backend,
          contextLength: 512,
        );
        try {
          expect(model.backend, backend);
          final turn = await model
              .chat(maxNewTokens: 32)
              .send(
                '画像に書かれた文字を、そのまま書き起こしてください。',
                images: [
                  ChatImage.rgb(
                    File(
                      '$validation/reference-ocr-ja/input.rgb',
                    ).readAsBytesSync(),
                    width: 384,
                    height: 192,
                  ),
                ],
              );
          expect(turn.text, '東京 123');
        } finally {
          await model.close();
        }
      }
    },
    timeout: const Timeout(Duration(minutes: 10)),
    skip: source == null || validation == null
        ? 'set DIANOIA_VL_BUNDLE and DIANOIA_VL_VALIDATION_DIR'
        : false,
  );
}
