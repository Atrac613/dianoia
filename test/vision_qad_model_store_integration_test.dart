// Optional real QAD download/import/pack and public-API arithmetic validation.
@Timeout(Duration(minutes: 10))
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:dianoia/dianoia.dart';
import 'package:test/test.dart';

void main() {
  final source = Platform.environment['DIANOIA_VL_QAD_DIR'];
  final imagePath = Platform.environment['DIANOIA_VL_QAD_IMAGE'];
  test(
    'official QAD installs through HTTP and preserves validated CPU/Metal logits',
    () async {
      final keep = Platform.environment['DIANOIA_VL_QAD_KEEP_DIR'];
      final root = keep == null
          ? Directory.systemTemp.createTempSync('dianoia-real-qad-')
          : (Directory(keep)..createSync(recursive: true));
      addTearDown(() {
        if (keep == null) root.deleteSync(recursive: true);
      });
      final catalog = VisionGgufSpec.lfm3VlQadQ4;
      final requests = <String>[];
      final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      addTearDown(() => server.close(force: true));
      final files = {
        for (final f in catalog.files)
          f.name: File(
            f.name.endsWith('.gguf')
                ? '$source/${f.name}'
                : '$source/source-config/${f.name}',
          ),
      };
      server.listen((r) async {
        final f = files[r.uri.pathSegments.last];
        if (f == null) {
          r.response.statusCode = 404;
          await r.response.close();
          return;
        }
        requests.add(f.path);
        final range = r.headers.value(HttpHeaders.rangeHeader);
        final from = range == null
            ? 0
            : int.parse(RegExp(r'bytes=(\d+)-').firstMatch(range)![1]!);
        if (range != null) {
          r.response.statusCode = 206;
          r.response.headers.set(
            HttpHeaders.contentRangeHeader,
            'bytes $from-${f.lengthSync() - 1}/${f.lengthSync()}',
          );
        }
        r.response.contentLength = f.lengthSync() - from;
        try {
          await r.response.addStream(f.openRead(from));
          await r.response.close();
        } on IOException {
          /* A cancelled client may close the socket. */
        }
      });
      final spec = catalog.fromMirror('http://127.0.0.1:${server.port}');
      final store = ModelStore('${root.path}/store');
      final progress = await store.install(spec).toList();
      expect(progress.last.stage, InstallStage.done);
      expect(store.isInstalled(spec), isTrue);
      expect(
        Directory('${root.path}/store/${spec.id}/downloads').existsSync(),
        isFalse,
      );
      final bundle = store.bundlePath(spec);
      final m =
          jsonDecode(File('$bundle/vl_bundle.json').readAsStringSync()) as Map;
      expect(m['input_embedding'], 'q8-cache');
      expect(m['weights'], 'q4');
      final records = <String, Object?>{};
      final image = await ChatImage.fromFile(imagePath!);
      String hash(Float32List v) => sha256
          .convert(v.buffer.asUint8List(v.offsetInBytes, v.lengthInBytes))
          .toString();
      for (final backend in [InferenceBackend.cpu, InferenceBackend.gpu]) {
        final original = await LocalVisionModel.load(
          '$source/vision',
          backboneDir: '$source/backbone',
          backend: backend,
          contextLength: 512,
        );
        final packed = await LocalVisionModel.loadBundle(
          bundle,
          backend: backend,
          contextLength: 512,
        );
        try {
          final a = <String>[], b = <String>[];
          const prompt = '画像に書かれた文字を、そのまま書き起こしてください。';
          final x = await original
              .chat(maxNewTokens: 32)
              .send(prompt, images: [image], onLogits: (v) => a.add(hash(v)));
          final y = await packed
              .chat(maxNewTokens: 32)
              .send(prompt, images: [image], onLogits: (v) => b.add(hash(v)));
          expect(x.text, '東京 123');
          expect(y.text, x.text);
          expect(y.generatedTokens, x.generatedTokens);
          expect(a, hasLength(x.generatedTokens + 1));
          expect(b, orderedEquals(a));
          records[backend.name] = {
            'text': y.text,
            'source_logit_sha256': a,
            'installed_logit_sha256': b,
            'byte_identical': true,
          };
        } finally {
          await original.close();
          await packed.close();
        }
      }
      File('${root.path}/integration.json').writeAsStringSync(
        '${const JsonEncoder.withIndent('  ').convert({'bundle': bundle, 'download_requests': requests.length, 'download_bytes': spec.downloadBytes, 'comparisons': records})}\n',
      );
      stdout.writeln('Installed QAD bundle: $bundle');
    },
    skip: source == null || imagePath == null
        ? 'set DIANOIA_VL_QAD_DIR and DIANOIA_VL_QAD_IMAGE'
        : false,
  );
}
