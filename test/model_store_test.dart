// ModelStore.install end to end, with the published files served from a
// local directory instead of Hugging Face. Needs dart-llm-lab's copies of
// them: DIANOIA_LAB_MODELS=path/to/dart-llm-lab/models; skipped otherwise.
import 'dart:async';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:dianoia/dianoia.dart';
import 'package:test/test.dart';

final _lab = Platform.environment['DIANOIA_LAB_MODELS'];

/// Serves files by name, honouring Range requests as Hugging Face's CDN
/// does.
Future<HttpServer> _serve(Map<String, String> files) async {
  final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
  server.listen((request) async {
    final response = request.response;
    final path = files[request.uri.pathSegments.last];
    if (path == null) {
      response.statusCode = HttpStatus.notFound;
      await response.close();
      return;
    }
    final length = File(path).lengthSync();
    final range = request.headers.value(HttpHeaders.rangeHeader);
    final from = range == null
        ? 0
        : int.parse(RegExp(r'bytes=(\d+)-').firstMatch(range)![1]!);
    if (range != null) {
      response
        ..statusCode = HttpStatus.partialContent
        ..headers.set(
          HttpHeaders.contentRangeHeader,
          'bytes $from-${length - 1}/$length',
        );
    }
    response.contentLength = length - from;
    try {
      await response.addStream(File(path).openRead(from));
      await response.close();
    } on IOException {
      // The client went away (a cancelled install).
    }
  });
  return server;
}

/// [ModelSpec.lfm3Q4] with its files at [base].
ModelSpec _localSpec(String base) {
  RemoteFile local(RemoteFile f) =>
      RemoteFile(f.name, '$base/${f.name}', size: f.size, sha256: f.sha256);
  const spec = ModelSpec.lfm3Q4;
  return ModelSpec(
    id: spec.id,
    name: spec.name,
    license: spec.license,
    licenseUrl: spec.licenseUrl,
    gguf: local(spec.gguf),
    config: local(spec.config),
    tokenizer: local(spec.tokenizer),
    licenseFile: local(spec.licenseFile),
  );
}

String _sha256(String path) =>
    sha256.convert(File(path).readAsBytesSync()).toString();

void main() {
  test('the LFM3 spec downloads 1.07 GB', () {
    expect(ModelSpec.lfm3Q4.downloadBytes, 1067536343);
    expect(ModelSpec.lfm3Q4.files.map((f) => f.name).toSet(), hasLength(4));
  });

  group(
    'install',
    () {
      late HttpServer server;
      late Directory dir;
      late ModelSpec spec;

      setUpAll(() async {
        server = await _serve({
          'LFM3-1.7B-QAD-Q4_0.gguf':
              '$_lab/LFM3-1.7B-QAD-GGUF/LFM3-1.7B-QAD-Q4_0.gguf',
          for (final name in ['config.json', 'tokenizer.json', 'LICENSE'])
            name: '$_lab/LFM3-1.7B/$name',
        });
        spec = _localSpec('http://127.0.0.1:${server.port}/files');
        dir = Directory.systemTemp.createTempSync('dianoia_store');
      });
      tearDownAll(() async {
        await server.close(force: true);
        dir.deleteSync(recursive: true);
      });

      test('a cancelled install resumes, then builds the lab bundle', () async {
        final store = ModelStore(dir.path);
        // Cancel three times, each resuming the last and going further: a
        // cancel must leave no write behind for the next install to race.
        for (final mb in [150, 300, 450]) {
          final reached = Completer<void>();
          final sub = store.install(spec).listen((progress) {
            if (progress.bytes > mb << 20 && !reached.isCompleted) {
              reached.complete();
            }
          });
          await reached.future;
          expect(() => store.install(spec), throwsStateError);
          await sub.cancel();
        }
        expect(store.isInstalled(spec), isFalse);

        // The GGUF comes last; its part must be picked up, not started over.
        final part = File(
          '${dir.path}/${spec.id}/downloads/${spec.gguf.name}.part',
        ).lengthSync();
        expect(part, greaterThan(400 << 20));
        final small = spec.downloadBytes - spec.gguf.size;
        final stages = <InstallStage>[];
        final ggufBytes = <int>[];
        await for (final progress in store.install(spec)) {
          stages.add(progress.stage);
          if (progress.stage == InstallStage.downloading &&
              progress.bytes > small) {
            ggufBytes.add(progress.bytes - small);
          }
        }
        expect(ggufBytes, isNotEmpty);
        expect(ggufBytes.first, greaterThanOrEqualTo(part));
        expect(stages.toSet(), {
          InstallStage.downloading,
          InstallStage.building,
          InstallStage.done,
        });
        expect(stages.last, InstallStage.done);
        expect(store.isInstalled(spec), isTrue);
        expect(
          Directory('${dir.path}/${spec.id}/downloads').existsSync(),
          false,
        );

        final bundle = store.bundlePath(spec);
        for (final name in [
          'lfm3-q4.cache',
          'model.safetensors',
          'config.json',
          'tokenizer.json',
        ]) {
          expect(
            _sha256('$bundle/$name'),
            _sha256('$_lab/LFM3-1.7B-QAD-Q4/$name'),
            reason: name,
          );
        }
        expect(File('$bundle/LICENSE').existsSync(), isTrue);
      });

      test('an installed model loads and answers', () async {
        final store = ModelStore(dir.path);
        expect(await store.install(spec).map((p) => p.stage).toList(), [
          InstallStage.done,
        ]);
        final model = await LocalModel.load(store.bundlePath(spec));
        addTearDown(model.close);
        final turn = await model.chat(maxNewTokens: 8).send({
          'role': 'user',
          'content': 'Say hi.',
        });
        expect(turn.generatedTokens, greaterThan(0));
      });
    },
    skip: _lab == null
        ? 'set DIANOIA_LAB_MODELS to dart-llm-lab/models'
        : false,
  );
}
