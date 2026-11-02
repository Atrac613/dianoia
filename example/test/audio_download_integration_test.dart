@Timeout(Duration(minutes: 5))
library;

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:dianoia_example/src/audio_worker.dart';
import 'package:dianoia_example/src/model_worker_gate.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test(
    'cancel followed by close drains a blocked download before releasing the worker',
    () async {
      final root = Directory.systemTemp.createTempSync('dianoia-audio-cancel-');
      final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      final requested = Completer<void>();
      server.listen((r) async {
        r.response.contentLength = Lfm3AudioQ4Source.config.size;
        r.response.add([32]);
        try {
          await r.response.flush();
        } on IOException {
          return;
        }
        if (!requested.isCompleted) requested.complete();
        // Leave the remaining body pending; only cancel/close can end the install.
      });
      final backend = AudioIsolateBackend(
        useGpu: false,
        mirror: 'http://127.0.0.1:${server.port}',
      );
      try {
        expect(await backend.installed(root.path), isNull);
        final installing = expectLater(
          backend.install(root.path, (_) {}),
          throwsStateError,
        );
        await requested.future.timeout(const Duration(seconds: 10));
        final cancelling = backend.cancel();
        final closing = backend.close();
        await installing.timeout(const Duration(seconds: 10));
        await cancelling.timeout(const Duration(seconds: 10));
        await closing.timeout(const Duration(seconds: 10));
        final next = ModelWorkerLease.reserve();
        await next.ready.timeout(const Duration(seconds: 3));
        next.release();
        expect(
          ModelStore(root.path).isInstalled(AudioGgufSpec.lfm3AudioQ4),
          isFalse,
        );
      } finally {
        await backend.close();
        await server.close(force: true);
        root.deleteSync(recursive: true);
      }
    },
  );

  final source = Platform.environment['DIANOIA_AUDIO_DOWNLOAD_SOURCE'];
  test(
    'official audio files install through HTTP, reload and synthesize in the worker',
    () async {
      final root = Directory.systemTemp.createTempSync(
        'dianoia-audio-real-install-',
      );
      final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      final requests = <(String, String?)>[];
      server.listen((r) async {
        final name = r.uri.pathSegments.last;
        final file = File('$source/$name');
        if (!file.existsSync()) {
          r.response.statusCode = 404;
          await r.response.close();
          return;
        }
        final range = r.headers.value(HttpHeaders.rangeHeader);
        requests.add((name, range));
        final from = range == null
            ? 0
            : int.parse(RegExp(r'bytes=(\d+)-').firstMatch(range)![1]!);
        if (range != null) {
          r.response.statusCode = 206;
          r.response.headers.set(
            HttpHeaders.contentRangeHeader,
            'bytes $from-${file.lengthSync() - 1}/${file.lengthSync()}',
          );
        }
        r.response.contentLength = file.lengthSync() - from;
        try {
          await r.response.addStream(file.openRead(from));
          await r.response.close();
        } on IOException {
          /* Cancellation closes the connection. */
        }
      });
      final backend = AudioIsolateBackend(
        useGpu: false,
        mirror: 'http://127.0.0.1:${server.port}',
      );
      try {
        const spec = AudioGgufSpec.lfm3AudioQ4;
        final downloads = Directory('${root.path}/${spec.id}/downloads')
          ..createSync(recursive: true);
        final part = File('${downloads.path}/${spec.language.name}.part');
        final handle = File('$source/${spec.language.name}').openSync();
        try {
          part.writeAsBytesSync(handle.readSync(1 << 20));
        } finally {
          handle.closeSync();
        }
        final stages = <InstallStage>{};
        final watch = Stopwatch()..start();
        final bundle = await backend.install(
          root.path,
          (p) => stages.add(p.stage),
        );
        expect(stages, containsAll(InstallStage.values));
        expect(requests, hasLength(5));
        expect(
          requests.any(
            (r) => r.$1 == spec.language.name && r.$2 == 'bytes=1048576-',
          ),
          isTrue,
        );
        expect(await backend.installed(root.path), bundle);
        expect(downloads.existsSync(), isFalse);
        expect(await backend.load(bundle), 'CPU');
        final wav = await backend.synthesize('Hello world.', 'en', 'female');
        expect(String.fromCharCodes(wav.sublist(0, 4)), 'RIFF');
        expect(ByteData.sublistView(wav).getUint32(24, Endian.little), 24000);
        expect(wav.length, greaterThan(44));
        stdout.writeln(
          'Audio install: ${spec.downloadBytes} bytes, ${requests.length} requests, '
          'Range resumed; ${(wav.length - 44) / 2 / 24000} s WAV; ${watch.elapsedMilliseconds} ms total',
        );
      } finally {
        await backend.close();
        await server.close(force: true);
        root.deleteSync(recursive: true);
      }
    },
    skip: source == null
        ? 'set DIANOIA_AUDIO_DOWNLOAD_SOURCE to the pinned source directory'
        : false,
  );
}
