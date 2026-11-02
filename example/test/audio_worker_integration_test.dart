import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia_example/src/audio_worker.dart';
import 'package:dianoia_example/src/model_worker_gate.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test(
    'failed model loads still close their workers and release the next lease',
    () async {
      final missing = Directory.systemTemp.createTempSync(
        'dianoia-audio-missing-',
      );
      try {
        for (var i = 0; i < 2; i++) {
          final backend = AudioIsolateBackend(useGpu: false);
          await expectLater(backend.load(missing.path), throwsStateError);
          await backend.close();
        }
        final next = ModelWorkerLease.reserve();
        await next.ready.timeout(const Duration(seconds: 3));
        next.release();
      } finally {
        missing.deleteSync();
      }
    },
  );

  test(
    'closing a queued audio worker cannot bypass the active model worker',
    () async {
      final active = ModelWorkerLease.reserve();
      await active.ready;
      final backend = AudioIsolateBackend(useGpu: false);
      final loading = expectLater(backend.load('/unused'), throwsStateError);
      final closing = backend.close();
      final next = ModelWorkerLease.reserve();
      var granted = false;
      next.ready.then((_) => granted = true);
      await Future<void>.delayed(Duration.zero);
      expect(granted, isFalse);
      active.release();
      await loading;
      await closing;
      await next.ready.timeout(const Duration(seconds: 3));
      next.release();
    },
  );
  final directory = Platform.environment['DIANOIA_AUDIO_EXAMPLE_DIR'];
  final pcmPath = Platform.environment['DIANOIA_AUDIO_EXAMPLE_PCM'];
  final gpu = Platform.environment['DIANOIA_AUDIO_EXAMPLE_GPU'] == '1';
  test(
    'prepared Q4 audio worker streams PCM, transcribes, cancels and exits',
    () async {
      final backend = AudioIsolateBackend(useGpu: gpu);
      try {
        expect(await backend.load(directory!), gpu ? 'GPU' : 'CPU');
        await backend.begin('English');
        final data = ByteData.sublistView(File(pcmPath!).readAsBytesSync());
        final samples = Float32List(data.lengthInBytes ~/ 4);
        for (var i = 0; i < samples.length; i++) {
          samples[i] = data.getFloat32(i * 4, Endian.little);
        }
        for (var at = 0; at < samples.length; at += 1280) {
          await backend.addPcm(
            samples.sublist(at, (at + 1280).clamp(0, samples.length)),
          );
        }
        expect(
          await backend.finish(),
          'The quick brown fox jumps over the lazy dog.',
        );
        await backend.begin('English');
        await backend.addPcm(samples.sublist(0, 1280));
        await backend.cancel();
        await expectLater(backend.finish(), throwsStateError);
        await backend.begin('English');
        await backend.addPcm(samples);
        expect(
          await backend.finish(),
          'The quick brown fox jumps over the lazy dog.',
        );
      } finally {
        await backend.close();
      }
      await backend.close();
    },
    skip: directory == null || pcmPath == null
        ? 'set DIANOIA_AUDIO_EXAMPLE_DIR and DIANOIA_AUDIO_EXAMPLE_PCM'
        : false,
  );
}
