import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia_example/src/audio_worker.dart';
import 'package:dianoia_example/src/live_translation.dart';
import 'package:dianoia_example/src/speech_model.dart';
import 'package:dianoia_example/src/translation_worker.dart';
import 'package:flutter_test/flutter_test.dart';

/// LiveASR on the audio worker and translation on its own CPU worker, fed a
/// multi-sentence English clip at the microphone's pace.
void main() {
  final pcmPath = Platform.environment['DIANOIA_TRANSLATION_EXAMPLE_PCM'];
  for (final model in SpeechModel.values) {
    final prefix = model == SpeechModel.audio
        ? 'DIANOIA_AUDIO_EXAMPLE'
        : 'DIANOIA_OMNI_EXAMPLE';
    final directory = Platform.environment['${prefix}_DIR'];
    final gpu = Platform.environment['${prefix}_GPU'] == '1';
    test(
      '${model.name} translates live transcript sentences while audio streams',
      () async {
        final audio = AudioIsolateBackend(model: model, useGpu: gpu);
        final translator = LiveTranslator(TranslationIsolateBackend());
        try {
          expect(await audio.load(directory!), gpu ? 'GPU' : 'CPU');
          await translator.prepare('$directory/language');
          final pcm = Float32List.sublistView(File(pcmPath!).readAsBytesSync());
          var transcript = '';
          final clock = Stopwatch()..start();
          final firstTranslation = Completer<double>();
          translator.addListener(() {
            if (!firstTranslation.isCompleted &&
                translator.segments.any((s) => s.done)) {
              firstTranslation.complete(clock.elapsedMilliseconds / 1000);
            }
          });
          await audio.begin(
            'English',
            live: true,
            onLiveText: (text) {
              transcript = text;
              translator.update(text);
            },
          );
          const chunk = 1280; // 80 ms, as the app sends it.
          for (var at = 0; at < pcm.length; at += chunk) {
            final end = at + chunk < pcm.length ? at + chunk : pcm.length;
            await audio.addPcm(Float32List.fromList(pcm.sublist(at, end)));
            // Real time: never ahead of the clip.
            final due = Duration(microseconds: end * 1000000 ~/ 16000);
            if (clock.elapsed < due) {
              await Future<void>.delayed(due - clock.elapsed);
            }
          }
          final audioSeconds = pcm.length / 16000;
          final lagAtEnd = clock.elapsedMilliseconds / 1000 - audioSeconds;
          transcript = await audio.finish();
          translator.update(transcript, flush: true);
          while (translator.busy) {
            await Future<void>.delayed(const Duration(milliseconds: 20));
          }
          final total = clock.elapsedMilliseconds / 1000;
          expect(firstTranslation.isCompleted, isTrue);
          expect(transcript, isNotEmpty);
          final first = await firstTranslation.future;
          for (final s in translator.segments) {
            stdout.writeln('${s.source}\n  -> ${s.translation}');
          }
          stdout.writeln(
            'audio ${audioSeconds.toStringAsFixed(1)} s; ASR lag at end '
            '${lagAtEnd.toStringAsFixed(2)} s; first translation at '
            '${first.toStringAsFixed(1)} s; '
            'all translated at ${total.toStringAsFixed(1)} s',
          );
          expect(translator.segments.length, greaterThanOrEqualTo(3));
          expect(
            translator.segments.every(
              (s) => s.done && s.translation.isNotEmpty,
            ),
            isTrue,
          );
          expect(translator.error, isNull);
          expect(first, lessThan(audioSeconds));
        } finally {
          await translator.close();
          await audio.close();
        }
      },
      skip: directory == null || pcmPath == null
          ? 'set ${prefix}_DIR and DIANOIA_TRANSLATION_EXAMPLE_PCM'
          : false,
      timeout: const Timeout(Duration(minutes: 3)),
    );
  }
}
