import 'dart:async';
import 'dart:typed_data';

import 'package:dianoia_example/src/audio_backend.dart';
import 'package:dianoia_example/src/audio_controller.dart';
import 'package:dianoia_example/src/microphone.dart';
import 'package:dianoia_example/src/pcm16_stream.dart';
import 'package:dianoia_example/src/speech_player.dart';
import 'package:flutter_test/flutter_test.dart';

class TestMicrophone implements Microphone {
  bool allowed = true;
  Completer<bool>? permission;
  Completer<Stream<Uint8List>>? starting;
  StreamController<Uint8List>? source;
  int starts = 0, stops = 0, disposals = 0;
  @override
  Future<bool> requestPermission() async =>
      permission == null ? allowed : await permission!.future;
  @override
  Future<Stream<Uint8List>> start() async {
    starts++;
    source = StreamController<Uint8List>();
    return starting == null ? source!.stream : await starting!.future;
  }

  @override
  Future<void> stop() async {
    stops++;
    if (source case final s? when !s.isClosed) unawaited(s.close());
  }

  @override
  Future<void> dispose() async {
    disposals++;
    await stop();
  }
}

class TestAudioBackend implements AudioBackend {
  final samples = <double>[];
  Completer<void>? ack;
  Completer<String>? finishing;
  Completer<Uint8List>? speech;
  final utterances = <(String, String, String)>[];
  int begins = 0, finishes = 0, cancels = 0, closes = 0;
  @override
  Future<String> load(String directory) async => 'CPU';
  @override
  Future<void> begin(String? language) async {
    begins++;
  }

  @override
  Future<void> addPcm(Float32List pcm) async {
    samples.addAll(pcm);
    await ack?.future;
  }

  @override
  Future<String> finish() async {
    finishes++;
    return finishing == null ? 'Recognized speech.' : await finishing!.future;
  }

  @override
  Future<Uint8List> synthesize(
    String text,
    String language,
    String voice,
  ) async {
    utterances.add((text, language, voice));
    return speech == null
        ? Uint8List.fromList([1, 2, 3])
        : await speech!.future;
  }

  @override
  Future<void> cancel() async {
    cancels++;
    if (ack case final c? when !c.isCompleted) {
      c.completeError(StateError('cancelled'));
    }
    if (finishing case final c? when !c.isCompleted) {
      c.completeError(StateError('cancelled'));
    }
  }

  @override
  Future<void> close() async {
    closes++;
  }
}

class TestSpeechPlayer implements SpeechPlayer {
  final played = <Uint8List>[];
  Completer<void>? completing;
  Object? error;
  int stops = 0, disposals = 0;
  @override
  Future<void> play(Uint8List wav) async {
    played.add(wav);
    if (error != null) throw error!;
    await completing?.future;
  }

  @override
  Future<void> stop() async {
    stops++;
    if (completing case final c? when !c.isCompleted) c.complete();
  }

  @override
  Future<void> dispose() async {
    disposals++;
    await stop();
  }
}

Future<void> events() => Future<void>.delayed(Duration.zero);

void main() {
  late AudioController audio;
  late TestMicrophone mic;
  late TestAudioBackend backend;
  late TestSpeechPlayer player;
  setUp(() async {
    mic = TestMicrophone();
    backend = TestAudioBackend();
    player = TestSpeechPlayer();
    audio = AudioController(backend: backend, microphone: mic, player: player);
    await audio.load('/prepared');
  });
  tearDown(() async {
    await audio.close();
    audio.dispose();
  });

  test(
    'speech is explicit, keeps options and excludes capture until playback ends',
    () async {
      expect(backend.utterances, isEmpty);
      player.completing = Completer<void>();
      final speaking = audio.speak('Hello.', language: 'en', voice: 'male');
      await events();
      expect(backend.utterances, [('Hello.', 'en', 'male')]);
      expect(audio.phase, AudioPhase.playing);
      await audio.record();
      await audio.speak('Ignored');
      expect(mic.starts, 0);
      expect(backend.utterances, hasLength(1));
      expect(player.played.single, [1, 2, 3]);
      player.completing!.complete();
      await speaking;
      expect(audio.phase, AudioPhase.ready);
      await audio.record();
      expect(mic.starts, 1);
      await audio.cancel();
    },
  );

  test(
    'cancel during generation never plays a late result and permits reuse',
    () async {
      backend.speech = Completer<Uint8List>();
      final speaking = audio.speak('Hello');
      expect(audio.phase, AudioPhase.synthesizing);
      await audio.cancel();
      backend.speech!.complete(Uint8List(44));
      await speaking;
      expect(player.played, isEmpty);
      expect(audio.phase, AudioPhase.ready);
      backend.speech = null;
      await audio.speak('Next');
      expect(player.played, hasLength(1));
    },
  );

  test(
    'cancelling from the playback transition cannot start a late player',
    () async {
      void cancelOnPlayback() {
        if (audio.phase == AudioPhase.playing) unawaited(audio.cancel());
      }

      audio.addListener(cancelOnPlayback);
      await audio.speak('Hello');
      await events();
      audio.removeListener(cancelOnPlayback);
      expect(player.played, isEmpty);
      expect(audio.phase, AudioPhase.ready);
    },
  );

  test(
    'speech stop and route close stop playback and release the player',
    () async {
      player.completing = Completer<void>();
      final speaking = audio.speak('Hello');
      await events();
      await audio.cancel();
      await speaking;
      expect(player.stops, 1);
      expect(audio.phase, AudioPhase.ready);
      player.completing = Completer<void>();
      final second = audio.speak('Next');
      await events();
      await audio.close();
      await second;
      expect(audio.phase, AudioPhase.closed);
      expect(player.disposals, 1);
      expect(backend.closes, 1);
    },
  );

  test(
    'blank text and playback failures leave a reusable ready state',
    () async {
      await audio.speak('  ');
      expect(backend.utterances, isEmpty);
      expect(audio.error, contains('Enter text'));
      player.error = StateError('Device unavailable');
      await audio.speak('Hello');
      expect(audio.error, contains('Device unavailable'));
      expect(audio.phase, AudioPhase.ready);
      player.error = null;
      await audio.speak('Retry');
      expect(audio.error, isNull);
      expect(audio.phase, AudioPhase.ready);
    },
  );

  test(
    'PCM16 packetizer preserves signed samples across odd byte boundaries',
    () {
      final packetizer = Pcm16Packetizer();
      expect(packetizer.add(Uint8List.fromList([0, 128, 255])), isEmpty);
      expect(
        packetizer.add(Uint8List.fromList([127, 0, 0, 255, 255])),
        isEmpty,
      );
      expect(packetizer.finish(), [-1, 32767 / 32768, 0, -1 / 32768]);
      expect(packetizer.finish, throwsStateError);
      expect(() => packetizer.add(Uint8List(0)), throwsStateError);
      final invalid = Pcm16Packetizer()..add(Uint8List.fromList([1]));
      expect(invalid.finish, throwsFormatException);
    },
  );

  test(
    'recording starts only after permission and flushes the final partial chunk',
    () async {
      expect(mic.starts, 0);
      expect(backend.begins, 0);
      await audio.record();
      final bytes = Uint8List(1281 * 2);
      final data = ByteData.sublistView(bytes);
      for (var i = 0; i < 1281; i++) {
        data.setInt16(i * 2, i - 640, Endian.little);
      }
      mic.source!.add(bytes.sublist(0, 7));
      mic.source!.add(bytes.sublist(7));
      await events();
      await audio.stop();
      expect(backend.samples, [
        for (var i = 0; i < 1281; i++) (i - 640) / 32768,
      ]);
      expect(backend.finishes, 1);
      expect(audio.transcript, 'Recognized speech.');
      expect(audio.phase, AudioPhase.ready);
      expect(audio.queuedSamples, 0);
    },
  );

  test('permission denial does not start capture or inference', () async {
    mic.allowed = false;
    await audio.record();
    expect(mic.starts, 0);
    expect(backend.begins, 0);
    expect(audio.phase, AudioPhase.ready);
    expect(audio.error, contains('denied'));
  });

  test(
    'cancel during a permission prompt prevents capture after it resolves',
    () async {
      mic.permission = Completer<bool>();
      final pending = audio.record();
      await events();
      await audio.cancel();
      mic.permission!.complete(true);
      await pending;
      expect(mic.starts, 0);
      expect(backend.begins, 0);
      expect(audio.transcript, isEmpty);
    },
  );

  test(
    'stop drains pending worker ACKs before asking for a transcript',
    () async {
      backend.ack = Completer<void>();
      await audio.record();
      mic.source!.add(Uint8List(2560));
      await events();
      final stop = audio.stop();
      await events();
      expect(backend.finishes, 0);
      expect(audio.phase, AudioPhase.finishing);
      backend.ack!.complete();
      await stop;
      expect(backend.finishes, 1);
      expect(audio.phase, AudioPhase.ready);
    },
  );

  test(
    'backpressure fails closed without silent loss or an unbounded queue',
    () async {
      backend.ack = Completer<void>();
      await audio.record();
      for (var i = 0; i < 13; i++) {
        mic.source!.add(Uint8List(2560));
      }
      await events();
      await events();
      await events();
      expect(
        audio.queuedSamples,
        lessThanOrEqualTo(AudioController.maxQueuedSamples),
      );
      expect(audio.error, contains('fell behind'));
      expect(mic.stops, greaterThan(0));
      expect(backend.finishes, 0);
      expect(audio.transcript, isEmpty);
    },
  );

  test(
    'cancel discards an in-flight transcription and the next recording works',
    () async {
      backend.finishing = Completer<String>();
      await audio.record();
      mic.source!.add(Uint8List(2560));
      await events();
      final stop = audio.stop();
      await events();
      await events();
      expect(backend.finishes, 1);
      await audio.cancel();
      await stop;
      expect(audio.transcript, isEmpty);
      expect(audio.phase, AudioPhase.ready);
      backend.finishing = null;
      await audio.record();
      mic.source!.add(Uint8List(32));
      await events();
      await audio.stop();
      expect(audio.transcript, 'Recognized speech.');
    },
  );

  test(
    'close releases the worker while microphone start is still pending',
    () async {
      mic.starting = Completer<Stream<Uint8List>>();
      final starting = audio.record();
      await events();
      await audio.close();
      expect(backend.closes, 1);
      mic.starting!.complete(mic.source!.stream);
      await starting;
      expect(audio.phase, AudioPhase.closed);
      expect(audio.transcript, isEmpty);
    },
  );

  test(
    'an incomplete final PCM sample is rejected, not silently truncated',
    () async {
      await audio.record();
      mic.source!.add(Uint8List.fromList([1]));
      await events();
      await audio.stop();
      expect(backend.finishes, 0);
      expect(audio.error, contains('Incomplete PCM16'));
      expect(audio.transcript, isEmpty);
    },
  );

  test(
    'unexpected microphone stream completion stops and transcribes once',
    () async {
      await audio.record();
      mic.source!.add(Uint8List(32));
      await mic.source!.close();
      await events();
      await events();
      expect(backend.finishes, 1);
      expect(audio.phase, AudioPhase.ready);
    },
  );
}
