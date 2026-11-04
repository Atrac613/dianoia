import 'dart:async';
import 'dart:typed_data';

import 'package:dianoia_example/src/audio_backend.dart';
import 'package:dianoia_example/src/audio_controller.dart';
import 'package:dianoia_example/src/microphone.dart';
import 'package:dianoia_example/src/pcm16_stream.dart';
import 'package:dianoia_example/src/speech_player.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:dianoia/dianoia.dart';

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
  String? installedPath;
  Completer<String>? installation;
  void Function(InstallProgress)? progress;
  int installs = 0, loads = 0;
  @override
  Future<String?> installed(String storeDirectory) async => installedPath;
  @override
  Future<String> install(
    String storeDirectory,
    void Function(InstallProgress) onProgress,
  ) async {
    installs++;
    progress = onProgress;
    return installation == null
        ? '/installed/audio'
        : await installation!.future;
  }

  final samples = <double>[];
  Completer<void>? ack;
  Completer<String>? finishing;
  Completer<Uint8List>? speech;
  final utterances = <(String, String, String)>[];
  int begins = 0, finishes = 0, cancels = 0, closes = 0;
  @override
  Future<String> load(String directory) async {
    loads++;
    return loading == null ? 'CPU' : await loading!.future;
  }

  Completer<String>? loading;

  @override
  Future<void> begin(
    String? language, {
    bool conversation = false,
    bool live = false,
    void Function(String)? onLiveText,
  }) async {
    begins++;
    conversationInput = conversation;
    liveInput = live;
    liveText = onLiveText;
    inputLanguage = language;
  }

  bool liveInput = false;
  String? inputLanguage;
  void Function(String)? liveText;

  bool conversationInput = false;
  Completer<AudioVoiceReply>? answering;
  void Function(String)? onReplyText;
  int replies = 0, acceptedReplies = 0, resets = 0;
  Completer<void>? acceptance;
  @override
  Future<AudioVoiceReply> finishConversation(
    void Function(String) onText,
  ) async {
    replies++;
    onReplyText = onText;
    return answering == null
        ? AudioVoiceReply('Four.', Uint8List.fromList([1, 2, 3]), replies)
        : await answering!.future;
  }

  @override
  Future<void> acceptConversationReply() async {
    acceptedReplies++;
    await acceptance?.future;
  }

  @override
  Future<void> newConversation() async {
    resets++;
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
    // The worker replies to a load it already finished before the cancel ACK.
    if (loading case final c? when !c.isCompleted) {
      try {
        await c.future;
      } catch (_) {}
    }
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
    'live recording accepts more than 30 seconds of samples without auto-stop',
    () async {
      audio.setTask(AudioTask.liveTranscription);
      await audio.record();
      for (var i = 0; i < 400; i++) {
        mic.source!.add(Uint8List(2560));
        await events();
      }
      expect(audio.recordedSeconds, 32);
      expect(audio.phase, AudioPhase.recording);
      expect(backend.finishes, 0);
      backend.liveText!('Transcript after thirty seconds');
      expect(audio.transcript, 'Transcript after thirty seconds');
      await audio.stop();
      expect(backend.finishes, 1);
    },
  );

  test(
    'live English text appears during recording and stop finalizes without speech playback',
    () async {
      audio.language = 'Japanese';
      audio.setTask(AudioTask.liveTranscription);
      await audio.record();
      expect(backend.liveInput, isTrue);
      expect(backend.inputLanguage, 'English');
      backend.liveText!('The quick');
      expect(audio.phase, AudioPhase.recording);
      expect(audio.transcript, 'The quick');
      backend.liveText!('The quick brown fox');
      expect(audio.transcript, 'The quick brown fox');
      mic.source!.add(Uint8List(32));
      await events();
      await audio.stop();
      expect(audio.phase, AudioPhase.ready);
      expect(audio.transcript, 'Recognized speech.');
      expect(player.played, isEmpty);
      expect(backend.replies, 0);
    },
  );

  test(
    'cancel live transcription discards partial text and ignores late updates',
    () async {
      audio.setTask(AudioTask.liveTranscription);
      await audio.record();
      backend.liveText!('Partial');
      final previous = backend.liveText!;
      await audio.cancel();
      previous('Late text');
      expect(audio.transcript, isEmpty);
      await audio.record();
      previous('Old text');
      expect(audio.transcript, isEmpty);
      backend.liveText!('New recording');
      expect(audio.transcript, 'New recording');
      await audio.cancel();
    },
  );

  test(
    'voice conversation accepts successive turns, plays answers and resets explicitly',
    () async {
      audio.setTask(AudioTask.conversation);
      for (var i = 0; i < 2; i++) {
        await audio.record();
        expect(backend.conversationInput, isTrue);
        mic.source!.add(Uint8List(32));
        await events();
        await audio.stop();
        expect(audio.phase, AudioPhase.ready);
        expect(audio.voiceTurns, hasLength(i + 1));
        expect(audio.voiceTurns.last.reply, 'Four.');
      }
      expect(backend.finishes, 0);
      expect(backend.acceptedReplies, 2);
      expect(player.played, hasLength(2));
      expect(() => audio.voiceTurns.clear(), throwsUnsupportedError);
      await audio.newConversation();
      expect(backend.resets, 1);
      expect(audio.voiceTurns, isEmpty);
      expect(audio.phase, AudioPhase.ready);
    },
  );

  test(
    'cancel rejects partial and late voice replies before acknowledgement',
    () async {
      audio.setTask(AudioTask.conversation);
      backend.answering = Completer<AudioVoiceReply>();
      await audio.record();
      mic.source!.add(Uint8List(32));
      await events();
      final stopping = audio.stop();
      await events();
      await events();
      expect(audio.phase, AudioPhase.replying);
      backend.onReplyText!('A partial answer');
      expect(audio.partialReply, 'A partial answer');
      audio.setTask(AudioTask.transcription);
      expect(audio.task, AudioTask.conversation);
      await audio.cancel();
      backend.answering!.complete(
        AudioVoiceReply('A late answer', Uint8List(44), 1),
      );
      await stopping;
      expect(audio.voiceTurns, isEmpty);
      expect(audio.partialReply, isEmpty);
      expect(backend.acceptedReplies, 0);
      expect(player.played, isEmpty);
    },
  );

  test(
    'cancel after acceptance keeps visible history but never starts late playback',
    () async {
      audio.setTask(AudioTask.conversation);
      backend.acceptance = Completer<void>();
      await audio.record();
      mic.source!.add(Uint8List(32));
      await events();
      final stopping = audio.stop();
      await events();
      await events();
      expect(backend.acceptedReplies, 1);
      expect(audio.voiceTurns, hasLength(1));
      await audio.cancel();
      backend.acceptance!.complete();
      await stopping;
      expect(player.played, isEmpty);
      expect(audio.voiceTurns, hasLength(1));
      expect(audio.phase, AudioPhase.ready);
    },
  );

  test(
    'stopping voice playback keeps completed turns for the next question',
    () async {
      audio.setTask(AudioTask.conversation);
      player.completing = Completer<void>();
      await audio.record();
      mic.source!.add(Uint8List(32));
      await events();
      final stopping = audio.stop();
      await events();
      await events();
      expect(audio.phase, AudioPhase.playing);
      await audio.cancel();
      await stopping;
      expect(audio.voiceTurns, hasLength(1));
      expect(backend.acceptedReplies, 1);
      player.completing = null;
      await audio.record();
      mic.source!.add(Uint8List(32));
      await events();
      await audio.stop();
      expect(audio.voiceTurns, hasLength(2));
    },
  );

  test(
    'model discovery never downloads; install progress leads to automatic load',
    () async {
      audio.phase = AudioPhase.needsModel;
      audio.backendName = null;
      backend.loads = 0;
      await audio.discover('/store');
      expect(audio.phase, AudioPhase.needsModel);
      expect(backend.installs, 0);
      backend.installation = Completer<String>();
      final installing = audio.downloadModel();
      await events();
      expect(audio.phase, AudioPhase.installing);
      backend.progress!(
        const InstallProgress(
          InstallStage.downloading,
          bytes: 10,
          totalBytes: 20,
        ),
      );
      expect(audio.installProgress!.fraction, .5);
      await audio.record();
      await audio.speak('Ignored');
      expect(mic.starts, 0);
      expect(backend.utterances, isEmpty);
      backend.installation!.complete('/store/bundle');
      await installing;
      expect(audio.phase, AudioPhase.ready);
      expect(audio.directory, '/store/bundle');
      expect(backend.loads, 1);
    },
  );

  test(
    'cancelled installation ignores late completion and resumes on request',
    () async {
      audio.phase = AudioPhase.needsModel;
      audio.backendName = null;
      backend.loads = 0;
      await audio.discover('/store');
      backend.installation = Completer<String>();
      final installing = audio.downloadModel();
      await events();
      await audio.cancel();
      backend.progress!(const InstallProgress(InstallStage.done));
      backend.installation!.complete('/late');
      await installing;
      expect(backend.loads, 0);
      expect(audio.phase, AudioPhase.needsModel);
      backend.installation = null;
      await audio.downloadModel();
      expect(backend.installs, 2);
      expect(audio.phase, AudioPhase.ready);
    },
  );

  test('previous installation is loaded without another download', () async {
    audio.phase = AudioPhase.needsModel;
    backend.installedPath = '/store/existing';
    await audio.discover('/store');
    expect(audio.directory, '/store/existing');
    expect(audio.phase, AudioPhase.ready);
    expect(backend.installs, 0);
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

  test('LiveASR queues several blocks while a block decodes', () async {
    audio.setTask(AudioTask.liveTranscription);
    backend.ack = Completer<void>();
    await audio.record();
    for (var i = 0; i < 13; i++) {
      mic.source!.add(Uint8List(2560));
    }
    await events();
    await events();
    expect(audio.error, isNull);
    expect(audio.phase, AudioPhase.recording);
    expect(
      audio.queuedSamples,
      lessThanOrEqualTo(AudioController.maxLiveQueuedSamples),
    );
    backend.ack!.complete();
    await events();
    expect(audio.queuedSamples, 0);
  });

  test('a cancel that overtakes a finished load keeps the model', () async {
    final fresh = AudioController(
      backend: backend,
      microphone: mic,
      player: player,
    );
    try {
      backend.loading = Completer<String>();
      final loading = fresh.load('/prepared');
      await events();
      expect(fresh.phase, AudioPhase.loading);
      final cancel = fresh.cancel();
      backend.loading!.complete('GPU');
      await cancel;
      await loading;
      expect(fresh.backendName, 'GPU');
      expect(fresh.phase, AudioPhase.ready);
    } finally {
      await fresh.close();
      fresh.dispose();
    }
  });

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
