import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:dianoia_example/src/speech_player.dart';
import 'package:flutter_test/flutter_test.dart';

class _Device extends Fake implements AudioPlayer {
  final completion = StreamController<void>.broadcast();
  final states = StreamController<PlayerState>.broadcast();
  final called = Completer<void>();
  Completer<void>? starting;
  String? path;
  int stops = 0, disposals = 0;
  @override
  Stream<void> get onPlayerComplete => completion.stream;
  @override
  Stream<PlayerState> get onPlayerStateChanged => states.stream;
  @override
  Future<void> play(
    Source source, {
    double? volume,
    double? balance,
    AudioContext? ctx,
    Duration? position,
    PlayerMode? mode,
  }) async {
    path = (source as DeviceFileSource).path;
    expect(File(path!).existsSync(), isTrue);
    if (!called.isCompleted) called.complete();
    await starting?.future;
  }

  @override
  Future<void> stop() async {
    stops++;
  }

  @override
  Future<void> dispose() async {
    disposals++;
    await completion.close();
    await states.close();
  }
}

void main() {
  late Directory temp;
  late _Device device;
  late LocalSpeechPlayer player;
  setUp(() {
    temp = Directory.systemTemp.createTempSync('dianoia-speech-test-');
    device = _Device();
    player = LocalSpeechPlayer(
      createPlayer: () => device,
      temporaryDirectory: () async => temp,
    );
  });
  tearDown(() async {
    await player.dispose();
    temp.deleteSync(recursive: true);
  });

  test('creates a missing temporary directory before writing', () async {
    final missing = Directory('${temp.path}/Caches/app');
    final fresh = LocalSpeechPlayer(
      createPlayer: () => device,
      temporaryDirectory: () async => missing,
    );
    final playing = fresh.play(Uint8List.fromList([1, 2, 3]));
    await device.called.future;
    expect(File(device.path!).readAsBytesSync(), [1, 2, 3]);
    device.completion.add(null);
    await playing;
    expect(missing.listSync(), isEmpty);
    await fresh.dispose();
  });

  test(
    'completion releases its private WAV; interruptions stop without resuming',
    () async {
      final first = player.play(Uint8List.fromList([1, 2, 3]));
      await device.called.future;
      expect(File(device.path!).readAsBytesSync(), [1, 2, 3]);
      device.completion.add(null);
      await first;
      expect(temp.listSync(), isEmpty);
      final second = player.play(Uint8List(4));
      while (device.path == null || !File(device.path!).existsSync()) {
        await Future<void>.delayed(Duration.zero);
      }
      device.states.add(PlayerState.paused);
      await second;
      expect(temp.listSync(), isEmpty);
      expect(device.stops, 2);
    },
  );

  test(
    'stop during source preparation waits for and stops a late native start',
    () async {
      device.starting = Completer<void>();
      final pending = player.play(Uint8List(4));
      await device.called.future;
      var stopped = false;
      final stopping = player.stop().then((_) => stopped = true);
      await Future<void>.delayed(Duration.zero);
      expect(stopped, isFalse);
      expect(File(device.path!).existsSync(), isTrue);
      await expectLater(player.play(Uint8List(4)), throwsStateError);
      device.starting!.complete();
      await stopping;
      await pending;
      expect(device.stops, 1);
      expect(temp.listSync(), isEmpty);
    },
  );

  test(
    'native playback errors clean the source; disposal rejects playback',
    () async {
      final pending = player.play(Uint8List(4));
      final failed = expectLater(pending, throwsStateError);
      await device.called.future;
      device.completion.addError(StateError('device failed'));
      await failed;
      expect(temp.listSync(), isEmpty);
      await player.dispose();
      expect(device.disposals, 1);
      await expectLater(player.play(Uint8List(4)), throwsStateError);
    },
  );

  test('dispose during directory lookup prevents device creation', () async {
    final root = Completer<Directory>();
    var created = false;
    final local = LocalSpeechPlayer(
      createPlayer: () {
        created = true;
        return device;
      },
      temporaryDirectory: () => root.future,
    );
    final pending = local.play(Uint8List(4));
    final closing = local.dispose();
    root.complete(temp);
    await closing;
    await pending;
    expect(created, isFalse);
    expect(temp.listSync(), isEmpty);
  });
}
