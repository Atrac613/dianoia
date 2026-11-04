import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:path_provider/path_provider.dart';

abstract interface class SpeechPlayer {
  /// Completes after playback ends or stop has released the audio source.
  Future<void> play(Uint8List wav);
  Future<void> stop();
  Future<void> dispose();
}

final class _Playback {
  bool stopped = false;
  final done = Completer<void>();
}

/// Lazy device access. A private temporary WAV makes local playback portable;
/// stop/completion/error removes it before another playback can start.
final class LocalSpeechPlayer implements SpeechPlayer {
  LocalSpeechPlayer({
    AudioPlayer Function()? createPlayer,
    Future<Directory> Function()? temporaryDirectory,
  }) : _createPlayer = createPlayer ?? AudioPlayer.new,
       _temporaryDirectory = temporaryDirectory ?? getTemporaryDirectory;
  final AudioPlayer Function() _createPlayer;
  final Future<Directory> Function() _temporaryDirectory;
  AudioPlayer? _player;
  _Playback? _current;
  Future<void>? _pending, _disposing;
  bool _closed = false;

  @override
  Future<void> play(Uint8List wav) {
    if (_closed || _current != null) {
      return Future.error(StateError('Speech player is unavailable'));
    }
    final current = _current = _Playback();
    // Native events can arrive while source preparation is still pending.
    current.done.future.ignore();
    return _pending = _play(current, wav);
  }

  Future<void> _play(_Playback current, Uint8List wav) async {
    Directory? temp;
    StreamSubscription<void>? subscription;
    StreamSubscription<PlayerState>? states;
    AudioPlayer? player;
    try {
      temp = await (await _temporaryDirectory()).createTemp('dianoia-speech-');
      if (current.stopped) return;
      final file = File('${temp.path}/speech.wav');
      await file.writeAsBytes(wav, flush: true);
      if (current.stopped) return;
      player = _player ??= _createPlayer();
      subscription = player.onPlayerComplete.listen(
        (_) {
          if (!current.done.isCompleted) current.done.complete();
        },
        onError: (Object error, StackTrace stack) {
          if (!current.done.isCompleted) {
            current.done.completeError(error, stack);
          }
        },
      );
      states = player.onPlayerStateChanged.listen((state) {
        if ((state == PlayerState.paused || state == PlayerState.stopped) &&
            !current.done.isCompleted) {
          current.done.complete();
        }
      });
      await player.play(
        DeviceFileSource(file.path, mimeType: 'audio/wav'),
        ctx: AudioContext(
          android: const AudioContextAndroid(
            contentType: AndroidContentType.speech,
          ),
        ),
      );
      await current.done.future;
    } finally {
      try {
        // A stop during async source preparation must also stop a late start.
        await player?.stop();
      } finally {
        try {
          await subscription?.cancel();
          await states?.cancel();
          if (temp != null) await temp.delete(recursive: true);
        } finally {
          _current = null;
        }
      }
    }
  }

  @override
  Future<void> stop() async {
    final current = _current;
    if (current == null) return;
    current.stopped = true;
    if (!current.done.isCompleted) current.done.complete();
    await _pending;
  }

  @override
  Future<void> dispose() => _disposing ??= _dispose();
  Future<void> _dispose() async {
    _closed = true;
    try {
      await stop();
    } finally {
      await _player?.dispose();
    }
  }
}
