import 'dart:async';
import 'dart:collection';

import 'package:flutter/foundation.dart';

import 'audio_backend.dart';
import 'audio_worker.dart';
import 'microphone.dart';
import 'pcm16_stream.dart';
import 'speech_player.dart';

enum AudioPhase {
  needsModel,
  loading,
  ready,
  requestingPermission,
  recording,
  finishing,
  synthesizing,
  playing,
  cancelling,
  failed,
  closed,
}

/// UI-side capture and bounded backpressure. Models and arithmetic stay in the
/// backend isolate; microphone PCM is never saved. Generated speech is passed
/// to the player, which owns its temporary local source until playback ends.
final class AudioController extends ChangeNotifier {
  AudioController({
    AudioBackend? backend,
    Microphone? microphone,
    SpeechPlayer? player,
  }) : _backend = backend ?? AudioIsolateBackend(),
       _microphone = microphone ?? RecordMicrophone(),
       _player = player ?? LocalSpeechPlayer();
  final AudioBackend _backend;
  final Microphone _microphone;
  final SpeechPlayer _player;
  static const maxSamples = 30 * 16000;
  static const maxQueuedSamples = 16000;
  final _queue = ListQueue<Float32List>();
  Pcm16Packetizer? _packetizer;
  StreamSubscription<Uint8List>? _subscription;
  Completer<void>? _captureDone;
  Timer? _limit;
  Future<void>? _pumping, _cancelling, _closing;
  int _epoch = 0, _bytes = 0, _queuedSamples = 0;
  bool _disposed = false;

  AudioPhase phase = AudioPhase.needsModel;
  String? directory, backendName, error;
  String transcript = '';
  String? language = 'Japanese';
  double get recordedSeconds => _bytes / 2 / 16000;
  int get queuedSamples => _queuedSamples;
  bool get active => switch (phase) {
    AudioPhase.requestingPermission ||
    AudioPhase.recording ||
    AudioPhase.finishing ||
    AudioPhase.synthesizing ||
    AudioPhase.playing ||
    AudioPhase.cancelling => true,
    _ => false,
  };

  bool _valid(int epoch) => !_disposed && _closing == null && epoch == _epoch;
  void _notify() {
    if (!_disposed) notifyListeners();
  }

  Future<void> load(String path) async {
    if ((phase != AudioPhase.needsModel && phase != AudioPhase.failed) ||
        _closing != null) {
      return;
    }
    final epoch = ++_epoch;
    phase = AudioPhase.loading;
    error = null;
    directory = path;
    _notify();
    try {
      final used = await _backend.load(path);
      if (!_valid(epoch)) return;
      backendName = used;
      phase = AudioPhase.ready;
    } catch (e) {
      if (!_valid(epoch)) return;
      error = '$e';
      phase = AudioPhase.failed;
    }
    _notify();
  }

  Future<void> speak(
    String text, {
    String language = 'ja',
    String voice = 'female',
  }) async {
    if (phase != AudioPhase.ready || _closing != null) return;
    if (text.trim().isEmpty) {
      error = 'Enter text to speak.';
      _notify();
      return;
    }
    final epoch = ++_epoch;
    phase = AudioPhase.synthesizing;
    error = null;
    _notify();
    if (!_valid(epoch)) return;
    try {
      final wav = await _backend.synthesize(text, language, voice);
      if (!_valid(epoch)) return;
      phase = AudioPhase.playing;
      _notify();
      if (!_valid(epoch)) return;
      await _player.play(wav);
      if (!_valid(epoch)) return;
      phase = AudioPhase.ready;
      _notify();
    } catch (e) {
      if (_valid(epoch)) await cancel(message: 'Speech failed: $e');
    }
  }

  Future<void> record() async {
    if (phase != AudioPhase.ready || _closing != null) return;
    final epoch = ++_epoch;
    phase = AudioPhase.requestingPermission;
    error = null;
    transcript = '';
    _bytes = 0;
    _queuedSamples = 0;
    _queue.clear();
    _packetizer = Pcm16Packetizer();
    _notify();
    try {
      final allowed = await _microphone.requestPermission();
      if (!_valid(epoch)) return;
      if (!allowed) {
        error =
            'Microphone access was denied. Allow it in system settings to record.';
        phase = AudioPhase.ready;
        _notify();
        return;
      }
      await _backend.begin(language);
      if (!_valid(epoch)) return;
      final stream = await _microphone.start();
      if (!_valid(epoch)) {
        await _microphone.stop();
        return;
      }
      _captureDone = Completer<void>();
      phase = AudioPhase.recording;
      _subscription = stream.listen(
        (bytes) => _onBytes(bytes, epoch),
        onError: (Object e) {
          if (_valid(epoch)) unawaited(cancel(message: 'Microphone error: $e'));
        },
        onDone: () {
          final done = _captureDone;
          if (done != null && !done.isCompleted) done.complete();
          if (_valid(epoch) && phase == AudioPhase.recording) unawaited(stop());
        },
      );
      _limit = Timer(const Duration(seconds: 30), () => unawaited(stop()));
      _notify();
    } catch (e) {
      if (_valid(epoch)) await cancel(message: 'Could not start recording: $e');
    }
  }

  void _onBytes(Uint8List bytes, int epoch) {
    if (!_valid(epoch) ||
        (phase != AudioPhase.recording && phase != AudioPhase.finishing)) {
      return;
    }
    final remaining = maxSamples * 2 - _bytes;
    if (remaining <= 0) return;
    final accepted = bytes.length <= remaining
        ? bytes
        : bytes.sublist(0, remaining);
    try {
      // Bound work before converting a native packet, rather than pausing an
      // unbounded plugin stream or silently discarding samples when behind.
      if (_queuedSamples +
              (_packetizer?.pendingSamples ?? 0) +
              (accepted.length + 1) ~/ 2 >
          maxQueuedSamples) {
        throw StateError(
          'Audio processing fell behind. Please try a shorter recording.',
        );
      }
      _bytes += accepted.length;
      for (final rows in _packetizer!.add(accepted)) {
        _enqueue(rows, epoch);
      }
      _notify();
      if (_bytes == maxSamples * 2 && phase == AudioPhase.recording) {
        unawaited(stop());
      }
    } catch (e) {
      unawaited(cancel(message: '$e'));
    }
  }

  void _enqueue(Float32List rows, int epoch) {
    if (rows.isEmpty || !_valid(epoch)) return;
    if (_queuedSamples + rows.length > maxQueuedSamples) {
      throw StateError(
        'Audio processing fell behind. Please try a shorter recording.',
      );
    }
    _queue.add(rows);
    _queuedSamples += rows.length;
    _pumping ??= _pump(epoch);
  }

  Future<void> _pump(int epoch) async {
    try {
      while (_valid(epoch) && _queue.isNotEmpty) {
        final samples = _queue.removeFirst();
        await _backend.addPcm(samples);
        if (_valid(epoch)) _queuedSamples -= samples.length;
      }
    } catch (e) {
      if (_valid(epoch)) {
        unawaited(cancel(message: 'Audio processing failed: $e'));
      }
    } finally {
      _pumping = null;
    }
  }

  Future<void> stop() async {
    if (phase != AudioPhase.recording || _closing != null) return;
    final epoch = _epoch;
    phase = AudioPhase.finishing;
    _limit?.cancel();
    _notify();
    try {
      await _microphone.stop();
      await _captureDone?.future.timeout(const Duration(seconds: 2));
      if (!_valid(epoch)) return;
      _enqueue(_packetizer!.finish(), epoch);
      await _pumping;
      if (!_valid(epoch)) return;
      if (_bytes == 0) {
        throw StateError('No audio was received. Please try again.');
      }
      final text = await _backend.finish();
      if (!_valid(epoch)) return;
      transcript = text;
      phase = AudioPhase.ready;
      await _subscription?.cancel();
      _subscription = null;
      _packetizer = null;
      _notify();
    } catch (e) {
      if (_valid(epoch)) await cancel(message: 'Transcription failed: $e');
    }
  }

  Future<void> cancel({String? message}) {
    if (_cancelling != null) return _cancelling!;
    if (!active && phase != AudioPhase.loading) return Future<void>.value();
    final task = _cancel(message);
    _cancelling = task;
    unawaited(task.whenComplete(() => _cancelling = null));
    return task;
  }

  Future<void> _cancel(String? message) async {
    final epoch = ++_epoch;
    phase = AudioPhase.cancelling;
    _limit?.cancel();
    transcript = '';
    _queue.clear();
    _queuedSamples = 0;
    _packetizer = null;
    _notify();
    // Interrupt the worker immediately, without waiting for an outstanding ACK.
    final cancelled = _quiet(_backend.cancel());
    final stopped = _quiet(_player.stop());
    try {
      await _microphone.stop();
    } catch (e) {
      message ??= 'Microphone stop failed: $e';
      await _quiet(_microphone.dispose());
    }
    await _subscription?.cancel();
    _subscription = null;
    await cancelled;
    await stopped;
    await _pumping;
    if (!_valid(epoch)) return;
    error = message;
    phase = backendName == null ? AudioPhase.needsModel : AudioPhase.ready;
    _notify();
  }

  Future<void> close() => _closing ??= _close();
  Future<void> _close() async {
    ++_epoch;
    _limit?.cancel();
    _queue.clear();
    _queuedSamples = 0;
    _packetizer = null;
    phase = AudioPhase.closed;
    final cancelled = _quiet(_backend.cancel());
    final workerClosed = _quiet(_backend.close());
    final playerClosed = _quiet(_player.dispose());
    await _quiet(_microphone.stop());
    await _subscription?.cancel();
    _subscription = null;
    await cancelled;
    await _pumping;
    await workerClosed;
    await playerClosed;
    await _quiet(_microphone.dispose());
  }

  static Future<void> _quiet(Future<void> work) async {
    try {
      await work;
    } catch (_) {
      /* Cleanup continues. */
    }
  }

  @override
  void dispose() {
    _disposed = true;
    unawaited(close());
    super.dispose();
  }
}
