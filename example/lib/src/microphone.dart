import 'dart:async';
import 'dart:typed_data';

import 'package:record/record.dart';

abstract interface class Microphone {
  Future<bool> requestPermission();
  Future<Stream<Uint8List>> start();
  Future<void> stop();
  Future<void> dispose();
}

final class RecordMicrophone implements Microphone {
  AudioRecorder? _recorder;
  AudioRecorder get _device => _recorder ??= AudioRecorder();
  StreamSubscription<Uint8List>? _source;

  @override
  Future<bool> requestPermission() => _device.hasPermission();

  @override
  Future<Stream<Uint8List>> start() async {
    final output = StreamController<Uint8List>();
    await _device.setOnConfigChanged((config) {
      if (config.sampleRate != 16000 ||
          config.numChannels != 1 ||
          config.encoder != AudioEncoder.pcm16bits) {
        if (!output.isClosed) {
          output.addError(
            StateError('Microphone cannot provide mono 16 kHz PCM16.'),
          );
        }
      }
    });
    final stream = await _device.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
        streamBufferSize: 2560,
      ),
    );
    _source = stream.listen(
      output.add,
      onError: output.addError,
      onDone: () => unawaited(output.close()),
    );
    return output.stream;
  }

  @override
  Future<void> stop() async {
    await _recorder?.stop();
  }

  @override
  Future<void> dispose() async {
    await _source?.cancel();
    await _recorder?.dispose();
    _source = null;
    _recorder = null;
  }
}
