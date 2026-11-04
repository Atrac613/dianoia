import 'dart:typed_data';

/// Converts mono PCM16LE packets to owned 80 ms float chunks at 16 kHz.
/// Native stream boundaries may split a 16-bit sample or an inference chunk.
final class Pcm16Packetizer {
  static const chunkSamples = 1280;
  Float32List _samples = Float32List(chunkSamples);
  int _used = 0;
  int? _low;
  bool _finished = false;
  int get pendingSamples => _used;

  List<Float32List> add(Uint8List bytes) {
    if (_finished) throw StateError('PCM stream is finished');
    final out = <Float32List>[];
    for (final byte in bytes) {
      final low = _low;
      if (low == null) {
        _low = byte;
        continue;
      }
      var sample = low | byte << 8;
      if (sample >= 32768) sample -= 65536;
      _samples[_used++] = sample / 32768;
      _low = null;
      if (_used == chunkSamples) {
        out.add(_samples);
        _samples = Float32List(chunkSamples);
        _used = 0;
      }
    }
    return out;
  }

  Float32List finish() {
    if (_finished) throw StateError('PCM stream is finished');
    _finished = true;
    if (_low != null) throw const FormatException('Incomplete PCM16 sample');
    return _samples.sublist(0, _used);
  }
}
