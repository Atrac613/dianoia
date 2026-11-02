/// CPU reference frontend for the official LFM3-Audio-2B checkpoint.
///
/// Accepts finite mono float PCM at 16 kHz. Resampling, file decoding and
/// microphone capture belong to the caller. No model weights are needed.
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// One clip's normalized log-mel features in [mel bin, frame] order.
final class AudioFeatures {
  AudioFeatures._(this.values, this.frames);

  /// Copies already normalized [128, frames] log-mel values.
  factory AudioFeatures.fromLogMel(Float32List values, {required int frames}) {
    if (frames < 0 ||
        values.length != Lfm3AudioFrontend.melBins * frames ||
        values.any((x) => !x.isFinite)) {
      throw ArgumentError('expected finite [128, frames] log-mel features');
    }
    return AudioFeatures._(Float32List.fromList(values), frames);
  }

  final Float32List values;
  final int frames;
  int get melBins => Lfm3AudioFrontend.melBins;

  /// Length after the encoder's three stride-2 downsampling convolutions.
  int get encoderTokens => (frames + 7) ~/ 8;
}

/// Fixed preprocessing contract of LFM3-Audio-2B format version 1.
///
/// Uses a periodic Hann window, uncentered 400-point DFT, power spectrum,
/// 128 area-normalized Slaney filters, log10 floor -12 and (x + 4) / 4.
/// The direct DFT is a portable reference, not a realtime optimized FFT.
final class Lfm3AudioFrontend {
  static const sampleRate = 16000;
  static const melBins = 128;
  static const windowSize = 400;
  static const hopSize = 160;
  static const _bins = windowSize ~/ 2 + 1;

  late final Float32List _window = Float32List.fromList([
    for (var i = 0; i < windowSize; i++)
      0.5 - 0.5 * math.cos(2 * math.pi * i / windowSize),
  ]);
  late final Float64List _cos = Float64List.fromList([
    for (var k = 0; k < _bins; k++)
      for (var i = 0; i < windowSize; i++)
        math.cos(2 * math.pi * k * i / windowSize),
  ]);
  late final Float64List _sin = Float64List.fromList([
    for (var k = 0; k < _bins; k++)
      for (var i = 0; i < windowSize; i++)
        math.sin(2 * math.pi * k * i / windowSize),
  ]);
  late final Float32List _filters = _makeFilters();

  /// Number of frames after right-padding a clip to include its last sample.
  static int frameCount(int samples) {
    if (samples < 0) throw ArgumentError.value(samples, 'samples');
    return samples <= windowSize
        ? 1
        : 1 + (samples - windowSize + hopSize - 1) ~/ hopSize;
  }

  AudioFeatures extract(Float32List pcm, {required int samplingRate}) {
    _validate(pcm, samplingRate);
    return _extract(pcm, frameCount(pcm.length));
  }

  /// Independent stream state; the immutable tables are shared by streams.
  Lfm3AudioFeatureStream startStream() => Lfm3AudioFeatureStream._(this);

  void _validate(Float32List pcm, int samplingRate) {
    if (samplingRate != sampleRate) {
      throw ArgumentError.value(samplingRate, 'samplingRate', 'expected 16000');
    }
    for (final sample in pcm) {
      if (!sample.isFinite) {
        throw ArgumentError('PCM samples must be finite');
      }
    }
  }

  AudioFeatures _extract(Float32List pcm, int frames) {
    final out = Float32List(melBins * frames);
    final windowed = Float32List(windowSize);
    final power = Float32List(_bins);
    for (var t = 0; t < frames; t++) {
      for (var i = 0; i < windowSize; i++) {
        final s = t * hopSize + i;
        windowed[i] = s < pcm.length ? pcm[s] * _window[i] : 0;
      }
      for (var k = 0; k < _bins; k++) {
        var real = 0.0;
        var imag = 0.0;
        for (var i = 0; i < windowSize; i++) {
          real += windowed[i] * _cos[k * windowSize + i];
          imag -= windowed[i] * _sin[k * windowSize + i];
        }
        power[k] = real * real + imag * imag;
      }
      for (var m = 0; m < melBins; m++) {
        var energy = 0.0;
        for (var k = 0; k < _bins; k++) {
          energy += _filters[m * _bins + k] * power[k];
        }
        final log = energy <= 1e-12 ? -12.0 : math.log(energy) / math.ln10;
        out[m * frames + t] = (log + 4) / 4;
      }
    }
    return AudioFeatures._(out, frames);
  }

  static Float32List _makeFilters() {
    const linearScale = 200.0 / 3;
    final logStep = math.log(6.4) / 27;
    double hzToMel(double hz) =>
        hz < 1000 ? hz / linearScale : 15 + math.log(hz / 1000) / logStep;
    double melToHz(double mel) =>
        mel < 15 ? linearScale * mel : 1000 * math.exp(logStep * (mel - 15));
    final maxMel = hzToMel(sampleRate / 2);
    final edges = [
      for (var i = 0; i < melBins + 2; i++) melToHz(maxMel * i / (melBins + 1)),
    ];
    final result = Float32List(melBins * _bins);
    for (var m = 0; m < melBins; m++) {
      final area = 2 / (edges[m + 2] - edges[m]);
      for (var k = 0; k < _bins; k++) {
        final hz = k * sampleRate / windowSize;
        final lower = (hz - edges[m]) / (edges[m + 1] - edges[m]);
        final upper = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
        result[m * _bins + k] = math.max(0, math.min(lower, upper)) * area;
      }
    }
    return result;
  }
}

/// Incremental frontend retaining only overlapping, unconsumed PCM samples.
///
/// Concatenate returned frames along the time axis. Only [finish] pads the
/// tail; frames from an intermediate chunk may be empty. This state covers
/// feature extraction only, not the audio encoder's lookahead or caches.
final class Lfm3AudioFeatureStream {
  Lfm3AudioFeatureStream._(this._frontend);

  final Lfm3AudioFrontend _frontend;
  Float32List _pending = Float32List(0);
  int _samples = 0;
  int _frames = 0;
  bool _finished = false;

  AudioFeatures add(Float32List pcm, {required int samplingRate}) =>
      _add(pcm, samplingRate, false);

  AudioFeatures finish() =>
      _add(Float32List(0), Lfm3AudioFrontend.sampleRate, true);

  AudioFeatures _add(Float32List pcm, int samplingRate, bool finalChunk) {
    if (_finished) throw StateError('Audio feature stream is finalized');
    _frontend._validate(pcm, samplingRate);
    final buffer = Float32List(_pending.length + pcm.length)
      ..setAll(0, _pending)
      ..setAll(_pending.length, pcm);
    final total = _samples + pcm.length;
    final count = finalChunk
        ? Lfm3AudioFrontend.frameCount(total) - _frames
        : buffer.length < Lfm3AudioFrontend.windowSize
        ? 0
        : 1 +
              (buffer.length - Lfm3AudioFrontend.windowSize) ~/
                  Lfm3AudioFrontend.hopSize;
    final features = _frontend._extract(buffer, count);
    _pending = finalChunk
        ? Float32List(0)
        : buffer.sublist(count * Lfm3AudioFrontend.hopSize);
    _samples = total;
    _frames += count;
    _finished = finalChunk;
    return features;
  }
}
