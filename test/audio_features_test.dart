import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

void main() {
  final frontend = Lfm3AudioFrontend();
  test('precomputed log-mel validates shape and owns its samples', () {
    final values = Float32List(128);
    final result = AudioFeatures.fromLogMel(values, frames: 1);
    values[0] = 1;
    expect(result.values[0], 0);
    expect(
      () => AudioFeatures.fromLogMel(values, frames: 2),
      throwsArgumentError,
    );
    expect(
      () => AudioFeatures.fromLogMel(Float32List(0), frames: -1),
      throwsArgumentError,
    );
    values[0] = double.nan;
    expect(
      () => AudioFeatures.fromLogMel(values, frames: 1),
      throwsArgumentError,
    );
  });
  final fixture =
      jsonDecode(
            File('test/fixtures/audio/features_oracle.json').readAsStringSync(),
          )
          as Map<String, dynamic>;
  Float32List samples(int count, {bool silence = false}) =>
      Float32List.fromList([
        for (var i = 0; i < count; i++)
          silence ? 0 : (((i * 73 + (i ~/ 7) * 19) % 1024) - 512) / 1024,
      ]);

  for (final c in fixture['cases'] as List) {
    test('official torch frontend: ${c['name']}', () {
      final result = frontend.extract(
        samples(c['samples'] as int, silence: c['name'] == 'silence'),
        samplingRate: 16000,
      );
      expect(result.frames, c['frames']);
      expect(result.melBins, 128);
      expect(result.encoderTokens, c['encoder_tokens']);
      final expected = (c['values'] as List).cast<num>();
      expect(result.values.length, expected.length);
      for (var i = 0; i < expected.length; i++) {
        expect(
          result.values[i],
          closeTo(expected[i], 2e-5),
          reason: 'bin/frame $i',
        );
      }
    });
  }

  for (final length in [0, 1, 399, 400, 401, 560, 561, 2000]) {
    for (final chunkSize in [1, 159, 160, 399, 400, 777]) {
      test('stream equals batch: $length samples, $chunkSize chunk', () {
        final pcm = samples(length);
        final expected = frontend.extract(pcm, samplingRate: 16000);
        final stream = frontend.startStream();
        final chunks = <AudioFeatures>[];
        for (var start = 0; start < pcm.length; start += chunkSize) {
          final end = (start + chunkSize).clamp(0, pcm.length);
          chunks.add(stream.add(pcm.sublist(start, end), samplingRate: 16000));
        }
        chunks.add(stream.finish());
        expect(chunks.fold(0, (n, c) => n + c.frames), expected.frames);
        final joined = <double>[];
        for (var m = 0; m < 128; m++) {
          for (final chunk in chunks) {
            joined.addAll(
              chunk.values.sublist(m * chunk.frames, (m + 1) * chunk.frames),
            );
          }
        }
        expect(joined, orderedEquals(expected.values));
        expect(stream.finish, throwsStateError);
        expect(
          () => stream.add(Float32List(0), samplingRate: 16000),
          throwsStateError,
        );
      });
    }
  }

  test(
    'rejects wrong sample rate and nonfinite PCM without consuming stream',
    () {
      final stream = frontend.startStream();
      for (final invalid in [
        double.nan,
        double.infinity,
        double.negativeInfinity,
      ]) {
        expect(
          () =>
              stream.add(Float32List.fromList([invalid]), samplingRate: 16000),
          throwsArgumentError,
        );
        expect(
          () => frontend.extract(
            Float32List.fromList([invalid]),
            samplingRate: 16000,
          ),
          throwsArgumentError,
        );
      }
      expect(
        () => stream.add(samples(400), samplingRate: 24000),
        throwsArgumentError,
      );
      expect(
        () => frontend.extract(samples(400), samplingRate: 24000),
        throwsArgumentError,
      );
      expect(() => Lfm3AudioFrontend.frameCount(-1), throwsArgumentError);
      final result = stream.add(samples(400), samplingRate: 16000);
      expect(
        result.values,
        orderedEquals(
          frontend.extract(samples(400), samplingRate: 16000).values,
        ),
      );
      expect(stream.finish().frames, 0);
    },
  );

  test('streams own their PCM buffers and can be interleaved', () {
    final pcm = samples(560);
    final a = frontend.startStream();
    final b = frontend.startStream();
    a.add(pcm.sublist(0, 300), samplingRate: 16000);
    final mutable = pcm.sublist(0, 300);
    b.add(mutable, samplingRate: 16000);
    mutable.fillRange(0, mutable.length, 0);
    expect(
      a.add(pcm.sublist(300), samplingRate: 16000).values,
      orderedEquals(b.add(pcm.sublist(300), samplingRate: 16000).values),
    );
  });
}
