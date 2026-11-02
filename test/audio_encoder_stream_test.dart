import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/gguf_fixture.dart';

Map<String, dynamic> _read(String path) =>
    jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;

AudioFeatures _features(int frames) => AudioFeatures.fromLogMel(
  Float32List.fromList([
    for (var i = 0; i < 128 * frames; i++)
      ((i * 17 + i ~/ 5 * 3) % 127 - 63) / 64,
  ]),
  frames: frames,
);

AudioFeatures _part(AudioFeatures all, int start, int count) =>
    AudioFeatures.fromLogMel(
      Float32List.fromList([
        for (var m = 0; m < 128; m++)
          ...all.values.sublist(
            m * all.frames + start,
            m * all.frames + start + count,
          ),
      ]),
      frames: count,
    );

void _near(List<double> actual, List<num> expected, double tolerance) {
  expect(actual.length, expected.length);
  for (var i = 0; i < actual.length; i++) {
    expect(
      actual[i].isFinite &&
          (actual[i] - expected[i]).abs() <=
              tolerance * (1 + expected[i].abs()),
      isTrue,
      reason: '$i: ${actual[i]} vs ${expected[i]}',
    );
  }
}

int _cacheBound(Lfm3AudioEncoderConfig c) =>
    (128 * 15 +
        c.convLayers.where((v) => v).length * 2 * c.hiddenSize +
        c.convLayers.where((v) => !v).length *
            2 *
            (c.window - 1) *
            c.numKvHeads *
            c.headDim) *
    4;

void _compareOracle(
  Lfm3AudioEncoder encoder,
  Map<String, dynamic> fixture,
  double tolerance,
) {
  for (final c in fixture['cases'] as List) {
    final features = _features(c['frames'] as int);
    final stream = encoder.startStream();
    var at = 0, emitted = 0;
    for (final chunk in c['chunks'] as List) {
      final count = chunk['input_frames'] as int;
      final rows = chunk['is_final'] as bool
          ? stream.finish()
          : stream.add(_part(features, at, count));
      at += count;
      emitted += rows.length ~/ encoder.config.projectionSize;
      expect(
        rows.length,
        chunk['output_tokens'] * encoder.config.projectionSize,
      );
      _near(rows, (chunk['projected'] as List).cast<num>(), tolerance);
      expect(stream.frames, at);
      expect(stream.tokens, emitted);
      expect(stream.retainedFeatureFrames, lessThanOrEqualTo(15));
      expect(
        stream.cachedStateBytes,
        lessThanOrEqualTo(_cacheBound(encoder.config)),
      );
    }
    expect(at, features.frames);
    expect(emitted, features.encoderTokens);
    expect(stream.isFinished, isTrue);
    expect(stream.cachedStateBytes, 0);
    expect(stream.finish, throwsStateError);
  }
}

void main() {
  final supported = NativeKernels.isAvailable;
  final oracle = _read('test/fixtures/audio/encoder_oracle.json');
  final streamOracle = _read('test/fixtures/audio/encoder_stream_oracle.json');
  late Directory temp;
  late Lfm3AudioEncoder encoder;
  setUp(() {
    temp = Directory.systemTemp.createTempSync('dianoia-audio-stream-');
    if (!supported) return;
    NativeKernels.load(null, threads: 2);
    writeTestGguf(
      '${temp.path}/mmproj.gguf',
      (oracle['metadata'] as Map).cast<String, Object>(),
      [
        for (final t in oracle['tensors'] as List)
          (
            t['name'] as String,
            (t['shape'] as List).cast<int>(),
            GgufType.values.byName(t['type'] as String),
            base64Decode(t['bytes'] as String),
          ),
      ],
    );
    encoder = Lfm3AudioEncoder.load('${temp.path}/mmproj.gguf');
  });
  tearDown(() {
    if (supported) encoder.close();
    temp.deleteSync(recursive: true);
  });

  group('streaming Q4 encoder', () {
    test('matches official streaming Torch outputs and emission counts', () {
      _compareOracle(encoder, streamOracle, 3e-5);
    });

    for (final chunkSize in [1, 7, 8, 9, 17, 64, 1000]) {
      test('matches batch at every stage with $chunkSize-frame chunks', () {
        final all = _features(265);
        final expected = <String, List<double>>{};
        encoder.encode(all, capture: (name, rows) => expected[name] = rows);
        final actual = <String, List<double>>{};
        void capture(String name, Float32List rows) =>
            actual.putIfAbsent(name, () => []).addAll(rows);
        final stream = encoder.startStream();
        var at = 0;
        while (at < all.frames) {
          final count = chunkSize.clamp(0, all.frames - at);
          stream.add(_part(all, at, count), capture: capture);
          at += count;
          expect(stream.tokens, at ~/ 8);
          expect(
            stream.cachedStateBytes,
            lessThanOrEqualTo(_cacheBound(encoder.config)),
          );
        }
        stream.finish(capture: capture);
        expect(actual.keys, expected.keys);
        for (final name in expected.keys) {
          _near(actual[name]!, expected[name]!, 3e-5);
        }
      });
    }

    test(
      'PCM frontend and encoder stream compose without retaining the clip',
      () {
        final pcm = Float32List.fromList([
          for (var i = 0; i < 12001; i++)
            ((i * 73 + i ~/ 7 * 19) % 1024 - 512) / 1024,
        ]);
        final frontend = Lfm3AudioFrontend();
        final expected = encoder.encode(
          frontend.extract(pcm, samplingRate: 16000),
        );
        final features = frontend.startStream(), stream = encoder.startStream();
        final actual = <double>[];
        for (var at = 0; at < pcm.length; at += 137) {
          actual.addAll(
            stream.add(
              features.add(
                pcm.sublist(at, (at + 137).clamp(0, pcm.length)),
                samplingRate: 16000,
              ),
            ),
          );
        }
        actual.addAll(stream.add(features.finish()));
        actual.addAll(stream.finish());
        _near(actual, expected, 3e-5);
      },
    );

    test('independent streams, invalid input retry and owned buffers', () {
      final all = _features(33), other = _features(9);
      final a = encoder.startStream(), b = encoder.startStream();
      final first = _part(all, 0, 7);
      expect(a.add(first), isEmpty);
      first.values.fillRange(0, first.values.length, 0);
      final invalid = _part(all, 7, 1)..values[0] = double.nan;
      expect(() => a.add(invalid), throwsArgumentError);
      expect(a.frames, 7);
      final bRows = [...b.add(other), ...b.finish()];
      final aRows = [...a.add(_part(all, 7, 26)), ...a.finish()];
      _near(aRows, encoder.encode(all), 3e-5);
      _near(bRows, encoder.encode(other), 3e-5);
      a.close();
      b.close();
      expect(() => a.add(_features(1)), throwsStateError);
    });

    test('callback failure releases caches and forbids reuse', () {
      final stream = encoder.startStream();
      expect(
        () => stream.add(
          _features(8),
          capture: (_, _) => throw StateError('stop'),
        ),
        throwsStateError,
      );
      expect(stream.isFinished, isTrue);
      expect(stream.cachedStateBytes, 0);
      expect(() => stream.add(_features(8)), throwsStateError);
      expect(encoder.encode(_features(8)), isNotEmpty);
    });

    test('guards reentrant calls and closes streams with their encoder', () {
      final stream = encoder.startStream();
      stream.add(
        _features(8),
        capture: (_, _) {
          expect(() => stream.add(_features(0)), throwsStateError);
          expect(stream.close, throwsStateError);
          expect(encoder.close, throwsStateError);
        },
      );
      expect(stream.isFinished, isFalse);
      encoder.close();
      expect(stream.isFinished, isTrue);
      expect(stream.cachedStateBytes, 0);
      expect(stream.finish, throwsStateError);
      expect(encoder.startStream, throwsStateError);
    });
  }, skip: supported ? false : 'native kernels unavailable');

  final real = Platform.environment['DIANOIA_AUDIO_Q4_MMPROJ'];
  final realOracle =
      Platform.environment['DIANOIA_AUDIO_ENCODER_STREAM_ORACLE'];
  test(
    'official Q4_0 streaming encoder matches upstream including window wrap',
    () async {
      final fixture = _read(realOracle!);
      final digest = (await sha256.bind(File(real!).openRead()).first)
          .toString();
      expect(digest, Lfm3AudioQ4Source.audio.sha256);
      expect(fixture['gguf_sha256'], digest);
      final realEncoder = Lfm3AudioEncoder.load(real);
      try {
        _compareOracle(realEncoder, fixture, 2e-4);
      } finally {
        realEncoder.close();
      }
    },
    skip: supported && real != null && realOracle != null
        ? false
        : 'set DIANOIA_AUDIO_Q4_MMPROJ and DIANOIA_AUDIO_ENCODER_STREAM_ORACLE',
  );
}
