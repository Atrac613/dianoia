import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/gguf_fixture.dart';

void main() {
  final supported = NativeKernels.isAvailable;
  final oracle =
      jsonDecode(
            File('test/fixtures/audio/encoder_oracle.json').readAsStringSync(),
          )
          as Map<String, dynamic>;
  final tensors = [
    for (final t in oracle['tensors'] as List)
      (
        t['name'] as String,
        (t['shape'] as List).cast<int>(),
        GgufType.values.byName(t['type'] as String),
        base64Decode(t['bytes'] as String),
      ),
  ];
  final metadata = (oracle['metadata'] as Map).cast<String, Object>();
  late Directory temp;
  late String path;
  setUp(() {
    temp = Directory.systemTemp.createTempSync('dianoia-audio-encoder-');
    path = '${temp.path}/mmproj.gguf';
    if (supported) NativeKernels.load(null, threads: 2);
    writeTestGguf(path, metadata, tensors);
  });
  tearDown(() => temp.deleteSync(recursive: true));

  void compare(
    Lfm3AudioEncoder encoder,
    Map<String, dynamic> fixture, {
    double tolerance = 3e-5,
  }) {
    for (final c in fixture['cases'] as List) {
      final frames = c['frames'] as int;
      final features = AudioFeatures.fromLogMel(
        Float32List.fromList([
          for (var i = 0; i < 128 * frames; i++)
            ((i * 17 + i ~/ 5 * 3) % 127 - 63) / 64,
        ]),
        frames: frames,
      );
      final stages = <String>[];
      final rows = encoder.encode(
        features,
        capture: (stage, rows) {
          stages.add(stage);
          final expected = (c['stages'][stage] as List).cast<num>();
          expect(rows.length, expected.length, reason: '$frames/$stage');
          var maxError = 0.0;
          for (var i = 0; i < rows.length; i++) {
            final error = (rows[i] - expected[i]).abs();
            maxError = math.max(maxError, error);
            expect(
              rows[i].isFinite && error <= tolerance * (1 + expected[i].abs()),
              isTrue,
              reason:
                  '$frames/$stage/$i: ${rows[i]} vs ${expected[i]}, max $maxError',
            );
          }
        },
      );
      expect(stages, (c['stages'] as Map).keys.toList());
      expect(
        rows.length,
        features.encoderTokens * encoder.config.projectionSize,
      );
    }
  }

  test('pinned Q4 source is distinct from QAD and covers both GGUF files', () {
    expect(
      Lfm3AudioQ4Source.files.fold(0, (n, f) => n + f.size),
      Lfm3AudioQ4Source.downloadBytes,
    );
    for (final file in Lfm3AudioQ4Source.files) {
      expect(file.url, contains('/${Lfm3AudioQ4Source.revision}/'));
      expect(file.name, endsWith('Q4_0.gguf'));
      expect(file.name, isNot(contains('QAD')));
      expect(file.sha256, matches(RegExp(r'^[0-9a-f]{64}$')));
    }
  });

  test('synthetic Q4 GGUF encoder matches official Torch at every stage', () {
    final encoder = Lfm3AudioEncoder.load(path);
    try {
      expect(encoder.config.convLayers, [true, false, true]);
      expect(encoder.config.window, 2);
      compare(encoder, oracle);
      final empty = AudioFeatures.fromLogMel(Float32List(0), frames: 0);
      expect(() => encoder.encode(empty), throwsArgumentError);
    } finally {
      encoder.close();
    }
    encoder.close();
    expect(
      () => encoder.encode(
        Lfm3AudioFrontend().extract(Float32List(0), samplingRate: 16000),
      ),
      throwsStateError,
    );
  }, skip: supported ? false : 'native kernels unavailable');

  test('validates all tensor shapes and Q4 types before loading matrices', () {
    final variants = [
      tensors.where((t) => t.$1 != 'a.blk.0.conv_pw1.weight').toList(),
      [
        for (final t in tensors)
          if (t.$1 == 'a.blk.0.conv_pw1.weight')
            (
              t.$1,
              t.$2,
              GgufType.f32,
              Uint8List(t.$2.reduce((a, b) => a * b) * 4),
            )
          else
            t,
      ],
      [
        ...tensors,
        ('a.blk.0.attn_q.weight', [32, 32], GgufType.q4_0, Uint8List(32 * 18)),
      ],
      [
        ...tensors,
        ('a.blk.0.unexpected.weight', [32], GgufType.f32, Uint8List(128)),
      ],
    ];
    for (final variant in variants) {
      writeTestGguf(path, metadata, variant);
      expect(() => Lfm3AudioEncoder.load(path), throwsFormatException);
    }
  }, skip: supported ? false : 'native kernels unavailable');

  test('rejects wrong projector and incompatible metadata', () {
    for (final change in <Map<String, Object>>[
      {'clip.audio.projector_type': 'lfm3vl'},
      {'clip.audio.attention.head_count': 3},
      {'clip.audio.attention.head_count_kv': 3},
      {'clip.audio.window_size': 0},
      {'clip.audio.num_mel_bins': 80},
      {'clip.audio.attention.layer_norm_epsilon': double.nan},
    ]) {
      writeTestGguf(path, {...metadata, ...change}, tensors);
      expect(() => Lfm3AudioEncoder.load(path), throwsFormatException);
    }
  }, skip: supported ? false : 'native kernels unavailable');

  test('copied GGUF Q4 blocks retain scales and both nibble halves', () {
    final raw = Uint8List.fromList([
      0,
      0x20,
      for (var i = 0; i < 16; i++) (i << 4) | (15 - i),
      0,
      0xa0,
      for (var i = 0; i < 16; i++) (i << 4) | i,
    ]);
    final matrix = NativeQ4Matrix.fromGgufBlocks(raw, 2, 32);
    try {
      raw.fillRange(0, raw.length, 0);
      for (var r = 0; r < 2; r++) {
        final scale = r == 0 ? 1 / 128 : -1 / 128;
        for (var c = 0; c < 32; c++) {
          final nibble = r == 0 && c < 16 ? 15 - c : c % 16;
          expect(matrix.weightAt(r, c), scale * (nibble - 8));
        }
      }
      final x = Float32List.fromList([for (var i = 0; i < 32; i++) i / 32]);
      final y = Float32List(2);
      matrix.apply(x, y, 1, int8Inputs: false);
      for (var r = 0; r < 2; r++) {
        var ref = 0.0;
        for (var c = 0; c < 32; c++) {
          ref += matrix.weightAt(r, c) * x[c];
        }
        expect(y[r], closeTo(ref, 1e-7));
      }
    } finally {
      matrix.free();
    }
    matrix.free();
    for (final shape in [(0, 32), (1, 31), (1, 32)]) {
      expect(
        () => NativeQ4Matrix.fromGgufBlocks(Uint8List(0), shape.$1, shape.$2),
        throwsArgumentError,
      );
    }
  }, skip: supported ? false : 'native kernels unavailable');

  final real = Platform.environment['DIANOIA_AUDIO_Q4_MMPROJ'];
  final realOracle = Platform.environment['DIANOIA_AUDIO_ENCODER_ORACLE'];
  test(
    'official Q4_0 mmproj matches dequantized upstream Torch graph',
    () async {
      final fixture =
          jsonDecode(File(realOracle!).readAsStringSync())
              as Map<String, dynamic>;
      final digest = (await sha256.bind(File(real!).openRead()).first)
          .toString();
      expect(digest, Lfm3AudioQ4Source.audio.sha256);
      expect(fixture['gguf_sha256'], digest);
      final encoder = Lfm3AudioEncoder.load(real);
      try {
        expect(encoder.config.hiddenSize, 768);
        expect(encoder.config.intermediateSize, 1792);
        compare(encoder, fixture, tolerance: 2e-4);
      } finally {
        encoder.close();
      }
    },
    skip: supported && real != null && realOracle != null
        ? false
        : 'set DIANOIA_AUDIO_Q4_MMPROJ and DIANOIA_AUDIO_ENCODER_ORACLE',
  );
}
