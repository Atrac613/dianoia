import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/synthetic_vision.dart';

/// Softmax attention of every token over all [n], in double precision.
Float64List _reference(
  Float32List q,
  Float32List k,
  Float32List v,
  int n,
  int heads,
  int kvHeads,
  int d,
  double scale,
) {
  final out = Float64List(n * heads * d);
  final kv = kvHeads * d;
  final group = heads ~/ kvHeads;
  final scores = Float64List(n);
  for (var t = 0; t < n; t++) {
    for (var h = 0; h < heads; h++) {
      final o = (h ~/ group) * d;
      var maximum = double.negativeInfinity;
      for (var s = 0; s < n; s++) {
        var dot = 0.0;
        for (var j = 0; j < d; j++) {
          dot += q[(t * heads + h) * d + j] * k[s * kv + o + j];
        }
        scores[s] = dot * scale;
        maximum = math.max(maximum, scores[s]);
      }
      var total = 0.0;
      for (var s = 0; s < n; s++) {
        scores[s] = math.exp(scores[s] - maximum);
        total += scores[s];
      }
      for (var j = 0; j < d; j++) {
        var sum = 0.0;
        for (var s = 0; s < n; s++) {
          sum += scores[s] * v[s * kv + o + j];
        }
        out[(t * heads + h) * d + j] = sum / total;
      }
    }
  }
  return out;
}

void main() {
  group(
    'native full attention',
    _tests,
    skip: NativeKernels.isAvailable ? false : 'native kernels unavailable',
  );
}

void _tests() {
  setUpAll(() => NativeKernels.load(null, threads: 4, attention: true));

  final random = math.Random(3);
  Float32List values(int count) => Float32List.fromList([
    for (var i = 0; i < count; i++) random.nextDouble() * 4 - 2,
  ]);

  // Head dims that take the 16-wide loop, its 4-wide tail, or both; token
  // counts that do not divide into threads or runs.
  for (final (n, heads, kvHeads, d) in [
    (1, 2, 1, 16),
    (5, 4, 2, 8),
    (37, 4, 4, 12),
    (130, 16, 8, 64),
  ]) {
    test('matches double-precision attention: n=$n heads=$heads/$kvHeads '
        'd=$d', () {
      final q = values(n * heads * d);
      final k = values(n * kvHeads * d), v = values(n * kvHeads * d);
      final scale = 1 / math.sqrt(d);
      final out = Float32List(n * heads * d);
      nativeFullAttention(
        q,
        k,
        v,
        out,
        n: n,
        heads: heads,
        kvHeads: kvHeads,
        headDim: d,
        scale: scale,
      );
      final expected = _reference(q, k, v, n, heads, kvHeads, d, scale);
      for (var i = 0; i < out.length; i++) {
        expect(out[i], closeTo(expected[i], 1e-5 + 1e-5 * expected[i].abs()));
      }
    });
  }

  test('rejects shapes it cannot read', () {
    final x = Float32List(64);
    void call({int n = 4, int heads = 2, int kvHeads = 2, int d = 8}) =>
        nativeFullAttention(
          x,
          x,
          x,
          x,
          n: n,
          heads: heads,
          kvHeads: kvHeads,
          headDim: d,
          scale: 1,
        );
    expect(() => call(d: 6), throwsArgumentError);
    expect(() => call(heads: 3), throwsArgumentError);
    expect(() => call(n: 5), throwsArgumentError);
    expect(() => call(n: 0), throwsArgumentError);
  });

  test('the encoder still matches the upstream outputs', () {
    final oracle =
        jsonDecode(
              File(
                'test/fixtures/vision/encoder_oracle.json',
              ).readAsStringSync(),
            )
            as Map<String, Object?>;
    final directory = Directory.systemTemp.createTempSync(
      'dianoia_vision_native_attention',
    );
    addTearDown(() => directory.deleteSync(recursive: true));
    File(
      '${directory.path}/config.json',
    ).writeAsStringSync(jsonEncode(oracle['config']));
    writeSyntheticVisionWeights(
      '${directory.path}/model.safetensors',
      oracle['shapes'] as Map<String, Object?>,
    );
    final encoder = Lfm3VisionEncoder.load(directory.path);
    for (final item in oracle['cases'] as List) {
      final c = item as Map<String, Object?>;
      final grid = (c['grid'] as List).cast<int>();
      final stages = c['stages'] as Map<String, Object?>;
      encoder.encode(
        syntheticVisionImage(grid[0], grid[1]),
        capture: (stage, rows) {
          final expected = stages[stage] as List;
          expect(rows.length, expected.length, reason: '$grid $stage');
          for (var i = 0; i < rows.length; i++) {
            final value = (expected[i] as num).toDouble();
            expect(
              (rows[i] - value).abs() <= 2e-5 + 2e-5 * value.abs(),
              isTrue,
              reason: '$grid $stage index $i: ${rows[i]} vs $value',
            );
          }
        },
      );
    }
  });
}
