// The half KV cache (KvCacheFormat.f16): the float32-to-half rounding, the
// C attention kernels that read halves, and a tiny synthetic Qwen3 model
// run with each cache format on the CPU and on the GPU.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/gpu_host.dart';
import 'support/synthetic_qwen3.dart';

Float32List _random(math.Random r, int length) => Float32List.fromList([
  for (var i = 0; i < length; i++) r.nextDouble() * 2 - 1,
]);

double _maxDiff(List<double> a, List<double> b) {
  var worst = 0.0;
  for (var i = 0; i < a.length; i++) {
    worst = math.max(worst, (a[i] - b[i]).abs());
  }
  return worst;
}

void main() {
  final native = NativeKernels.isAvailable;
  setUpAll(() {
    if (native && !NativeKernels.isLoaded) NativeKernels.load(null);
  });

  group('half rounding', () {
    test('rounds to nearest, ties to even', () {
      final src = Float32List.fromList([
        0,
        1,
        -2,
        65504,
        1 + 1 / 2048, // halfway between 1 and the next half: down to even
        1 + 3 / 2048, // halfway again: up to even
        1 / 16384, // the smallest normal half
        1 / 16777216, // the smallest subnormal half
        1e6, // clamped to the largest finite half
        -1e6,
      ]);
      final halves = Uint16List(src.length);
      nativeF32ToF16(src, 0, halves, 0, src.length);
      expect(halves, [
        0x0000,
        0x3c00,
        0xc000,
        0x7bff,
        0x3c00,
        0x3c02,
        0x0400,
        0x0001,
        0x7bff,
        0xfbff,
      ]);
      final back = Float32List(src.length);
      nativeF16ToF32(halves, back);
      expect(back.sublist(0, 4), src.sublist(0, 4));
      expect(back[6], src[6]);
      expect(back[8], 65504);
    });

    test('converts a range with an odd tail', () {
      final src = _random(math.Random(3), 23);
      final halves = Uint16List(30);
      nativeF32ToF16(src, 2, halves, 5, 21);
      final back = Float32List(30);
      nativeF16ToF32(halves, back);
      expect(back.sublist(0, 5), everyElement(0));
      for (var i = 0; i < 21; i++) {
        expect(back[5 + i], closeTo(src[2 + i], 1e-3));
      }
    });
  }, skip: native ? false : 'native kernels unavailable');

  group('half attention', () {
    tearDown(() {
      nativeAttentionSplitFrom = 512;
      nativeAttentionBlock = 256;
    });

    // The half kernels against the float32 one over the same values, the
    // halves widened: equal up to summation order.
    for (final (name, heads, kvHeads, n, position, window, gated, split) in [
      ('decode, GQA 2', 16, 8, 1, 1500, 0, false, false),
      ('decode, GQA 2, split', 16, 8, 1, 1500, 0, false, true),
      ('chunk of 5, gated', 16, 4, 5, 700, 0, true, false),
      ('chunk of 5, gated, split', 16, 4, 5, 700, 0, true, true),
      ('sliding window, split', 8, 2, 3, 1200, 256, false, true),
      ('a prompt of 600', 4, 2, 600, 0, 0, false, false),
    ]) {
      test(name, () {
        const hd = 128;
        final r = math.Random(11);
        final kvDim = kvHeads * hd;
        final rows = (position + n) * kvDim;
        final q = _random(r, n * heads * hd);
        final gate = gated ? _random(r, n * heads) : null;
        final halfKeys = Uint16List(rows), halfValues = Uint16List(rows);
        nativeF32ToF16(_random(r, rows), 0, halfKeys, 0, rows);
        nativeF32ToF16(_random(r, rows), 0, halfValues, 0, rows);
        final keys = Float32List(rows), values = Float32List(rows);
        nativeF16ToF32(halfKeys, keys);
        nativeF16ToF32(halfValues, values);
        Float32List run(TypedData k, TypedData v) {
          final out = Float32List(n * heads * hd);
          nativeAttention(
            q,
            k,
            v,
            gate,
            out,
            n: n,
            position: position,
            heads: heads,
            kvHeads: kvHeads,
            headDim: hd,
            window: window,
          );
          return out;
        }

        nativeAttentionSplitFrom = 0;
        final reference = run(keys, values);
        nativeAttentionSplitFrom = split ? 1 : 0;
        nativeAttentionBlock = 100;
        expect(_maxDiff(reference, run(halfKeys, halfValues)), lessThan(1e-5));
      });
    }
  }, skip: native ? false : 'native kernels unavailable');

  group('synthetic Qwen3', () {
    late Directory root;
    late String bundle;

    setUpAll(() async {
      root = Directory.systemTemp.createTempSync('dianoia-kv-cache-');
      writeSyntheticQwen3('${root.path}/hf');
      bundle = '${root.path}/bundle';
      await buildTextBundle(
        '${root.path}/hf',
        bundle,
        stagingDir: '${root.path}/staging',
        weights: WeightFormat.q8,
      );
    });
    tearDownAll(() => root.deleteSync(recursive: true));

    final prompt = [5, 9, 12, 7, 30, 31, 2, 8, 40, 41, 42, 17, 3, 60, 61];

    Future<TextModel> load(bool gpu, KvCacheFormat? kv) => TextModel.load(
      bundle,
      maxSeq: 64,
      maxChunk: 4,
      weights: WeightFormat.q8,
      q8CachePath: '$bundle/qwen3-q8.cache',
      gpu: gpu,
      kvCache: kv,
    );

    for (final gpu in [false, true]) {
      final backend = gpu ? 'GPU' : 'CPU';
      final skip = !native
          ? 'native kernels unavailable'
          : gpu && !hostHasGpu
          ? noGpuSkip
          : null;

      test('$backend: half logits stay close to float32 ones', () async {
        if (gpu && !gpuStarts) {
          markTestSkipped(gpuDoesNotStartSkip);
          return;
        }
        final logits = <KvCacheFormat, Float32List>{};
        for (final kv in KvCacheFormat.values) {
          final model = await load(gpu, kv);
          try {
            await model.forward(prompt);
            logits[kv] = Float32List.fromList(model.logits);
          } finally {
            await model.close();
          }
        }
        final f32 = logits[KvCacheFormat.f32]!;
        expect(f32.any((v) => v != 0), isTrue);
        expect(_maxDiff(f32, logits[KvCacheFormat.f16]!), lessThan(0.05));
      }, skip: skip);

      test('$backend: a half cache saves and restores its state', () async {
        if (gpu && !gpuStarts) {
          markTestSkipped(gpuDoesNotStartSkip);
          return;
        }
        final path = '${root.path}/state-$backend';
        final model = await load(gpu, KvCacheFormat.f16);
        final Float32List expected;
        try {
          await model.forward(prompt.sublist(0, 12));
          await model.saveState(path);
          await model.forward(prompt.sublist(12));
          expected = Float32List.fromList(model.logits);
        } finally {
          await model.close();
        }
        final restored = await load(gpu, KvCacheFormat.f16);
        try {
          expect(await restored.loadState(path), isTrue);
          await restored.forward(prompt.sublist(12));
          expect(restored.logits, orderedEquals(expected));
        } finally {
          await restored.close();
        }
        // A float32 cache refuses the half state.
        final other = await load(gpu, KvCacheFormat.f32);
        try {
          expect(await other.loadState(path), isFalse);
        } finally {
          await other.close();
        }
      }, skip: skip);
    }

    test('Qwen3 defaults to a half cache, LFM3 to float32', () async {
      expect(
        KvCacheFormat.defaultFor(TextModelConfig.load(bundle)),
        KvCacheFormat.f16,
      );
      final vision =
          jsonDecode(
                File('test/fixtures/vision/config.json').readAsStringSync(),
              )
              as Map<String, Object?>;
      final lfm3 = TextModelConfig.fromJson(
        vision['text_config'] as Map<String, Object?>,
      );
      expect(KvCacheFormat.defaultFor(lfm3), KvCacheFormat.f32);
      // TextModel.load resolves the default: half on the CPU with the
      // package's kernels.
      final model = await load(false, null);
      try {
        expect(model.kvCache, KvCacheFormat.f16);
      } finally {
        await model.close();
      }
    }, skip: native ? false : 'native kernels unavailable');

    test('the Dart reference attention reads a half cache', () async {
      final kernels = NativeKernels.instance;
      final model = await load(false, KvCacheFormat.f16);
      final Float32List viaC;
      try {
        await model.forward(prompt);
        viaC = Float32List.fromList(model.logits);
      } finally {
        await model.close();
      }
      NativeKernels.load(
        null,
        threads: kernels.threads,
        int8Inputs: kernels.int8Inputs,
        attention: false,
      );
      try {
        final reference = await load(false, KvCacheFormat.f16);
        try {
          await reference.forward(prompt);
          expect(_maxDiff(viaC, reference.logits), lessThan(1e-3));
        } finally {
          await reference.close();
        }
      } finally {
        NativeKernels.load(
          null,
          threads: kernels.threads,
          int8Inputs: kernels.int8Inputs,
        );
      }
    }, skip: native ? false : 'native kernels unavailable');
  });
}
