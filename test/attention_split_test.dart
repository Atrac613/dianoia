import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

Float32List _random(math.Random r, int length) => Float32List.fromList([
  for (var i = 0; i < length; i++) r.nextDouble() * 2 - 1,
]);

/// The split C attention (lab_attention_split) against the one-task-per-head
/// kernel (lab_attention) on the same cache: equal up to summation order.
void main() {
  group('native split attention', () {
    setUpAll(() {
      if (!NativeKernels.isLoaded) NativeKernels.load(null, threads: 3);
    });
    tearDown(() {
      nativeAttentionSplitFrom = 512;
      nativeAttentionBlock = 256;
    });

    for (final (name, heads, kvHeads, n, position, window, gated, block) in [
      ('decode, GQA 2, ungated', 16, 8, 1, 1500, 0, false, 256),
      ('decode, GQA 4, gated', 16, 4, 1, 900, 0, true, 128),
      ('chunk of 5, uneven blocks', 16, 8, 5, 700, 0, false, 100),
      ('sliding window', 8, 2, 3, 1200, 256, true, 64),
      ('one block', 4, 4, 2, 40, 0, false, 512),
    ]) {
      test(name, () {
        const hd = 128;
        final r = math.Random(7);
        final kvDim = kvHeads * hd;
        final rows = position + n;
        final q = _random(r, n * heads * hd);
        final keys = _random(r, rows * kvDim);
        final values = _random(r, rows * kvDim);
        final gate = gated ? _random(r, n * heads) : null;
        Float32List run(int from) {
          nativeAttentionSplitFrom = from;
          nativeAttentionBlock = block;
          final out = Float32List(n * heads * hd);
          nativeAttention(
            q,
            keys,
            values,
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

        final reference = run(0);
        final split = run(1);
        var worst = 0.0;
        for (var i = 0; i < reference.length; i++) {
          worst = math.max(worst, (reference[i] - split[i]).abs());
        }
        expect(worst, lessThan(1e-5));
      });
    }
  }, skip: NativeKernels.isAvailable ? false : 'native kernels unavailable');
}
