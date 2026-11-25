// GPU attention split by blocks of keys (attention_split, attention_merge)
// against the whole kernel, through a Qwen3 model deep enough into its cache
// to split. Point DIANOIA_QWEN3_DIR at a Qwen3 bundle holding its Q8 cache
// (qwen3-q8.cache, written by `dianoia pack`) to run it.
import 'dart:io';
import 'dart:math' as math;

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

void main() {
  final dir = Platform.environment['DIANOIA_QWEN3_DIR'];
  final cache = dir == null ? null : '$dir/qwen3-q8.cache';
  final skip = cache == null || !File(cache).existsSync()
      ? 'set DIANOIA_QWEN3_DIR to a Qwen3 bundle with qwen3-q8.cache'
      : null;

  test('split attention matches the whole kernel', () async {
    if (!NativeKernels.isLoaded) NativeKernels.load(null);
    final model = await TextModel.load(
      dir!,
      maxSeq: 1024,
      weights: WeightFormat.q8,
      q8CachePath: cache,
      gpu: true,
    );
    final from = GpuForward.attentionSplitFrom;
    final block = GpuForward.attentionBlock;
    try {
      const depth = 700;
      final prompt = [for (var i = 0; i < depth; i++) 1000 + (7 * i) % 9000];
      for (var i = 0; i < depth; i += 256) {
        await model.forward(
          prompt.sublist(i, math.min(depth, i + 256)),
          computeLogits: false,
        );
      }
      // One token, and a chunk whose tokens see different numbers of keys.
      Future<List<double>> logits(List<int> tokens) async {
        await model.restorePromptPrefix([...prompt, 0]);
        await model.forward(tokens);
        return List.of(model.logits);
      }

      for (final tokens in [
        [5000],
        [5000, 5001, 5002],
      ]) {
        GpuForward.attentionSplitFrom = 0;
        final whole = await logits(tokens);
        for (final b in [
          GpuForward.minAttentionBlock,
          GpuForward.maxAttentionBlock,
        ]) {
          GpuForward.attentionSplitFrom = 512;
          GpuForward.attentionBlock = b;
          final split = await logits(tokens);
          var worst = 0.0;
          for (var i = 0; i < whole.length; i++) {
            worst = math.max(worst, (whole[i] - split[i]).abs());
          }
          expect(worst, lessThan(1e-4), reason: '$tokens, blocks of $b');
        }
      }
    } finally {
      GpuForward.attentionSplitFrom = from;
      GpuForward.attentionBlock = block;
      await model.close();
    }
  }, skip: skip);
}
