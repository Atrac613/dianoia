/// A control vector on the CPU and the GPU of one device, shared by
/// `tool/control_vector_bench.dart` on a Mac and the device bench app
/// (tool/device_bench), so their records compare one to one.
///
/// Per backend it records:
///
/// * the greedy reply token ids to each prompt without and with the vector;
/// * for refusal prompts, only how many replies open with a refusal,
///   without and with the vector (no ids, no text);
/// * decoding and prefill rates without and with the vector, alternating,
///   on all prompts joined into one;
/// * a self-check needing no reference: in project mode the largest
///   |x · u| / |x| over the steered layers' captured rows (about 0 at
///   scale 1), and
///   whether the layer before the steered range is unchanged from the run
///   without the vector;
/// * whether setting and then clearing the vector leaves the logits
///   bit-identical to never having set it.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'chat_template.dart';
import 'control_vector.dart';
import 'control_vector_eval.dart';
import 'gpu.dart';
import 'model.dart';
import 'quant.dart';
import 'tokenizer.dart';

/// Runs the checks on each of [backends] (`false` the CPU kernels, `true`
/// the GPU) for the bundle in [bundle] with [weights], and returns the
/// record; [onRecord] receives it after every step.
Future<Map<String, Object?>> benchmarkControlVector({
  required String bundle,
  required WeightFormat weights,
  required ControlVector vector,
  required List<String> prompts,
  List<String> refusalPrompts = const [],
  List<bool> backends = const [false, true],
  int maxNewTokens = 24,
  int refusalTokens = 8,
  int rateTokens = 128,
  int rateRounds = 3,
  int contextLength = 1024,
  void Function(String) log = _ignore,
  void Function(Map<String, Object?>) onRecord = _ignoreRecord,
}) async {
  final tokenizer = BpeTokenizer.load(bundle);
  List<int> idsOf(String prompt) => tokenizer.encode(
    renderChat([
      {'role': 'user', 'content': prompt},
    ], enableThinking: false),
  );
  final record = <String, Object?>{
    'os': Platform.operatingSystemVersion,
    'weights': weights.name,
    'vector': {
      'mode': vector.mode.name,
      'scale': vector.scale,
      'first_layer': vector.firstLayer,
      'last_layer': vector.lastLayer,
      'fingerprint': vector.fingerprint,
    },
    'max_new_tokens': maxNewTokens,
  };
  for (final gpu in backends) {
    final name = gpu ? 'gpu' : 'cpu';
    final model = await TextModel.load(
      bundle,
      maxSeq: contextLength,
      weights: weights,
      q8CachePath: '$bundle/lfm3-${weights.name}.cache',
      gpu: gpu,
    );
    final result = <String, Object?>{};
    record[name] = result;
    try {
      if (gpu) record['device'] = GpuForward.deviceFacts;
      final steered = [
        for (var l = 0; l < model.config.numLayers; l++)
          if (vector.steers(l)) l,
      ];
      result['steered_layers'] = steered;

      Future<List<int>> reply(List<int> ids, ControlVector? v, int n) async {
        await model.reset();
        model.setControlVector(v);
        await model.forward(ids);
        final out = <int>[];
        while (out.length < n) {
          final next = _argmax(model.logits);
          if (next == model.config.eosTokenId) break;
          out.add(next);
          if (out.length < n) await model.step(next);
        }
        return out;
      }

      // Greedy replies.
      final replies = <Map<String, Object?>>[];
      result['replies'] = replies;
      for (final (i, prompt) in prompts.indexed) {
        final ids = idsOf(prompt);
        replies.add({
          'without': await reply(ids, null, maxNewTokens),
          'with': await reply(ids, vector, maxNewTokens),
        });
        log('$name: reply ${i + 1}/${prompts.length}');
      }
      onRecord(record);

      // Refusals: counts only.
      if (refusalPrompts.isNotEmpty) {
        var without = 0, withVector = 0;
        for (final prompt in refusalPrompts) {
          final ids = idsOf(prompt);
          final plain = tokenizer.decode(await reply(ids, null, refusalTokens));
          final changed = tokenizer.decode(
            await reply(ids, vector, refusalTokens),
          );
          if (opensWithRefusal(plain)) without++;
          if (opensWithRefusal(changed)) withVector++;
        }
        result['refusals'] = {
          'prompts': refusalPrompts.length,
          'without': without,
          'with': withVector,
        };
        log(
          '$name: refusals $without -> $withVector of '
          '${refusalPrompts.length}',
        );
        onRecord(record);
      }

      // Rates, alternating without and with the vector.
      // All prompts as one user turn: a prompt of a few hundred tokens.
      final rateIds = idsOf(prompts.join(' '));
      final rates = <String, List<Map<String, double>>>{
        'without': [],
        'with': [],
      };
      for (var round = 0; round < rateRounds; round++) {
        for (final v in [null, vector]) {
          await model.reset();
          model.setControlVector(v);
          final sw = Stopwatch()..start();
          await model.forward(rateIds);
          final prefill = rateIds.length / (sw.elapsedMicroseconds / 1e6);
          sw
            ..reset()
            ..start();
          var token = _argmax(model.logits);
          for (var t = 0; t < rateTokens; t++) {
            await model.step(token);
            token = _argmax(model.logits);
          }
          rates[v == null ? 'without' : 'with']!.add({
            'prefill_tok_per_s': prefill,
            'decode_tok_per_s': rateTokens / (sw.elapsedMicroseconds / 1e6),
          });
        }
      }
      result['rates'] = {
        'prompt_tokens': rateIds.length,
        'decode_tokens': rateTokens,
        ...rates,
      };
      log(
        '$name: decode tok/s without '
        '${[for (final r in rates['without']!) r['decode_tok_per_s']!.toStringAsFixed(1)]}'
        ' with ${[for (final r in rates['with']!) r['decode_tok_per_s']!.toStringAsFixed(1)]}',
      );
      onRecord(record);

      // Self-check.
      final h = model.config.hiddenSize;
      Future<List<Float32List>> capture(ControlVector? v) async {
        await model.reset();
        model.setControlVector(v);
        final slots = [
          for (var s = 0; s <= model.config.numLayers; s++)
            Float32List(rateIds.length * h),
        ];
        await model.forward(
          rateIds,
          capture: (slot, first, rows, count) =>
              slots[slot].setRange(first * h, (first + count) * h, rows),
        );
        return slots;
      }

      final plain = await capture(null);
      final steeredSlots = await capture(vector);
      final check = <String, Object?>{};
      if (vector.mode == ControlMode.project) {
        var worst = 0.0;
        for (final l in steered) {
          final u = vector.appliedVector(l)!;
          final rows = steeredSlots[l + 1];
          for (var t = 0; t < rateIds.length; t++) {
            var dot = 0.0, norm = 0.0;
            for (var j = 0; j < h; j++) {
              final x = rows[t * h + j];
              dot += x * u[j];
              norm += x * x;
            }
            if (norm > 0) {
              worst = math.max(worst, dot.abs() / math.sqrt(norm));
            }
          }
        }
        // Scale s leaves (1 - s) of each component: about 0 only at 1.
        check['max_component_along_u'] = worst;
      }
      if (steered.isNotEmpty && steered.first > 0) {
        final before = steered.first; // slot of the output of layer first-1
        var diff = 0.0;
        for (var i = 0; i < plain[before].length; i++) {
          diff = math.max(
            diff,
            (plain[before][i] - steeredSlots[before][i]).abs(),
          );
        }
        check['unsteered_layer'] = before - 1;
        check['unsteered_identical'] = diff == 0;
        check['unsteered_max_abs_diff'] = diff;
      }

      // Setting and clearing.
      await model.reset();
      model.setControlVector(null);
      await model.forward(rateIds);
      final fresh = Float32List.fromList(model.logits);
      await model.reset();
      model.setControlVector(vector);
      await model.forward(rateIds);
      await model.reset();
      model.setControlVector(null);
      await model.forward(rateIds);
      var identical = true;
      for (var i = 0; i < fresh.length; i++) {
        if (fresh[i] != model.logits[i]) {
          identical = false;
          break;
        }
      }
      check['set_then_clear_identical'] = identical;
      result['self_check'] = check;
      log('$name: self-check $check');
      onRecord(record);
    } finally {
      await model.close();
    }
  }
  record['finished'] = true;
  onRecord(record);
  return record;
}

void _ignore(String _) {}

void _ignoreRecord(Map<String, Object?> _) {}

int _argmax(Float32List v) {
  var best = 0;
  for (var i = 1; i < v.length; i++) {
    if (v[i] > v[best]) best = i;
  }
  return best;
}
