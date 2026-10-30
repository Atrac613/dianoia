/// Fixed-context sustained experiment. Kept outside the inference library.
library;

import 'dart:io';
import 'dart:typed_data';
import 'package:dianoia/dianoia.dart';

Future<Map<String, Object?>> runGpuSustained(
  Lfm3Model model,
  List<int> ids,
  int seconds,
  void Function(String) log,
  void Function(Map<String, Object?>) update,
) async {
  if (seconds < 1 || seconds > 3600) {
    throw ArgumentError('duration must be 1..3600 seconds');
  }
  const steps = 32;
  int argmax(Float32List x) {
    var best = 0;
    for (var i = 1; i < x.length; i++) {
      if (x[i] > x[best]) best = i;
    }
    return best;
  }

  String? clock() {
    try {
      return File(
        '/sys/devices/platform/1f000000.mali/cur_freq',
      ).readAsStringSync().trim();
    } on FileSystemException {
      return null;
    }
  }

  // A frozen old-int8-GEMM reference and continuation, separate from timing.
  GpuForward.decodeRows = 0;
  GpuForward.setQ8Vector(false);
  await model.reset();
  await model.forward(ids);
  final tokens = <int>[];
  for (var i = 0; i < steps; i++) {
    tokens.add(argmax(model.logits));
    await model.step(tokens.last);
  }
  final reference = Uint32List.fromList(
    model.logits.buffer.asUint32List(
      model.logits.offsetInBytes,
      model.logits.length,
    ),
  );
  final samples = <Map<String, Object?>>[];
  final record = <String, Object?>{
    'requested_seconds': seconds,
    'prompt_tokens': ids.length,
    'decode_steps_per_block': steps,
    'context_end': ids.length + steps,
    'workload':
        'reset/prefill fixed prompt, then fixed continuation; no intentional idle',
    'profiling': 'alternating ABBA cycles with/without per-dispatch timestamps',
    'samples': samples,
    'started_utc': DateTime.now().toUtc().toIso8601String(),
  };
  final elapsed = Stopwatch()..start();
  var nextReport = 0.0, cycle = 0;
  try {
    GpuForward.decodeRows = 4;
    GpuForward.setQ8Workgroup(128);
    while (elapsed.elapsedMicroseconds / 1e6 < seconds) {
      final profiled = cycle.isEven;
      GpuForward.profiling = profiled;
      for (final automatic in [false, true, true, false]) {
        if (automatic) {
          GpuForward.restoreQ8VectorDefault();
        } else {
          GpuForward.setQ8Vector(false);
        }
        await model.reset();
        final prefill = Stopwatch()..start();
        await model.forward(ids);
        final prefillMs = prefill.elapsedMicroseconds / 1000;
        GpuForward.takeProfile();
        model.gpu!.clearStats();
        final clockBefore = clock();
        final wall = Stopwatch()..start();
        for (final token in tokens) {
          await model.step(token);
        }
        final wallMs = wall.elapsedMicroseconds / 1000 / steps;
        final profile = GpuForward.takeProfile();
        double ms(String kind) => (profile[kind]?.$1 ?? 0) / steps;
        int count(String kind) => profile[kind]?.$2 ?? 0;
        final total = profiled
            ? profile.values.fold(0.0, (s, p) => s + p.$1) / steps
            : model.gpu!.gpuMicros / 1000 / steps;
        final q4 = ms('gemv_q4_i8');
        final q8 = ms('gemv_q8_i8') + ms('gemv_q8_i8_vec');
        // Compare final logits every block, outside its measured wall time.
        final actual = model.logits;
        final bits = actual.buffer.asUint32List(
          actual.offsetInBytes,
          actual.length,
        );
        var differing = 0, nonFinite = 0;
        for (var j = 0; j < bits.length; j++) {
          if (bits[j] != reference[j]) differing++;
          if (!actual[j].isFinite) nonFinite++;
        }
        samples.add({
          'cycle': cycle,
          'automatic': automatic,
          'profiled': profiled,
          'elapsed_s': elapsed.elapsedMicroseconds / 1e6,
          'gpu_ms_per_token': total,
          'q4_ms_per_token': profiled ? q4 : null,
          'q8_ms_per_token': profiled ? q8 : null,
          'other_ms_per_token': profiled ? total - q4 - q8 : null,
          'wall_ms_per_token': wallMs,
          'prefill_ms': prefillMs,
          'q4_dispatches_per_token': profiled
              ? count('gemv_q4_i8') / steps
              : null,
          'vector_head_dispatches_per_token': profiled
              ? count('gemv_q8_i8_vec') / steps
              : null,
          'scalar_head_dispatches_per_token': profiled
              ? count('gemv_q8_i8') / steps
              : null,
          'total_dispatches_per_token': profiled
              ? profile.values.fold(0, (s, p) => s + p.$2) / steps
              : null,
          'encoded_graph_ops_per_token': model.gpu!.dispatches / steps,
          'gpu_clock_raw_before': clockBefore,
          'gpu_clock_raw_after': clock(),
          'differing_final_logits': differing,
          'non_finite': nonFinite,
        });
        if (differing != 0 || nonFinite != 0 || (profiled && q4 <= 0)) {
          record['failed'] = true;
          update(record);
          throw StateError('sustained correctness/profile failure');
        }
      }
      cycle++;
      record['elapsed_s'] = elapsed.elapsedMicroseconds / 1e6;
      if ((record['elapsed_s'] as double) >= nextReport) {
        log(
          'Sustained ${elapsed.elapsed.inSeconds}s: ${samples.length} blocks, last '
          '${(samples.last['gpu_ms_per_token'] as double).toStringAsFixed(2)} GPU ms/token, profiled=$profiled',
        );
        update(record);
        nextReport = (record['elapsed_s'] as double) + 30;
      }
    }
    record['elapsed_s'] = elapsed.elapsedMicroseconds / 1e6;
    record['finished'] = true;
    record['finished_utc'] = DateTime.now().toUtc().toIso8601String();
    update(record);
    return record;
  } finally {
    GpuForward.profiling = false;
    GpuForward.decodeRows = 4;
    GpuForward.restoreQ8VectorDefault();
  }
}
