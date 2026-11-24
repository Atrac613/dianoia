/// The device benchmark's `qwen3` mode: Qwen3 from Hugging Face to a reply,
/// on the phone alone. ModelStore downloads Qwen's Q8_0 GGUF and imports it
/// (unless installed), then the model answers fixed prompts on the GPU and
/// on the CPU: greedy, a tool call, a second turn that reuses the first,
/// and sampled thinking with a fixed seed. Free of Flutter, so the same
/// checks run on a Mac (run_ios_qwen3.sh) for comparison.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';

const _weatherTool = {
  'type': 'function',
  'function': {
    'name': 'get_weather',
    'description': 'Get the current weather for a city.',
    'parameters': {
      'type': 'object',
      'properties': {
        'city': {'type': 'string', 'description': 'City name'},
      },
      'required': ['city'],
    },
  },
};

/// Installs `config['spec']` (`0.6b`, the default, or `1.7b`) into
/// [storeDir] unless installed, runs the checks on `config['backends']`
/// (default GPU, then CPU), and fills [record], calling [save] after each
/// step.
Future<void> qwen3Check(
  Map<String, Object?> config,
  String storeDir,
  Map<String, Object?> record,
  void Function(String) log,
  void Function() save, {
  int Function()? footprint,
}) async {
  final spec = switch (config['spec'] ?? '0.6b') {
    '1.7b' => ModelSpec.qwen3_1_7bQ8,
    _ => ModelSpec.qwen3_0_6bQ8,
  };
  record['spec'] = spec.id;
  final store = ModelStore(storeDir);
  if (config['reinstall'] == true) await store.remove(spec);
  record['install'] = await _install(store, spec, log);
  final bundle = store.bundlePath(spec);
  final files = {
    for (final f in Directory(bundle).listSync().whereType<File>())
      f.uri.pathSegments.last: f.lengthSync(),
  };
  (record['install'] as Map)['bundle_files'] = files;
  save();

  // Room for thinking, which took 1,699 tokens on the Mac for 1.7B.
  final contextLength = (config['context_length'] as num?)?.toInt() ?? 3072;
  if (config['verify_cost'] == true) {
    // Speculative decoding: 0.6B drafts, 1.7B verifies. Both are needed.
    final costs = <String, Object?>{};
    record['verify_cost'] = costs;
    for (final s in [ModelSpec.qwen3_0_6bQ8, ModelSpec.qwen3_1_7bQ8]) {
      if (s != spec) {
        record['install_${s.name}'] = await _install(store, s, log);
      }
      costs[s.id] = await qwen3VerifyCost(store.bundlePath(s), log);
      save();
    }
    return;
  }
  if (config['gpu_profile'] == true) {
    record['gpu_profile'] = await qwen3GpuProfile(
      bundle,
      log,
      contextLength: contextLength,
      shapes: config['gpu_shapes'] == true,
    );
    save();
    return;
  }
  final backends = (config['backends'] as List? ?? ['gpu', 'cpu']).map(
    (b) => InferenceBackend.values.byName(b as String),
  );
  for (final backend in backends) {
    final result = <String, Object?>{};
    record[backend.name] = result;
    final sw = Stopwatch()..start();
    final model = await LocalModel.load(
      bundle,
      backend: backend,
      contextLength: contextLength,
    );
    result['load_s'] = sw.elapsedMilliseconds / 1000;
    log('${backend.name}: loaded in ${result['load_s']} s');
    try {
      Map<String, Object?> describe(ChatTurn t) => {
        'content': t.reply.content,
        'reasoning_chars': t.reply.reasoning?.length,
        'tool_calls': [
          for (final c in t.reply.toolCalls)
            {'name': c.name, 'arguments': c.arguments},
        ],
        'prompt_tokens': t.promptTokens,
        'reused_tokens': t.reusedTokens,
        'generated_tokens': t.generatedTokens,
        'prefill_s': t.prefillSeconds,
        // JSON has no NaN or Infinity: an empty reply decodes at 0.
        'decode_tok_per_s': t.decodeSeconds > 0
            ? t.generatedTokens / t.decodeSeconds
            : 0.0,
        if (footprint != null) 'footprint_mb': footprint() >> 20,
      };
      // Each prompt's reply is checked, so that a run needs no reading to
      // tell whether it passed; `passed` gathers every check.
      void report(String name, ChatTurn t, bool passed) {
        final d = describe(t)..['passed'] = passed;
        result[name] = d;
        record['passed'] = (record['passed'] as bool? ?? true) && passed;
        log(
          '  $name ${passed ? 'ok' : 'FAILED'}: ${t.generatedTokens} tokens at '
          '${(d['decode_tok_per_s'] as double).toStringAsFixed(1)} tok/s, '
          'reused ${t.reusedTokens}/${t.promptTokens}: '
          '${t.reply.toolCalls.isEmpty ? t.reply.content : d['tool_calls']}',
        );
        save();
      }

      final chat = model.chat(maxNewTokens: 64);
      final greedy = await chat.send({
        'role': 'user',
        'content': 'What is the capital of Japan? Answer in one sentence.',
      });
      report('greedy', greedy, greedy.reply.content.contains('Tokyo'));
      final second = await chat.send({
        'role': 'user',
        'content': 'And of France? Answer in one sentence.',
      });
      report(
        'second_turn',
        second,
        second.reply.content.contains('Paris') && second.reusedTokens > 0,
      );
      final tool = await model
          .chat(tools: [_weatherTool], maxNewTokens: 64)
          .send({
            'role': 'user',
            'content': "What's the weather like in Paris right now?",
          });
      report(
        'tool_call',
        tool,
        tool.reply.toolCalls.length == 1 &&
            tool.reply.toolCalls.single.name == 'get_weather' &&
            tool.reply.toolCalls.single.arguments['city'] == 'Paris',
      );
      if (backend != InferenceBackend.cpu || config['cpu_thinking'] == true) {
        final thinking = await model
            .chat(
              maxNewTokens: 3000,
              enableThinking: true,
              // Qwen3's thinking preset, seeded.
              sampling: const SamplingParams(
                temperature: 0.6,
                topK: 20,
                topP: 0.95,
                seed: 7,
              ),
            )
            .send({'role': 'user', 'content': 'What is 17 * 23?'});
        // The think block must close, then the answer follow.
        report(
          'thinking',
          thinking,
          thinking.reply.reasoning != null &&
              thinking.reply.content.contains('391'),
        );
      }
    } finally {
      await model.close();
    }
  }
}

/// Installs [spec] unless installed, logging progress; the record of it.
Future<Map<String, Object?>> _install(
  ModelStore store,
  ModelSpec spec,
  void Function(String) log,
) async {
  if (store.isInstalled(spec)) {
    log('${spec.name} already installed');
    return {'already_installed': true};
  }
  log('installing ${spec.name} (${spec.downloadBytes ~/ 1000000} MB)');
  final sw = Stopwatch()..start();
  double? downloaded;
  var lastLog = 0;
  await for (final p in store.install(spec)) {
    if (p.stage == InstallStage.downloading && p.bytes - lastLog >= 100000000) {
      lastLog = p.bytes;
      log('  ${p.bytes ~/ 1000000} / ${p.totalBytes ~/ 1000000} MB');
    }
    if (p.stage != InstallStage.downloading && downloaded == null) {
      downloaded = sw.elapsedMilliseconds / 1000;
      log('  downloaded in ${downloaded.toStringAsFixed(1)} s, importing');
    }
  }
  final total = sw.elapsedMilliseconds / 1000;
  log('  installed in ${total.toStringAsFixed(1)} s');
  return {
    'download_s': downloaded,
    'import_s': total - (downloaded ?? total),
    'total_s': total,
  };
}

/// What verifying drafted tokens would cost, on the GPU and the CPU: the
/// time of one pass over n tokens (n = 1 to 8) after a 200-token prompt,
/// rewound after each pass, and of one more LM head (a pass leaves the
/// last token's logits; verifying needs every token's). n = 1 is timed
/// again at the end, to show throttling.
Future<Map<String, Object?>> qwen3VerifyCost(
  String bundle,
  void Function(String) log, {
  int reps = 10,
}) async {
  final result = <String, Object?>{};
  // LocalModel.load's default; TextModel.load does not load the kernels.
  if (!NativeKernels.isLoaded) NativeKernels.load(null, threads: 4);
  final config = TextModelConfig.load(bundle);
  final cache = '$bundle/${quantizedCacheName(config, WeightFormat.q8)}';
  for (final gpu in [true, false]) {
    final model = await TextModel.load(
      bundle,
      maxSeq: 1024,
      weights: WeightFormat.q8,
      q8CachePath: cache,
      gpu: gpu,
    );
    try {
      final prompt = [for (var i = 0; i < 200; i++) 1000 + 7 * i];
      await model.forward(prompt);
      Future<double> time(int n) async {
        final tokens = [for (var i = 0; i < n; i++) 5000 + i];
        final sw = Stopwatch();
        for (var r = -2; r < reps; r++) {
          if (r == 0) sw.reset();
          sw.start();
          await model.forward(tokens);
          sw.stop();
          await model.restorePromptPrefix([...prompt, 0]);
        }
        return sw.elapsedMicroseconds / 1000 / reps;
      }

      final ms = <String, double>{};
      for (final n in [1, 2, 3, 4, 5, 6, 8]) {
        ms['$n'] = await time(n);
      }
      ms['1_again'] = await time(1);
      final residual = Float32List(config.hiddenSize)
        ..fillRange(0, config.hiddenSize, 0.01);
      final sw = Stopwatch()..start();
      for (var r = 0; r < reps; r++) {
        await model.logitsOf(residual);
      }
      final backend = gpu ? 'gpu' : 'cpu';
      result[backend] = {
        'pass_ms_by_tokens': ms,
        'lm_head_ms': sw.elapsedMicroseconds / 1000 / reps,
      };
      log(
        '$backend ${config.numLayers} layers: $ms, lm head ${result[backend]}',
      );
    } finally {
      await model.close();
    }
  }
  return result;
}

/// The process's memory by kind, in MB, where Linux reports it
/// (/proc/self/status). RssAnon is the heap; RssFile holds the mapped
/// weights, which the OS can drop, but on Android also the GPU driver's
/// allocations (Mali maps them from a device file), which it cannot.
/// Empty elsewhere.
Map<String, int> memoryBreakdown() {
  if (!Platform.isAndroid && !Platform.isLinux) return const {};
  final status = File('/proc/self/status').readAsStringSync();
  return {
    for (final key in ['VmRSS', 'RssAnon', 'RssFile', 'RssShmem'])
      if (RegExp('$key:\\s*(\\d+)').firstMatch(status)?[1] case final kb?)
        key: int.parse(kb) >> 10,
  };
}

/// Where a Qwen3 decode step's time and memory go on the GPU, with the
/// bundle loaded for [contextLength] tokens as [qwen3Check] loads it: 32
/// greedy steps split into the step and the argmax after it, the GPU's
/// time and Dart's encoding, then 32 more timed per kernel kind (Vulkan
/// times every dispatch), and the memory by kind before loading, after it,
/// while decoding and after closing.
///
/// With [shapes], it then times every kind of decode matrix across its
/// layers in each layout (GpuForward.benchmarkDecodeShapes).
Future<Map<String, Object?>> qwen3GpuProfile(
  String bundle,
  void Function(String) log, {
  required int contextLength,
  bool shapes = false,
}) async {
  final record = <String, Object?>{'memory_before': memoryBreakdown()};
  final cache =
      '$bundle/${quantizedCacheName(TextModelConfig.load(bundle), WeightFormat.q8)}';
  final model = await TextModel.load(
    bundle,
    maxSeq: contextLength,
    weights: WeightFormat.q8,
    q8CachePath: cache,
    gpu: true,
  );
  try {
    final gpu = model.gpu!;
    record['memory_loaded'] = memoryBreakdown();
    record['weight_segments'] = gpu.weightSegments;
    final prompt = BpeTokenizer.load(bundle).encode(
      renderQwen3Chat([
        {'role': 'user', 'content': 'Tell me about the history of Kyoto.'},
      ], enableThinking: false),
    );

    /// Feeds the prompt, then [steps] greedy tokens, timing the steps and
    /// the argmaxes apart.
    Future<({int stepMicros, int argmaxMicros})> decode(int steps) async {
      await model.reset();
      await model.forward(prompt);
      final step = Stopwatch(), argmax = Stopwatch();
      for (var i = 0; i < steps; i++) {
        argmax.start();
        final next = argmaxToken(model.logits);
        argmax.stop();
        step.start();
        await model.step(next);
        step.stop();
      }
      return (
        stepMicros: step.elapsedMicroseconds,
        argmaxMicros: argmax.elapsedMicroseconds,
      );
    }

    await decode(8); // warm-up
    gpu.clearStats();
    final timed = await decode(32);
    // The step includes copying the logits out of the GPU's buffer
    // (GpuForward.logits), so the argmax reads ordinary memory.
    record['decode'] = {
      'step_ms': timed.stepMicros / 1000 / 32,
      'argmax_ms': timed.argmaxMicros / 1000 / 32,
      // The prompt's chunk is one more GPU call: per call, not per token.
      'gpu_ms_per_call': gpu.gpuMicros / 1000 / gpu.calls,
      'encode_ms_per_call': gpu.encodeMicros / 1000 / gpu.calls,
      'dispatches_per_call': gpu.dispatches / gpu.calls,
    };
    log('decode: ${record['decode']}');
    record['memory_decoding'] = memoryBreakdown();

    GpuForward.profiling = true;
    try {
      await model.reset();
      await model.forward(prompt);
      GpuForward.takeProfile();
      for (var i = 0; i < 32; i++) {
        await model.step(argmaxToken(model.logits));
      }
      final kinds = GpuForward.takeProfile().entries.toList()
        ..sort((a, b) => b.value.$1.compareTo(a.value.$1));
      record['kernels'] = {
        for (final MapEntry(key: kind, value: (ms, count)) in kinds)
          kind: {'ms_per_token': ms / 32, 'dispatches_per_token': count / 32},
      };
      log(
        'kernels: ${kinds.map((e) => '${e.key} ${(e.value.$1 / 32).toStringAsFixed(2)} ms x${e.value.$2 ~/ 32}').join(', ')}',
      );
    } finally {
      // A process-wide switch: left on, every later pass is slower.
      GpuForward.profiling = false;
    }
    log('memory: ${record['memory_loaded']} -> ${record['memory_decoding']}');
    if (shapes) {
      record['shapes'] = gpu.benchmarkDecodeShapes(
        onSample: (s) => log(
          '${s['kind']} [${s['matrix_rows']}, ${s['matrix_cols']}] x${s['matrices']}, '
          'rows ${s['decode_rows']}, group ${s['q8_workgroup']}: '
          '${(s['gpu_ms_per_matrix'] as double).toStringAsFixed(3)} ms, '
          '${(s['gb_per_s'] as double).toStringAsFixed(1)} GB/s, '
          '${s['differing_outputs']} differing',
        ),
      );
    }
  } finally {
    await model.close();
  }
  record['memory_closed'] = memoryBreakdown();
  record['cache_mapped_after_close'] = MappedFile.isMapped(cache);
  return record;
}
