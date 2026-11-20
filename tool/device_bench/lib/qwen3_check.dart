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

  if (store.isInstalled(spec)) {
    record['install'] = <String, Object?>{'already_installed': true};
    log('${spec.name} already installed');
  } else {
    log('installing ${spec.name} (${spec.downloadBytes ~/ 1000000} MB)');
    final sw = Stopwatch()..start();
    double? downloaded;
    var lastLog = 0;
    await for (final p in store.install(spec)) {
      if (p.stage == InstallStage.downloading &&
          p.bytes - lastLog >= 100000000) {
        lastLog = p.bytes;
        log('  ${p.bytes ~/ 1000000} / ${p.totalBytes ~/ 1000000} MB');
      }
      if (p.stage != InstallStage.downloading && downloaded == null) {
        downloaded = sw.elapsedMilliseconds / 1000;
        log('  downloaded in ${downloaded.toStringAsFixed(1)} s, importing');
      }
    }
    final total = sw.elapsedMilliseconds / 1000;
    record['install'] = <String, Object?>{
      'download_s': downloaded,
      'import_s': total - (downloaded ?? total),
      'total_s': total,
    };
    log('  installed in ${total.toStringAsFixed(1)} s');
  }
  final bundle = store.bundlePath(spec);
  final files = {
    for (final f in Directory(bundle).listSync().whereType<File>())
      f.uri.pathSegments.last: f.lengthSync(),
  };
  (record['install'] as Map)['bundle_files'] = files;
  save();

  if (config['gpu_profile'] == true) {
    record['gpu_profile'] = await qwen3GpuProfile(bundle, log);
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
      // Room for thinking, which took 1,699 tokens on the Mac for 1.7B.
      contextLength: (config['context_length'] as num?)?.toInt() ?? 3072,
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

/// The process's memory by kind, in MB, where Linux reports it
/// (/proc/self/status): anonymous (heap, GPU copies), file-backed (mapped
/// weights, which the OS can drop) and shared. Empty elsewhere.
Map<String, int> memoryBreakdown() {
  if (!Platform.isAndroid && !Platform.isLinux) return const {};
  final status = File('/proc/self/status').readAsStringSync();
  return {
    for (final key in ['VmRSS', 'RssAnon', 'RssFile', 'RssShmem'])
      if (RegExp('$key:\\s*(\\d+)').firstMatch(status)?[1] case final kb?)
        key: int.parse(kb) >> 10,
  };
}

/// Where a Qwen3 decode step's time and memory go on the GPU: the bundle
/// loaded on the GPU, 32 greedy steps timed per kernel kind (Vulkan times
/// every dispatch), the same steps unprofiled split into GPU time and
/// Dart's encoding, and the memory by kind before and after loading.
Future<Map<String, Object?>> qwen3GpuProfile(
  String bundle,
  void Function(String) log,
) async {
  final record = <String, Object?>{'memory_before': memoryBreakdown()};
  final config = TextModelConfig.load(bundle);
  final model = await TextModel.load(
    bundle,
    maxSeq: 2048,
    weights: WeightFormat.q8,
    q8CachePath: '$bundle/${quantizedCacheName(config, WeightFormat.q8)}',
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
    Future<List<int>> decode(int steps) async {
      await model.reset();
      await model.forward(prompt);
      final tokens = <int>[];
      for (var i = 0; i < steps; i++) {
        final next = argmaxToken(model.logits);
        tokens.add(next);
        await model.step(next);
      }
      return tokens;
    }

    // Unprofiled: GPU time against the wall clock and Dart's encoding.
    await decode(8); // warm-up
    await model.reset();
    await model.forward(prompt);
    gpu.clearStats();
    final wall = Stopwatch()..start();
    for (var i = 0; i < 32; i++) {
      await model.step(argmaxToken(model.logits));
    }
    final steps = gpu.calls;
    record['decode'] = {
      'wall_ms_per_token': wall.elapsedMicroseconds / 1000 / 32,
      'gpu_ms_per_token': gpu.gpuMicros / 1000 / steps,
      'encode_ms_per_token': gpu.encodeMicros / 1000 / steps,
      'dispatches_per_token': gpu.dispatches / steps,
    };
    // The wall clock against the GPU's: the step alone, and reading the
    // logits it leaves in the GPU's host-visible memory (an argmax, and a
    // copy into ordinary memory followed by the same argmax).
    final stepWall = Stopwatch(), argmaxWall = Stopwatch();
    final copyWall = Stopwatch(), copiedArgmaxWall = Stopwatch();
    var next = argmaxToken(model.logits);
    for (var i = 0; i < 32; i++) {
      stepWall.start();
      await model.step(next);
      stepWall.stop();
      argmaxWall.start();
      next = argmaxToken(model.logits);
      argmaxWall.stop();
      copyWall.start();
      final copy = Float32List.fromList(model.logits);
      copyWall.stop();
      copiedArgmaxWall.start();
      argmaxToken(copy);
      copiedArgmaxWall.stop();
    }
    (record['decode'] as Map)
      ..['step_wall_ms'] = stepWall.elapsedMicroseconds / 1000 / 32
      ..['argmax_ms'] = argmaxWall.elapsedMicroseconds / 1000 / 32
      ..['copy_logits_ms'] = copyWall.elapsedMicroseconds / 1000 / 32
      ..['argmax_copied_ms'] = copiedArgmaxWall.elapsedMicroseconds / 1000 / 32;
    log('decode: ${record['decode']}');
    record['memory_decoding'] = memoryBreakdown();

    // Profiled: GPU milliseconds per kernel kind and token.
    GpuForward.profiling = true;
    await model.reset();
    await model.forward(prompt);
    GpuForward.takeProfile();
    for (var i = 0; i < 32; i++) {
      await model.step(argmaxToken(model.logits));
    }
    final profile = GpuForward.takeProfile();
    GpuForward.profiling = false;
    final kinds = profile.entries.toList()
      ..sort((a, b) => b.value.$1.compareTo(a.value.$1));
    record['kernels'] = {
      for (final MapEntry(key: kind, value: (ms, count)) in kinds)
        kind: {'ms_per_token': ms / 32, 'dispatches_per_token': count / 32},
    };
    log(
      'kernels: ${kinds.map((e) => '${e.key} ${(e.value.$1 / 32).toStringAsFixed(2)} ms x${e.value.$2 ~/ 32}').join(', ')}',
    );
    log('memory: ${record['memory_loaded']} -> ${record['memory_decoding']}');
  } finally {
    await model.close();
  }
  record['memory_closed'] = memoryBreakdown();
  record['cache_mapped_after_close'] = MappedFile.isMapped(
    '$bundle/${quantizedCacheName(TextModelConfig.load(bundle), WeightFormat.q8)}',
  );
  return record;
}
