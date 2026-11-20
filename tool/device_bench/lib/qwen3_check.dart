/// The device benchmark's `qwen3` mode: Qwen3 from Hugging Face to a reply,
/// on the phone alone. ModelStore downloads Qwen's Q8_0 GGUF and imports it
/// (unless installed), then the model answers fixed prompts on the GPU and
/// on the CPU: greedy, a tool call, a second turn that reuses the first,
/// and sampled thinking with a fixed seed. Free of Flutter, so the same
/// checks run on a Mac (run_ios_qwen3.sh) for comparison.
library;

import 'dart:io';

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
      contextLength: 2048,
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
        'decode_tok_per_s': t.generatedTokens / t.decodeSeconds,
        if (footprint != null) 'footprint_mb': footprint() >> 20,
      };
      void report(String name, ChatTurn t) {
        final d = describe(t);
        result[name] = d;
        log(
          '  $name: ${t.generatedTokens} tokens at '
          '${(d['decode_tok_per_s'] as double).toStringAsFixed(1)} tok/s, '
          'reused ${t.reusedTokens}/${t.promptTokens}: '
          '${t.reply.toolCalls.isEmpty ? t.reply.content : d['tool_calls']}',
        );
        save();
      }

      final chat = model.chat(maxNewTokens: 64);
      report(
        'greedy',
        await chat.send({
          'role': 'user',
          'content': 'What is the capital of Japan? Answer in one sentence.',
        }),
      );
      report(
        'second_turn',
        await chat.send({
          'role': 'user',
          'content': 'And of France? Answer in one sentence.',
        }),
      );
      report(
        'tool_call',
        await model.chat(tools: [_weatherTool], maxNewTokens: 64).send({
          'role': 'user',
          'content': "What's the weather like in Paris right now?",
        }),
      );
      if (backend != InferenceBackend.cpu || config['cpu_thinking'] == true) {
        report(
          'thinking',
          await model
              .chat(
                maxNewTokens: 2048,
                enableThinking: true,
                // Qwen3's thinking preset, seeded.
                sampling: const SamplingParams(
                  temperature: 0.6,
                  topK: 20,
                  topP: 0.95,
                  seed: 7,
                ),
              )
              .send({'role': 'user', 'content': 'What is 17 * 23?'}),
        );
      }
    } finally {
      await model.close();
    }
  }
}
