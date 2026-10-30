/// The dianoia command-line tool, for LFM3 checkpoints.
///
///   dianoia ask --model=DIR --prompt=TEXT [--system=TEXT] [--tools=FILE.json]
///            [--think] [--max-new=256] [--workers=N]
///       Chat from text: renders the chat template, tokenizes, generates
///       greedily with streaming output, and prints parsed tool calls.
///
///   --workers=N splits every matrix product across N worker isolates by
///   rows (0, the default, runs everything in the main isolate).
///   --weights=f32|q5p4|q4p5|q7p3|q8|q4 quantizes the matrices at load time;
///   q8 and q4 run on the native kernels the build hook compiled
///   (--native-lib=PATH for a library built elsewhere, --native-threads=4,
///   --native-inputs=int8|f32) and cache the quantized weights
///   (--q8-cache=PATH, default `MODEL/lfm3-<weights>.cache`, or none).
///   --keep-f32=a,b keeps tensors whose name contains a or b in float32.
///   --gpu runs whole tokens on the GPU (Metal or Vulkan) from the cache
///   (needs --weights=q8 or q4); --gpu-lib=PATH takes the GPU layer from a
///   library built elsewhere.
///   --profile (run only) prints where decoding time goes.
///   --force=FILE (run only) feeds those token ids as the continuation
///   instead of the greedy choice, for teacher-forced quality comparison.
///
///   dianoia pack --model=DIR --out=DIR [--weights=q8|q4]
///            [--q8-cache=PATH]
///       Writes a bundle for devices: config.json, tokenizer.json, the
///       cache, and a model.safetensors with only what the cache lacks (the
///       1-D tensors, the conv kernels and the bf16 embedding for token
///       lookup). --gpu runs from it as from the full model, at
///       2.5 GB instead of 5.4 GB for Q8.
///
///   dianoia import-gguf --gguf=FILE --model=DIR --out=DIR
///       Turns Liquid's QAD Q4_0 GGUF of the model in DIR into a bundle
///       for --weights=q4 (see lib/src/gguf.dart), and prints how far each
///       imported tensor family is from DIR's weights.
///
///   dianoia toolbench --model=DIR --cases=FILE [--weights=... --gpu]
///       Asks each case of FILE (example/tool_cases.json) greedily with its
///       tools and checks the tool call against the expected name and
///       arguments (each argument must contain one of the listed
///       fragments, case-insensitively); null expects no call.
///
///   dianoia chatbench --model=DIR --cases=FILE [--state=PATH]
///            [--weights=... --gpu]
///       Time to first token with one system prompt and the whole tool
///       catalog of FILE, cold, warm (the system and tools block reused,
///       also through a state file, default DIR/chatbench.state) and in a
///       tool round trip; checks that reuse changes no logit (see
///       lib/src/chat_bench.dart).
///
///   dianoia selftest --model=DIR --oracle=FILE
///       Requires the tokenizer and chat template to reproduce the
///       Hugging Face oracle (test/fixtures/text_oracle.json, written by
///       dart-llm-lab's tools/lfm3_text_oracle.py) exactly.
///
///   dianoia run --model=DIR --ids=FILE --out=DIR [--max-new=N]
///            [--all-logits-max=512] [--max-seq=4096] [--chunk=64]
///            [--workers=N]
///       Runs a token-id prompt and dumps what dart-llm-lab's
///       tools/lfm3_compare.py needs:
///       hidden.f32 [T, L + 1, H], logits.f32 for the positions in
///       meta.json, gen.i32 (greedy) and gen_logits.f32.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';

/// Parses `--key=value` and bare `--flag` (stored as "true").
Map<String, String> _parseFlags(List<String> args) {
  final flags = <String, String>{};
  for (final arg in args) {
    if (!arg.startsWith('--')) {
      throw ArgumentError('expected --key=value or --flag, got $arg');
    }
    final eq = arg.indexOf('=');
    if (eq < 0) {
      flags[arg.substring(2)] = 'true';
    } else {
      flags[arg.substring(2, eq)] = arg.substring(eq + 1);
    }
  }
  return flags;
}

void _write(String path, TypedData data) => File(path).writeAsBytesSync(
  data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
);

int _argmax(Float32List v) {
  var best = 0;
  for (var i = 1; i < v.length; i++) {
    if (v[i] > v[best]) best = i;
  }
  return best;
}

Future<void> main(List<String> args) async {
  const commands = {
    'run': _run,
    'ask': _ask,
    'selftest': _selftest,
    'pack': _pack,
    'import-gguf': _importGguf,
    'toolbench': _toolbench,
    'chatbench': _chatbench,
  };
  final command = args.isEmpty ? null : commands[args.first];
  if (command == null) {
    stderr.writeln(
      'usage: dianoia ask|selftest|run|pack|import-gguf|toolbench|chatbench '
      '--key=value ...',
    );
    exitCode = 2;
    return;
  }
  exitCode = await command(_parseFlags(args.sublist(1)));
}

/// Enables the C kernels when [needed] (--weights=q8 or q4, pack,
/// import-gguf) or asked for: those the package's build hook compiled, or
/// with --native-lib=PATH a library built elsewhere. --native-threads=N
/// (default 4), --native-inputs=int8|f32 (default int8, the sdot path) and
/// --native-attention=false (attention stays in Dart) configure them.
void _loadNative(Map<String, String> flags, {bool needed = false}) {
  final path = flags['native-lib'];
  if (path == null && !needed) return;
  final inputs = flags['native-inputs'] ?? 'int8';
  if (inputs != 'int8' && inputs != 'f32') {
    throw ArgumentError('--native-inputs must be int8 or f32');
  }
  NativeKernels.load(
    path,
    threads: int.parse(flags['native-threads'] ?? '4'),
    int8Inputs: inputs == 'int8',
    attention: flags['native-attention'] != 'false',
  );
}

/// --q8-cache=PATH for --weights=q8 or q4; defaults to `lfm3-<weights>.cache`
/// in the model directory, and --q8-cache=none quantizes on every start.
String? _q8CachePath(
  Map<String, String> flags,
  String modelDir,
  WeightFormat weights,
) {
  if (!weights.isNative) return null;
  final path = flags['q8-cache'] ?? '$modelDir/lfm3-${weights.name}.cache';
  return path == 'none' ? null : path;
}

String Function(String) _requirer(Map<String, String> flags) =>
    (key) => flags[key] ?? (throw ArgumentError('missing --$key'));

Future<int> _run(Map<String, String> flags) async {
  final require = _requirer(flags);
  final modelDir = require('model');
  final outDir = require('out');
  final maxNew = int.parse(flags['max-new'] ?? '32');
  final allLogitsMax = int.parse(flags['all-logits-max'] ?? '512');
  final maxSeq = int.parse(flags['max-seq'] ?? '4096');
  final chunk = int.parse(flags['chunk'] ?? '64');
  final workers = int.parse(flags['workers'] ?? '0');
  final weights = WeightFormat.values.byName(flags['weights'] ?? 'f32');
  final keepF32 = (flags['keep-f32'] ?? '').split(',')..remove('');
  _loadNative(flags, needed: weights.isNative);
  final forced = switch (flags['force']) {
    final path? => Int32List.sublistView(File(path).readAsBytesSync()),
    null => null,
  };
  Directory(outDir).createSync(recursive: true);

  final ids = Int32List.sublistView(File(require('ids')).readAsBytesSync());
  final sw = Stopwatch()..start();
  final model = await Lfm3Model.load(
    modelDir,
    maxSeq: maxSeq,
    maxChunk: chunk,
    workers: workers,
    weights: weights,
    keepF32: keepF32,
    q8CachePath: _q8CachePath(flags, modelDir, weights),
    gpu: flags['gpu'] == 'true',
    gpuLib: flags['gpu-lib'],
  );
  final tokenizer = Lfm3Tokenizer.load(modelDir);
  final loadS = sw.elapsedMilliseconds / 1000;
  final c = model.config;
  final h = c.hiddenSize;
  final slots = c.numLayers + 1;
  final t = ids.length;
  stdout.writeln('loaded in ${loadS.toStringAsFixed(1)} s; prompt $t tokens');

  final positions = t <= allLogitsMax
      ? List<int>.generate(t, (i) => i)
      : [t - 1];
  final hidden = Float32List(t * slots * h);
  sw
    ..reset()
    ..start();
  // The whole prompt in one call: chunks of --chunk tokens use gemm, and
  // --chunk=1 exercises the single-token decode path instead.
  await model.forward(
    ids,
    capture: (slot, first, rows, count) {
      for (var r = 0; r < count; r++) {
        final pos = first + r;
        hidden.setRange(
          (pos * slots + slot) * h,
          (pos * slots + slot + 1) * h,
          rows,
          r * h,
        );
      }
    },
  );
  final promptS = sw.elapsedMilliseconds / 1000;
  final gpuPrompt = model.gpu;
  if (flags['profile'] == 'true' && gpuPrompt != null) {
    final gpu = gpuPrompt;
    stdout.writeln(
      'prompt profile: ${gpu.calls} chunks of up to ${model.maxChunk} tokens, '
      'GPU ${(gpu.gpuMicros / 1000).toStringAsFixed(1)} ms '
      '(${(gpu.gpuMicros / 1000 / t).toStringAsFixed(2)} ms/token), '
      'encoding ${(gpu.encodeMicros / 1000).toStringAsFixed(2)} ms',
    );
  }
  final lastLogits = Float32List.fromList(model.logits);
  final promptLogits = Float32List(positions.length * c.vocabSize);
  for (var i = 0; i < positions.length; i++) {
    final base = (positions[i] * slots + c.numLayers) * h;
    promptLogits.setRange(
      i * c.vocabSize,
      (i + 1) * c.vocabSize,
      await model.logitsOf(Float32List.sublistView(hidden, base, base + h)),
    );
  }
  model.logits.setAll(0, lastLogits);

  final gen = <int>[];
  final genLogits = <Float32List>[];
  sw
    ..reset()
    ..start();
  // With --force the continuation is fed instead of chosen (teacher
  // forcing), so every step's logits line up with the reference's.
  final steps = forced?.length ?? maxNew;
  var decodeSteps = 0;
  model.executorStats.clear();
  model.gpu?.clearStats();
  while (gen.length < steps) {
    final next = forced?[gen.length] ?? _argmax(model.logits);
    gen.add(next);
    genLogits.add(Float32List.fromList(model.logits));
    // The last token is only chosen, never fed, so it costs no forward pass.
    if (next == c.eosTokenId || gen.length == steps) break;
    await model.step(next);
    decodeSteps++;
  }
  final genS = sw.elapsedMilliseconds / 1000;
  final gpu = model.gpu;
  if (flags['profile'] == 'true' && decodeSteps > 0 && gpu != null) {
    double perToken(int micros) => micros / gpu.calls / 1000;
    stdout.writeln(
      'decode profile per token: total ${(genS * 1000 / decodeSteps).toStringAsFixed(1)} ms, '
      'encoding ${(gpu.dispatches / gpu.calls).toStringAsFixed(0)} dispatches '
      '${perToken(gpu.encodeMicros).toStringAsFixed(2)} ms, '
      'GPU ${perToken(gpu.gpuMicros).toStringAsFixed(1)} ms',
    );
  } else if (flags['profile'] == 'true' && decodeSteps > 0) {
    final st = model.executorStats;
    double perToken(int micros) => micros / decodeSteps / 1000;
    stdout.writeln(
      'decode profile per token: total ${(genS * 1000 / decodeSteps).toStringAsFixed(1)} ms, '
      '${(st.calls / decodeSteps).toStringAsFixed(0)} shard calls taking '
      '${perToken(st.wallMicros).toStringAsFixed(1)} ms wall; slowest shard computing '
      '${perToken(st.slowestComputeMicros).toStringAsFixed(1)} ms '
      '(mean shard ${perToken(st.totalComputeMicros ~/ math.max(1, workers)).toStringAsFixed(1)} ms); '
      'summing partials ${perToken(st.sumMicros).toStringAsFixed(1)} ms; '
      'outside shard calls ${(genS * 1000 / decodeSteps - perToken(st.wallMicros) - perToken(st.sumMicros)).toStringAsFixed(1)} ms',
    );
    if (workers > 0 && st.calls > 0) {
      final messages = st.calls * workers;
      stdout.writeln(
        'message delivery: request mean '
        '${(st.requestLatencyMicros / messages).toStringAsFixed(0)} us '
        '(worst ${st.worstRequestLatency} us), reply mean '
        '${(st.replyLatencyMicros / messages).toStringAsFixed(0)} us '
        '(worst ${st.worstReplyLatency} us)',
      );
    }
  }
  await model.close();

  _write('$outDir/hidden.f32', hidden);
  _write('$outDir/logits.f32', promptLogits);
  _write('$outDir/gen.i32', Int32List.fromList(gen));
  final flat = Float32List(genLogits.length * c.vocabSize);
  for (var i = 0; i < genLogits.length; i++) {
    flat.setRange(i * c.vocabSize, (i + 1) * c.vocabSize, genLogits[i]);
  }
  _write('$outDir/gen_logits.f32', flat);
  final text = tokenizer.decode(gen);
  File('$outDir/meta.json').writeAsStringSync(
    '${const JsonEncoder.withIndent('  ').convert({'seq_len': t, 'hidden_size': h, 'num_layers': c.numLayers, 'vocab_size': c.vocabSize, 'logit_positions': positions, 'gen_len': gen.length, 'generated': text, 'load_s': loadS, 'prompt_s': promptS, 'prompt_tok_per_s': t / promptS, 'workers': workers, 'weights': weights.name, 'native_threads': NativeKernels.isLoaded ? NativeKernels.instance.threads : null, 'native_int8_inputs': NativeKernels.isLoaded ? NativeKernels.instance.int8Inputs : null, 'native_attention': NativeKernels.isLoaded ? NativeKernels.instance.attention : null, 'gpu': gpu != null, 'keep_f32': keepF32, 'forced': forced != null, 'decode_s': genS, 'decode_tok_per_s': decodeSteps > 0 ? decodeSteps / genS : null})}\n',
  );
  stdout
    ..writeln(
      'prompt: ${(t / promptS).toStringAsFixed(2)} tok/s; '
      'decode: ${decodeSteps > 0 ? (decodeSteps / genS).toStringAsFixed(2) : '-'} tok/s',
    )
    ..writeln('generated: ${jsonEncode(text)}');
  return 0;
}

Future<int> _ask(Map<String, String> flags) async {
  final require = _requirer(flags);
  final modelDir = require('model');
  final maxNew = int.parse(flags['max-new'] ?? '256');
  final workers = int.parse(flags['workers'] ?? '0');
  final weights = WeightFormat.values.byName(flags['weights'] ?? 'f32');
  final keepF32 = (flags['keep-f32'] ?? '').split(',')..remove('');
  _loadNative(flags, needed: weights.isNative);
  final messages = <Map<String, Object?>>[
    if (flags['system'] case final system?)
      {'role': 'system', 'content': system},
    {'role': 'user', 'content': require('prompt')},
  ];
  final tools = switch (flags['tools']) {
    final path? => [
      for (final tool in jsonDecode(File(path).readAsStringSync()) as List)
        (tool as Map).cast<String, Object?>(),
    ],
    null => null,
  };

  final sw = Stopwatch()..start();
  final tokenizer = Lfm3Tokenizer.load(modelDir);
  final model = await Lfm3Model.load(
    modelDir,
    workers: workers,
    weights: weights,
    keepF32: keepF32,
    q8CachePath: _q8CachePath(flags, modelDir, weights),
    gpu: flags['gpu'] == 'true',
    gpuLib: flags['gpu-lib'],
  );
  final eos = model.config.eosTokenId;
  stdout.writeln(
    'loaded in ${(sw.elapsedMilliseconds / 1000).toStringAsFixed(1)} s',
  );

  final prompt = renderChat(
    messages,
    tools: tools,
    enableThinking: flags['think'] == 'true',
  );
  final ids = tokenizer.encode(prompt);
  sw
    ..reset()
    ..start();
  await model.forward(ids);
  final prefillS = sw.elapsedMilliseconds / 1000;
  stdout.writeln(
    'prompt: ${ids.length} tokens in ${prefillS.toStringAsFixed(1)} s '
    '(${(ids.length / prefillS).toStringAsFixed(2)} tok/s)\n',
  );

  final gen = <int>[];
  var shown = 0;
  sw
    ..reset()
    ..start();
  while (gen.length < maxNew) {
    final next = _argmax(model.logits);
    if (next == eos) break;
    gen.add(next);
    // Print only complete characters; a multi-byte character can span
    // several tokens.
    final text = tokenizer.decode(gen);
    if (!text.endsWith('�')) {
      stdout.write(text.substring(shown));
      shown = text.length;
    }
    await model.step(next);
  }
  final decodeS = sw.elapsedMilliseconds / 1000;
  await model.close();
  stdout.writeln(
    '\n\ndecode: ${gen.length} tokens in ${decodeS.toStringAsFixed(1)} s '
    '(${(gen.length / decodeS).toStringAsFixed(2)} tok/s)',
  );

  final reply = parseAssistantReply(tokenizer.decode(gen));
  if (reply.toolCalls.isNotEmpty) {
    stdout.writeln(
      'tool calls: '
      '${const JsonEncoder.withIndent('  ').convert([for (final c in reply.toolCalls) c.toJson()])}',
    );
  }
  return 0;
}

Future<int> _toolbench(Map<String, String> flags) async {
  final require = _requirer(flags);
  final modelDir = require('model');
  final weights = WeightFormat.values.byName(flags['weights'] ?? 'f32');
  _loadNative(flags, needed: weights.isNative);
  final spec =
      jsonDecode(File(require('cases')).readAsStringSync())
          as Map<String, Object?>;
  final catalog = (spec['tools'] as Map).cast<String, Object?>();
  final cases = (spec['cases'] as List).cast<Map<String, Object?>>();
  final tokenizer = Lfm3Tokenizer.load(modelDir);
  final model = await Lfm3Model.load(
    modelDir,
    workers: int.parse(flags['workers'] ?? '0'),
    weights: weights,
    q8CachePath: _q8CachePath(flags, modelDir, weights),
    gpu: flags['gpu'] == 'true',
    gpuLib: flags['gpu-lib'],
  );
  var passed = 0;
  for (final (i, c) in cases.indexed) {
    final tools = [
      for (final name in (c['tools'] as List).cast<String>())
        <String, Object?>{
          'type': 'function',
          'function': <String, Object?>{
            'name': name,
            ...(catalog[name] as Map).cast<String, Object?>(),
          },
        },
    ];
    final ids = tokenizer.encode(
      renderChat(
        [
          {'role': 'user', 'content': c['prompt']},
        ],
        tools: tools,
        enableThinking: false,
      ),
    );
    await model.reset();
    await model.forward(ids);
    final gen = <int>[];
    while (gen.length < 96) {
      final next = _argmax(model.logits);
      if (next == model.config.eosTokenId) break;
      gen.add(next);
      await model.step(next);
    }
    final text = tokenizer.decode(gen);
    final calls = parseAssistantReply(text).toolCalls;
    final ok = toolCallMatches(calls, c['expect'] as Map<String, Object?>?);
    if (ok) passed++;
    final shown = calls.isEmpty
        ? jsonEncode(text.length > 60 ? '${text.substring(0, 60)}...' : text)
        : [
            for (final call in calls)
              '${call.name}${jsonEncode(call.arguments)}',
          ].join(', ');
    stdout.writeln(
      '${ok ? 'pass' : 'FAIL'} ${i + 1}: ${c['prompt']} -> $shown',
    );
  }
  await model.close();
  stdout.writeln('passed $passed/${cases.length}');
  return 0;
}

Future<int> _chatbench(Map<String, String> flags) async {
  final require = _requirer(flags);
  final modelDir = require('model');
  final weights = WeightFormat.values.byName(flags['weights'] ?? 'f32');
  _loadNative(flags, needed: weights.isNative);
  final spec =
      jsonDecode(File(require('cases')).readAsStringSync())
          as Map<String, Object?>;
  final model = await Lfm3Model.load(
    modelDir,
    weights: weights,
    q8CachePath: _q8CachePath(flags, modelDir, weights),
    gpu: flags['gpu'] == 'true',
    gpuLib: flags['gpu-lib'],
  );
  final record = await chatBench(
    model,
    Lfm3Tokenizer.load(modelDir),
    spec,
    stdout.writeln,
    statePath: flags['state'] ?? '$modelDir/chatbench.state',
  );
  await model.close();
  return record['identical_logits'] == true ? 0 : 1;
}

Future<int> _pack(Map<String, String> flags) async {
  final require = _requirer(flags);
  final modelDir = require('model');
  final outDir = require('out');
  _loadNative(flags, needed: true);
  final weights = WeightFormat.values.byName(flags['weights'] ?? 'q8');
  if (!weights.isNative) throw ArgumentError('pack needs --weights=q8 or q4');
  final cachePath = _q8CachePath(flags, modelDir, weights)!;
  final cache =
      Q8Cache.open(
        cachePath,
        Q8Cache.keyFor(modelDir, weights: weights.name),
      ) ??
      (throw StateError(
        'no ${weights.name} cache for $modelDir: run with --weights=${weights.name}',
      ));
  Directory(outDir).createSync(recursive: true);
  for (final name in ['config.json', 'tokenizer.json']) {
    File('$modelDir/$name').copySync('$outDir/$name');
  }
  final st = SafetensorsFile.open('$modelDir/model.safetensors');
  try {
    // Matrices live in the cache, except that the embedding is also read as
    // bf16 rows for token lookup.
    st.writeSubset(
      '$outDir/model.safetensors',
      (name, info) =>
          info.shape.length != 2 || name == 'model.embed_tokens.weight',
    );
  } finally {
    st.close();
  }
  // The cache key fingerprints the safetensors header, so it changes.
  Q8Cache.write(
    '$outDir/lfm3-${weights.name}.cache',
    Q8Cache.keyFor(outDir, weights: weights.name),
    {
      for (final (name, rows, cols) in cache.shapes)
        name: cache.matrix(name, rows, cols)!,
    },
  );
  var bytes = 0;
  for (final f in Directory(outDir).listSync().whereType<File>()) {
    bytes += f.lengthSync();
  }
  stdout.writeln('wrote $outDir: ${(bytes / 1e9).toStringAsFixed(2)} GB');
  return 0;
}

Future<int> _importGguf(Map<String, String> flags) async {
  final require = _requirer(flags);
  final outDir = require('out');
  _loadNative(flags, needed: true);
  final sw = Stopwatch()..start();
  final comparison = importGguf(require('gguf'), require('model'), outDir);
  var bytes = 0;
  for (final f in Directory(outDir).listSync().whereType<File>()) {
    bytes += f.lengthSync();
  }
  stdout.writeln(
    'wrote $outDir: ${(bytes / 1e9).toStringAsFixed(2)} GB in '
    '${(sw.elapsedMilliseconds / 1000).toStringAsFixed(1)} s',
  );
  if (comparison.isNotEmpty) {
    stdout.writeln(
      'relative L2 difference from the model directory\'s weights:',
    );
    comparison.forEach(stdout.writeln);
  }
  return 0;
}

Future<int> _selftest(Map<String, String> flags) async {
  final require = _requirer(flags);
  final tokenizer = Lfm3Tokenizer.load(require('model'));
  final oracle =
      jsonDecode(File(require('oracle')).readAsStringSync())
          as Map<String, Object?>;
  var failures = 0;

  final encodeCases = oracle['encode'] as List;
  var encodeOk = 0;
  for (final c in encodeCases.cast<Map<String, Object?>>()) {
    final text = c['text'] as String;
    final expected = (c['ids'] as List).cast<int>();
    final got = tokenizer.encode(text);
    final diff = _firstDifference(got, expected);
    if (diff != null) {
      failures++;
      stdout.writeln(
        'ENCODE MISMATCH at token $diff of ${jsonEncode(_clip(text))}\n'
        '  dart: ${got.skip(diff).take(6).map((id) => jsonEncode(tokenizer.decode([id]))).toList()}\n'
        '  hf:   ${expected.skip(diff).take(6).map((id) => jsonEncode(tokenizer.decode([id]))).toList()}',
      );
    } else if (tokenizer.decode(expected) != text) {
      failures++;
      stdout.writeln('DECODE MISMATCH for ${jsonEncode(_clip(text))}');
    } else {
      encodeOk++;
    }
  }
  stdout.writeln(
    'encode/decode: $encodeOk/${encodeCases.length} texts match exactly',
  );

  final templates = oracle['templates'] as List;
  var templateOk = 0;
  for (final c in templates.cast<Map<String, Object?>>()) {
    final rendered = renderChat(
      [
        for (final m in c['messages'] as List)
          (m as Map).cast<String, Object?>(),
      ],
      tools: switch (c['tools']) {
        final List tools => [
          for (final t in tools) (t as Map).cast<String, Object?>(),
        ],
        _ => null,
      },
      enableThinking: c['enable_thinking'] as bool,
    );
    final expected = c['prompt'] as String;
    if (rendered != expected) {
      failures++;
      var i = 0;
      while (i < rendered.length &&
          i < expected.length &&
          rendered[i] == expected[i]) {
        i++;
      }
      stdout.writeln(
        'TEMPLATE MISMATCH in ${c['name']} at char $i\n'
        '  dart: ${jsonEncode(rendered.substring(i, math.min(i + 60, rendered.length)))}\n'
        '  hf:   ${jsonEncode(expected.substring(i, math.min(i + 60, expected.length)))}',
      );
    } else if (_firstDifference(
          tokenizer.encode(rendered),
          (c['ids'] as List).cast<int>(),
        ) !=
        null) {
      failures++;
      stdout.writeln('TEMPLATE TOKENS MISMATCH in ${c['name']}');
    } else {
      templateOk++;
    }
  }
  stdout.writeln(
    'chat template: $templateOk/${templates.length} prompts match exactly',
  );

  var parseOk = 0;
  for (final (text, check) in _parseCases) {
    final problem = check(parseAssistantReply(text));
    if (problem != null) {
      failures++;
      stdout.writeln('PARSE MISMATCH for ${jsonEncode(text)}: $problem');
    } else {
      parseOk++;
    }
  }
  stdout.writeln('tool-call parser: $parseOk/${_parseCases.length} cases');

  _loadNative(flags, needed: NativeKernels.isAvailable);
  if (NativeKernels.isLoaded) {
    final nativeProblems = [..._nativeQ8Check(), ..._nativeAttentionCheck()];
    failures += nativeProblems.length;
    nativeProblems.forEach(stdout.writeln);
    stdout.writeln(
      'native Q8, Q4 and attention kernels: ${nativeProblems.isEmpty ? 'match' : 'MISMATCH'}',
    );
  } else {
    stdout.writeln('native Q8 kernels: skipped (not built for this platform)');
  }

  final quantProblems = _quantKernelCheck();
  failures += quantProblems.length;
  quantProblems.forEach(stdout.writeln);
  stdout.writeln(
    'quantized kernels: ${quantProblems.isEmpty ? 'gemv and gemm match' : 'MISMATCH'}',
  );
  stdout.writeln(
    failures == 0 ? 'selftest passed' : 'selftest FAILED ($failures)',
  );
  return failures == 0 ? 0 : 1;
}

final _parseCases = <(String, String? Function(AssistantReply))>[
  (
    '<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n'
        '</function>\n</tool_call><|im_end|>',
    (r) =>
        r.toolCalls.length == 1 &&
            r.toolCalls.first.name == 'get_weather' &&
            r.toolCalls.first.arguments['city'] == 'Paris' &&
            r.content.isEmpty &&
            r.reasoning == null
        ? null
        : 'got ${jsonEncode([for (final c in r.toolCalls) c.toJson()])}, content ${jsonEncode(r.content)}',
  ),
  (
    'Let me check.<tool_call>\n<function=get_forecast>\n<parameter=city>\nTokyo\n</parameter>\n'
        '<parameter=days>\n3\n</parameter>\n<parameter=tags>\n["rain", "wind"]\n</parameter>\n'
        '</function>\n</tool_call>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\n'
        'Osaka\n</parameter>\n</function>\n</tool_call>',
    (r) =>
        r.toolCalls.length == 2 &&
            r.content == 'Let me check.' &&
            r.toolCalls[0].arguments['days'] == '3' &&
            r.toolCalls[0].arguments['tags'] == '["rain", "wind"]' &&
            r.toolCalls[1].arguments['city'] == 'Osaka'
        ? null
        : 'got ${jsonEncode([for (final c in r.toolCalls) c.toJson()])}, content ${jsonEncode(r.content)}',
  ),
  (
    '<think>\nThe user wants a sum.</think>\n\n2 + 2 = 4.<|im_end|>',
    (r) => r.reasoning == 'The user wants a sum.' && r.content == '2 + 2 = 4.'
        ? null
        : 'reasoning ${jsonEncode(r.reasoning)}, content ${jsonEncode(r.content)}',
  ),
  (
    'Line one\n<parameter=not_a_call> stays text',
    (r) =>
        r.toolCalls.isEmpty &&
            r.content == 'Line one\n<parameter=not_a_call> stays text'
        ? null
        : 'got ${r.toolCalls.length} calls',
  ),
];

int? _firstDifference(List<int> a, List<int> b) {
  for (var i = 0; i < math.min(a.length, b.length); i++) {
    if (a[i] != b[i]) return i;
  }
  return a.length == b.length ? null : math.min(a.length, b.length);
}

String _clip(String text) =>
    text.length <= 60 ? text : '${text.substring(0, 60)}...';

/// gemv and the tiled gemm of every quantized format must compute exactly
/// the product with the weights they decode (`weightAt`), up to float32
/// summation error. The shapes cover a column count that needs padding
/// (2048 for q4p5), a row count that leaves a partial and an odd tile, and
/// a batch that leaves a tail after the 4-vector blocks.
List<String> _quantKernelCheck() {
  final problems = <String>[];
  final rng = math.Random(7);
  for (final format in [
    WeightFormat.q5p4,
    WeightFormat.q4p5,
    WeightFormat.q7p3,
  ]) {
    for (final (rows, cols, n) in [
      (35, 2048, 1),
      (35, 2048, 7),
      (18, 7680, 6),
    ]) {
      final w = Float32List(rows * cols);
      for (var i = 0; i < w.length; i++) {
        w[i] = rng.nextDouble() - 0.5;
      }
      final xs = Float32List(n * cols);
      for (var i = 0; i < xs.length; i++) {
        xs[i] = rng.nextDouble() * 2 - 1;
      }
      final q = QuantizedMatrix.quantize(w, rows, cols, format);
      final ys = Float32List(n * rows);
      if (n == 1) {
        q.gemv(xs, ys);
      } else {
        q.gemm(xs, ys, n);
      }
      var worst = 0.0;
      for (var t = 0; t < n; t++) {
        for (var r = 0; r < rows; r++) {
          var ref = 0.0, mag = 0.0;
          for (var c = 0; c < cols; c++) {
            final d = q.weightAt(r, c) * xs[t * cols + c];
            ref += d;
            mag += d.abs();
          }
          worst = math.max(worst, (ys[t * rows + r] - ref).abs() / mag);
        }
      }
      if (worst > 1e-5) {
        problems.add(
          'QUANT MISMATCH ${format.name} $rows x $cols n=$n: relative error $worst',
        );
      }
    }
  }
  return problems;
}

/// The native Q8 and Q4 products must equal, up to float32 summation error,
/// the product with the weights they store (`weightAt`). With int8 inputs
/// the reference quantizes the inputs exactly as the kernel does (per
/// 32-value block, scale = max|x| / 127, round half away from zero).
List<String> _nativeQ8Check() {
  final problems = <String>[];
  final rng = math.Random(11);
  for (final (format, quantize) in [
    ('Q8', NativeQ8Matrix.quantize),
    ('Q4', NativeQ4Matrix.quantize),
  ]) {
    for (final (rows, cols, n) in [
      (37, 2048, 1),
      (37, 2048, 6),
      (20, 7680, 5),
    ]) {
      final w = Float32List(rows * cols);
      for (var i = 0; i < w.length; i++) {
        w[i] = rng.nextDouble() - 0.5;
      }
      final xs = Float32List(n * cols);
      for (var i = 0; i < xs.length; i++) {
        xs[i] = rng.nextDouble() * 2 - 1;
      }
      final NativeMatrix q = quantize(w, rows, cols);
      for (final int8 in [false, true]) {
        final ys = Float32List(n * rows);
        q.apply(xs, ys, n, int8Inputs: int8);
        final x = Float64List(n * cols);
        for (var t = 0; t < n; t++) {
          for (var b = 0; b < cols ~/ 32; b++) {
            final base = t * cols + b * 32;
            var amax = 0.0;
            for (var k = 0; k < 32; k++) {
              amax = math.max(amax, xs[base + k].abs());
            }
            final scale = Float32List.fromList([amax / 127])[0];
            final inv = scale > 0 ? Float32List.fromList([1 / scale])[0] : 0.0;
            for (var k = 0; k < 32; k++) {
              if (!int8) {
                x[base + k] = xs[base + k];
                continue;
              }
              final v = Float32List.fromList([xs[base + k] * inv])[0];
              final r = (v < 0 ? v - 0.5 : v + 0.5).truncate().clamp(-127, 127);
              x[base + k] = r * scale;
            }
          }
        }
        var worst = 0.0;
        for (var t = 0; t < n; t++) {
          for (var r = 0; r < rows; r++) {
            var ref = 0.0, mag = 0.0;
            for (var c = 0; c < cols; c++) {
              final d = q.weightAt(r, c) * x[t * cols + c];
              ref += d;
              mag += d.abs();
            }
            worst = math.max(worst, (ys[t * rows + r] - ref).abs() / mag);
          }
        }
        if (worst > 1e-5) {
          problems.add(
            'NATIVE $format MISMATCH ${int8 ? 'int8' : 'f32'} inputs $rows x $cols n=$n: relative error $worst',
          );
        }
      }
    }
  }
  return problems;
}

/// Native attention against a scalar double-precision version of the same
/// math, over a full-attention chunk, a decode step past the sliding window,
/// and a chunk that crosses the window boundary.
List<String> _nativeAttentionCheck() {
  final problems = <String>[];
  final rng = math.Random(5);
  const heads = 16, kvHeads = 4, hd = 128, window = 256;
  for (final (n, position, win) in [
    (5, 10, 0),
    (1, 300, window),
    (7, 252, window),
  ]) {
    final kvDim = kvHeads * hd;
    final len = position + n;
    Float32List random(int count, double spread) {
      final v = Float32List(count);
      for (var i = 0; i < count; i++) {
        v[i] = (rng.nextDouble() * 2 - 1) * spread;
      }
      return v;
    }

    final q = random(n * heads * hd, 0.3);
    final keys = random(len * kvDim, 0.3);
    final values = random(len * kvDim, 1);
    final gate = random(n * heads, 2);
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
      window: win,
    );
    var worst = 0.0;
    for (var t = 0; t < n; t++) {
      final pos = position + t;
      final start = win > 0 ? math.max(0, pos - win + 1) : 0;
      for (var h = 0; h < heads; h++) {
        final kvo = (h ~/ (heads ~/ kvHeads)) * hd;
        final scores = <double>[];
        for (var s = start; s <= pos; s++) {
          var dot = 0.0;
          for (var d = 0; d < hd; d++) {
            dot += q[t * heads * hd + h * hd + d] * keys[s * kvDim + kvo + d];
          }
          scores.add(dot / math.sqrt(hd));
        }
        final mx = scores.reduce(math.max);
        final e = [for (final v in scores) math.exp(v - mx)];
        final sum = e.reduce((a, b) => a + b);
        final g = 1 / (1 + math.exp(-gate[t * heads + h]));
        for (var d = 0; d < hd; d++) {
          var acc = 0.0, mag = 0.0;
          for (var s = start; s <= pos; s++) {
            final v = e[s - start] * values[s * kvDim + kvo + d];
            acc += v;
            mag += v.abs();
          }
          final ref = acc / sum * g;
          worst = math.max(
            worst,
            (out[t * heads * hd + h * hd + d] - ref).abs() / (mag / sum * g),
          );
        }
      }
    }
    if (worst > 1e-5) {
      problems.add(
        'NATIVE ATTENTION MISMATCH n=$n position=$position window=$win: relative error $worst',
      );
    }
  }
  return problems;
}
