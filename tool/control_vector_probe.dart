/// Compares replies with and without a control vector.
/// Usage: `dart run tool/control_vector_probe.dart <bundle> <vector> <prompts.txt>`
///        `[--mode=add|project] [--scale=1] [--layers=FIRST-LAST]`
///        `[--max-new-tokens=48] [--system=prompt] [--threads=4] [--gpu]`
/// The vector is a file from tool/control_vector_extract.dart or a llama.cpp
/// GGUF control vector. Each non-empty line of the prompts file is one user
/// turn; both replies are greedy from a reset model, so they differ only by
/// the vector. --layers takes zero-based layers, inclusive: `8-12`, `8-`
/// (8 to the last) or `8`. Runs on the CPU kernels, or with --gpu on the
/// GPU (Metal or Vulkan); also timing decoding with and without the vector.
library;

import 'dart:io';

import 'package:dianoia/lab.dart';

const _usage =
    'usage: dart run tool/control_vector_probe.dart <bundle> <vector> '
    '<prompts.txt> [--mode=add|project] [--scale=1] [--layers=FIRST-LAST] '
    '[--max-new-tokens=48] [--system=prompt] [--threads=4] [--gpu]';

Future<void> main(List<String> args) async {
  final positional = <String>[];
  final options = <String, String>{};
  var gpu = false;
  for (final arg in args) {
    if (arg == '--gpu') {
      gpu = true;
    } else if (arg.startsWith('--') && arg.contains('=')) {
      final at = arg.indexOf('=');
      options[arg.substring(2, at)] = arg.substring(at + 1);
    } else {
      positional.add(arg);
    }
  }
  const known = {
    'mode',
    'scale',
    'layers',
    'max-new-tokens',
    'system',
    'threads',
  };
  final mode = ControlMode.values
      .where((m) => m.name == (options['mode'] ?? 'add'))
      .firstOrNull;
  final scale = double.tryParse(options['scale'] ?? '1');
  final maxNew = int.tryParse(options['max-new-tokens'] ?? '48');
  final threads = int.tryParse(options['threads'] ?? '4');
  final layers = _layerRange(options['layers']);
  if (positional.length != 3 ||
      !known.containsAll(options.keys) ||
      mode == null ||
      scale == null ||
      maxNew == null ||
      maxNew <= 0 ||
      threads == null ||
      threads <= 0 ||
      layers == null) {
    stderr.writeln(_usage);
    exitCode = 64;
    return;
  }
  final [bundle, vectorPath, promptsPath] = positional;
  final prompts = [
    for (final line in File(promptsPath).readAsLinesSync())
      if (line.trim().isNotEmpty) line.trim(),
  ];

  NativeKernels.load(null, threads: threads);
  final vector = ControlVector.load(
    vectorPath,
    mode: mode,
    scale: scale,
    firstLayer: layers.$1,
    lastLayer: layers.$2,
  );
  final tokenizer = Lfm3Tokenizer.load(bundle);
  final model = await _load(bundle, gpu: gpu);
  try {
    final steered = [
      for (var l = 0; l < model.config.numLayers; l++)
        if (vector.steers(l)) l,
    ];
    print(
      'mode=${mode.name} scale=$scale layers=${steered.join(',')} '
      '(${steered.length} of ${model.config.numLayers})',
    );
    final session = ChatSession(
      model,
      tokenizer,
      system: options['system'],
      maxNew: maxNew,
    );
    Future<ChatTurn> reply(String prompt, ControlVector? control) async {
      await model.reset();
      model.setControlVector(control);
      session.clear();
      return session.send({'role': 'user', 'content': prompt});
    }

    String rate(ChatTurn turn) =>
        '${turn.generatedTokens} tokens, '
        '${(turn.generatedTokens / turn.decodeSeconds).toStringAsFixed(1)}'
        ' tokens/s';

    for (final (i, prompt) in prompts.indexed) {
      final base = await reply(prompt, null);
      final changed = await reply(prompt, vector);
      print('\n=== ${i + 1}/${prompts.length}: $prompt');
      print('--- without (${rate(base)})');
      print(base.text);
      print('--- with (${rate(changed)})');
      print(changed.text);
    }
  } finally {
    await model.close();
  }
}

/// `FIRST-LAST`, `FIRST-` or `LAYER` as (first, last); (0, null) without
/// [text]; null if malformed.
(int, int?)? _layerRange(String? text) {
  if (text == null) return (0, null);
  final match = RegExp(r'^(\d+)(-(\d*))?$').firstMatch(text);
  if (match == null) return null;
  final first = int.parse(match[1]!);
  if (match[2] == null) return (first, first);
  final last = match[3]!.isEmpty ? null : int.parse(match[3]!);
  if (last != null && last < first) return null;
  return (first, last);
}

/// The bundle's Q4 or Q8 weights, on the CPU kernels or with [gpu] on the
/// GPU (Metal or Vulkan).
Future<TextModel> _load(String bundle, {required bool gpu}) {
  final weights = [WeightFormat.q4, WeightFormat.q8].firstWhere(
    (w) => File('$bundle/lfm3-${w.name}.cache').existsSync(),
    orElse: () => throw ArgumentError('$bundle has no lfm3-q4/q8.cache'),
  );
  return TextModel.load(
    bundle,
    maxSeq: 2048,
    weights: weights,
    q8CachePath: '$bundle/lfm3-${weights.name}.cache',
    gpu: gpu,
  );
}
