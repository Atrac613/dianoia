/// Runs the control-vector checks of lib/src/control_vector_bench.dart on
/// this machine's CPU and GPU, and compares them with a record from the
/// device bench app (tool/device_bench/run_ios_control_vector.sh).
/// Usage: `dart run tool/control_vector_bench.dart <bundle> <vector> <prompts.txt>`
///        `[--refusals=prompts.txt] [--mode=add|project] [--scale=1]`
///        `[--layers=FIRST-LAST] [--weights=q4|q8] [--max-new-tokens=24]`
///        `[--out=record.json] [--compare=device.json]`
/// Prints the rates, the self-check and the refusal counts of each record,
/// never reply text, and with --compare whether the device's greedy token
/// ids equal this machine's, per prompt and backend, with the first index
/// where they differ.
library;

import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';

const _usage =
    'usage: dart run tool/control_vector_bench.dart <bundle> <vector> '
    '<prompts.txt> [--refusals=prompts.txt] [--mode=add|project] [--scale=1] '
    '[--layers=FIRST-LAST] [--weights=q4|q8] [--max-new-tokens=24] '
    '[--out=record.json] [--compare=device.json]';

Future<void> main(List<String> args) async {
  final positional = <String>[];
  final options = <String, String>{};
  for (final arg in args) {
    if (arg.startsWith('--') && arg.contains('=')) {
      final at = arg.indexOf('=');
      options[arg.substring(2, at)] = arg.substring(at + 1);
    } else {
      positional.add(arg);
    }
  }
  const known = {
    'refusals',
    'mode',
    'scale',
    'layers',
    'weights',
    'max-new-tokens',
    'out',
    'compare',
  };
  final mode = ControlMode.values
      .where((m) => m.name == (options['mode'] ?? 'project'))
      .firstOrNull;
  final scale = double.tryParse(options['scale'] ?? '1');
  final layers = parseLayerRange(options['layers']);
  final weights = WeightFormat.values
      .where((w) => w.name == (options['weights'] ?? 'q4') && w.isNative)
      .firstOrNull;
  final maxNew = int.tryParse(options['max-new-tokens'] ?? '24');
  if (positional.length != 3 ||
      !known.containsAll(options.keys) ||
      mode == null ||
      scale == null ||
      layers == null ||
      weights == null ||
      maxNew == null ||
      maxNew <= 0) {
    stderr.writeln(_usage);
    exitCode = 64;
    return;
  }
  final [bundle, vectorPath, promptsPath] = positional;
  NativeKernels.load(null, threads: 6);
  final record = await benchmarkControlVector(
    bundle: bundle,
    weights: weights,
    vector: ControlVector.load(
      vectorPath,
      mode: mode,
      scale: scale,
      firstLayer: layers.$1,
      lastLayer: layers.$2,
    ),
    prompts: readPrompts(promptsPath),
    refusalPrompts: switch (options['refusals']) {
      null => const [],
      final path => readPrompts(path),
    },
    maxNewTokens: maxNew,
    log: stderr.writeln,
  );
  if (options['out'] case final out?) {
    File(
      out,
    ).writeAsStringSync(const JsonEncoder.withIndent('  ').convert(record));
  }
  final device = switch (options['compare']) {
    null => null,
    final path =>
      (jsonDecode(File(path).readAsStringSync()) as Map)
          .cast<String, Object?>(),
  };
  if (device != null) {
    print('== device');
    _summary(device);
  }
  print('== this machine');
  _summary(record);
  if (device != null) _compare(device, record);
}

/// The non-empty lines of [path].
List<String> readPrompts(String path) => [
  for (final line in File(path).readAsLinesSync())
    if (line.trim().isNotEmpty) line.trim(),
];

/// `FIRST-LAST`, `FIRST-` or `LAYER` as (first, last); (0, null) without
/// [text]; null if malformed.
(int, int?)? parseLayerRange(String? text) {
  if (text == null) return (0, null);
  final match = RegExp(r'^(\d+)(-(\d*))?$').firstMatch(text);
  if (match == null) return null;
  final first = int.parse(match[1]!);
  if (match[2] == null) return (first, first);
  final last = match[3]!.isEmpty ? null : int.parse(match[3]!);
  if (last != null && last < first) return null;
  return (first, last);
}

void _summary(Map<String, Object?> r) {
  final d = (r['device'] as Map?)?.cast<String, Object?>();
  print(
    '  ${d?['gpu'] ?? '?'}, ${r['os']}, weights ${r['weights']}, '
    'vector ${r['vector']}',
  );
  for (final name in ['cpu', 'gpu']) {
    final b = (r[name] as Map?)?.cast<String, Object?>();
    if (b == null) {
      print('  $name: not run');
      continue;
    }
    final rates = (b['rates'] as Map).cast<String, Object?>();
    String list(String key, String field) => [
      for (final run in (rates[key] as List).cast<Map>())
        (run[field] as num).toStringAsFixed(1),
    ].join(', ');
    print(
      '  $name decode tok/s without: ${list('without', 'decode_tok_per_s')}'
      ' | with: ${list('with', 'decode_tok_per_s')}',
    );
    print(
      '  $name prefill tok/s (${rates['prompt_tokens']} tokens) without: '
      '${list('without', 'prefill_tok_per_s')} | with: '
      '${list('with', 'prefill_tok_per_s')}',
    );
    print('  $name self-check: ${b['self_check']}');
    if (b['refusals'] case final Map refusals) {
      print(
        '  $name refusals: ${refusals['without']}/${refusals['prompts']} '
        'without, ${refusals['with']}/${refusals['prompts']} with',
      );
    }
  }
}

void _compare(Map<String, Object?> device, Map<String, Object?> mac) {
  for (final name in ['cpu', 'gpu']) {
    final a = (device[name] as Map?)?['replies'] as List?;
    final b = (mac[name] as Map?)?['replies'] as List?;
    if (a == null || b == null) continue;
    print('== $name token ids, device against this machine');
    for (var i = 0; i < a.length && i < b.length; i++) {
      final parts = <String>[];
      for (final key in ['without', 'with']) {
        final x = ((a[i] as Map)[key] as List).cast<int>();
        final y = ((b[i] as Map)[key] as List).cast<int>();
        var first = -1;
        for (var t = 0; t < x.length && t < y.length; t++) {
          if (x[t] != y[t]) {
            first = t;
            break;
          }
        }
        if (first < 0 && x.length != y.length) {
          first = x.length < y.length ? x.length : y.length;
        }
        parts.add(
          first < 0
              ? '$key identical (${x.length})'
              : '$key differ at $first of ${x.length}/${y.length}',
        );
      }
      print('  prompt ${i + 1}: ${parts.join(', ')}');
    }
  }
}
