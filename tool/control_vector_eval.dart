/// Evaluates one control-vector configuration on labelled prompt sets
/// (ControlVectorEval in the lab API; docs/control_vectors.md).
/// Usage: `dart run tool/control_vector_eval.dart <bundle> <vector>`
///        `[--mode=add|project] [--scale=1] [--layers=FIRST-LAST]`
///        `[--refused=NAME:path]... [--benign=NAME:path]...`
///        `[--reference=NAME:path]... [--refusal-tokens=12]`
///        `[--compare-tokens=40] [--reference-tokens=96]`
///        `[--gpu] [--threads=4] [--out=result.json]`
/// Sets are `.txt` (one prompt per line) or `.jsonl` (`prompt`, and
/// `reference` for --reference). Prints a table and writes the numbers as
/// JSON: counts, rates and likelihoods only, never generated text or token
/// ids. Progress goes to stderr.
library;

import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';

const _usage =
    'usage: dart run tool/control_vector_eval.dart <bundle> <vector> '
    '[--mode=add|project] [--scale=1] [--layers=FIRST-LAST] '
    '[--refused=NAME:path]... [--benign=NAME:path]... '
    '[--reference=NAME:path]... [--refusal-tokens=12] [--compare-tokens=40] '
    '[--reference-tokens=96] [--gpu] [--threads=4] [--out=result.json]';

Future<void> main(List<String> args) async {
  final positional = <String>[];
  final options = <String, String>{};
  final sets = <(String, String, String)>[]; // kind, name, path
  var gpu = false;
  for (final arg in args) {
    if (arg == '--gpu') {
      gpu = true;
    } else if (arg.startsWith('--') && arg.contains('=')) {
      final at = arg.indexOf('=');
      final key = arg.substring(2, at), value = arg.substring(at + 1);
      if (const {'refused', 'benign', 'reference'}.contains(key)) {
        final colon = value.indexOf(':');
        if (colon <= 0 || colon == value.length - 1) {
          stderr.writeln('--$key takes NAME:path');
          exitCode = 64;
          return;
        }
        sets.add((key, value.substring(0, colon), value.substring(colon + 1)));
      } else {
        options[key] = value;
      }
    } else {
      positional.add(arg);
    }
  }
  const known = {
    'mode',
    'scale',
    'layers',
    'refusal-tokens',
    'compare-tokens',
    'reference-tokens',
    'threads',
    'out',
  };
  int? positive(String key, int fallback) {
    final v = int.tryParse(options[key] ?? '$fallback');
    return v != null && v > 0 ? v : null;
  }

  final mode = ControlMode.values
      .where((m) => m.name == (options['mode'] ?? 'project'))
      .firstOrNull;
  final scale = double.tryParse(options['scale'] ?? '1');
  final layers = _layerRange(options['layers']);
  final refusalTokens = positive('refusal-tokens', 12);
  final compareTokens = positive('compare-tokens', 40);
  final referenceTokens = positive('reference-tokens', 96);
  final threads = positive('threads', 4);
  if (positional.length != 2 ||
      !known.containsAll(options.keys) ||
      sets.isEmpty ||
      mode == null ||
      scale == null ||
      layers == null ||
      refusalTokens == null ||
      compareTokens == null ||
      referenceTokens == null ||
      threads == null) {
    stderr.writeln(_usage);
    exitCode = 64;
    return;
  }
  final [bundle, vectorPath] = positional;
  final data = [
    for (final (kind, name, path) in sets) (kind, name, readEvalRows(path)),
  ];

  NativeKernels.load(null, threads: threads);
  final vector = ControlVector.load(
    vectorPath,
    mode: mode,
    scale: scale,
    firstLayer: layers.$1,
    lastLayer: layers.$2,
  );
  final weights = [WeightFormat.q4, WeightFormat.q8].firstWhere(
    (w) => File('$bundle/lfm3-${w.name}.cache').existsSync(),
    orElse: () => throw ArgumentError('$bundle has no lfm3-q4/q8.cache'),
  );
  final model = await Lfm3Model.load(
    bundle,
    maxSeq: 2048,
    weights: weights,
    q8CachePath: '$bundle/lfm3-${weights.name}.cache',
    gpu: gpu,
  );
  final clock = Stopwatch()..start();
  final results = <String, Object?>{};
  try {
    final eval = ControlVectorEval(
      model,
      Lfm3Tokenizer.load(bundle),
      vector,
      refusalTokens: refusalTokens,
      compareTokens: compareTokens,
      referenceTokens: referenceTokens,
      log: stderr.writeln,
    );
    for (final (kind, name, rows) in data) {
      final started = clock.elapsedMilliseconds;
      final result = await switch (kind) {
        'refused' => eval.refused(name, rows),
        'benign' => eval.benign(name, rows),
        _ => eval.reference(name, rows),
      };
      result['seconds'] = (clock.elapsedMilliseconds - started) / 1000;
      results[name] = result;
    }
  } finally {
    await model.close();
  }
  final steered = [
    for (var l = 0; l < model.config.numLayers; l++)
      if (vector.steers(l)) l,
  ];
  final record = {
    'bundle': bundle,
    'weights': weights.name,
    'backend': gpu ? 'gpu' : 'cpu',
    'vector': {
      'file': vectorPath,
      'mode': mode.name,
      'scale': scale,
      'steered_layers': steered,
      'fingerprint': vector.fingerprint,
    },
    'refusal_tokens': refusalTokens,
    'compare_tokens': compareTokens,
    'reference_tokens': referenceTokens,
    'seconds': clock.elapsedMilliseconds / 1000,
    'sets': results,
  };
  if (options['out'] case final out?) {
    File(
      out,
    ).writeAsStringSync(const JsonEncoder.withIndent('  ').convert(record));
  }
  _print(record);
}

void _print(Map<String, Object?> record) {
  final v = record['vector'] as Map<String, Object?>;
  final layers = (v['steered_layers'] as List).cast<int>();
  print(
    '${record['backend']} ${record['weights']}, ${v['mode']} scale '
    '${v['scale']}, layers ${layers.isEmpty ? '-' : '${layers.first}-${layers.last}'}'
    ' (${layers.length})',
  );
  String pct(Object? r) {
    final m = r as Map<String, Object?>;
    final ci = (m['ci95'] as List).cast<double>();
    return '${m['count']} ${(100 * (m['rate'] as double)).toStringAsFixed(1)}% '
        '[${(100 * ci[0]).toStringAsFixed(1)}, ${(100 * ci[1]).toStringAsFixed(1)}]';
  }

  final sets = record['sets'] as Map<String, Object?>;
  for (final MapEntry(key: name, value: r as Map<String, Object?>)
      in sets.entries) {
    final time = '${(r['seconds'] as double).toStringAsFixed(0)} s';
    switch (r['kind']) {
      case 'refused' || 'benign':
        print(
          '${r['kind']} $name (${r['prompts']}, $time): refusal openings '
          '${pct(r['refusal_without'])} -> ${pct(r['refusal_with'])}; '
          'refuse->comply ${r['refuse_to_comply']}, comply->refuse '
          '${r['comply_to_refuse']}',
        );
        if (r['kind'] == 'benign') {
          print(
            '    first-token KL ${(r['first_token_kl'] as double).toStringAsFixed(4)}; '
            'common prefix ${(r['mean_common_prefix'] as double).toStringAsFixed(1)} '
            'of ${r['compare_tokens']} tokens; identical ${r['identical']}/${r['prompts']}',
          );
        }
      default:
        final ci = (r['nll_diff_ci95'] as List).cast<double>();
        print(
          'reference $name (${r['rows']} rows, ${r['scored_tokens']} tokens, '
          '$time): NLL/token ${(r['nll_without'] as double).toStringAsFixed(4)}'
          ' -> ${(r['nll_with'] as double).toStringAsFixed(4)} (ppl '
          '${(r['ppl_without'] as double).toStringAsFixed(2)} -> '
          '${(r['ppl_with'] as double).toStringAsFixed(2)}); diff '
          '${(r['nll_diff'] as double).toStringAsFixed(4)} '
          '[${ci[0].toStringAsFixed(4)}, ${ci[1].toStringAsFixed(4)}]; '
          'worse ${r['rows_worse']}/${r['rows']}',
        );
    }
  }
  print('total ${(record['seconds'] as double).toStringAsFixed(0)} s');
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
