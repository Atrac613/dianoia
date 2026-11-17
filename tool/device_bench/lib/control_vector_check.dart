/// The device benchmark's `control_vector` mode: a control vector on this
/// phone's CPU and GPU (benchmarkControlVector in the dianoia lab API),
/// for run_ios_control_vector.sh to compare with the same run on a Mac.
library;

import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';

/// Runs the checks [config] describes (paths relative to [docs]) and writes
/// the record to `bench/control-vector-latest.json`, finished or not, after
/// every step. Records token ids, counts and rates only, never reply text.
Future<Map<String, Object?>> controlVectorCheck(
  Map<String, Object?> config,
  String docs,
  void Function(String) log,
) async {
  final output = File('$docs/bench/control-vector-latest.json');
  output.parent.createSync(recursive: true);
  final started = DateTime.now().toIso8601String();
  void write(Map<String, Object?> record) => output.writeAsStringSync(
    const JsonEncoder.withIndent('  ').convert({
      'started': started,
      'bundle': config['bundle'],
      'finished': false,
      ...record,
    }),
  );

  List<String> lines(String path) => [
    for (final line in File('$docs/$path').readAsLinesSync())
      if (line.trim().isNotEmpty) line.trim(),
  ];

  write({});
  NativeKernels.load(null, threads: 6);
  final vector = ControlVector.load(
    '$docs/${config['vector'] as String}',
    mode: ControlMode.values.byName(config['vector_mode'] as String),
    scale: (config['scale'] as num).toDouble(),
    firstLayer: (config['first_layer'] as num).toInt(),
    lastLayer: (config['last_layer'] as num?)?.toInt(),
  );
  return benchmarkControlVector(
    bundle: '$docs/${config['bundle'] as String}',
    weights: WeightFormat.values.byName(config['weights'] as String),
    vector: vector,
    prompts: lines(config['prompts'] as String),
    refusalPrompts: switch (config['refusals']) {
      final String path => lines(path),
      _ => const [],
    },
    maxNewTokens: (config['max_new_tokens'] as num?)?.toInt() ?? 24,
    log: log,
    onRecord: write,
  );
}
