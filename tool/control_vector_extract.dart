/// Extracts a control vector from two contrasting sets of prompts: per
/// layer, the mean residual stream leaving that layer at the last prompt
/// token over set A minus the mean over set B.
/// Usage: `dart run tool/control_vector_extract.dart <bundle> <a.txt> <b.txt>`
///        `--out=vector.safetensors [--min-separation=2] [--system=prompt]`
///        `[--threads=4] [--gpu]`
/// Each non-empty line of a prompt file, or each `prompt` of a `.jsonl`
/// file, is one user turn, rendered through the chat template with the
/// assistant's generation prompt (thinking off).
/// Runs on the CPU kernels, or with --gpu on the GPU (Metal or Vulkan).
///
/// Prints per layer the direction's norm, the mean residual norm and their
/// ratio, each set's mean and sample standard deviation of its held-out
/// projections (each prompt on the direction built without it), d'
/// (LayerSeparation: the gap between those means over their pooled
/// deviation; near zero or negative when the direction does not generalise)
/// and the cosine with the previous layer's direction. The file keeps the
/// directions from all prompts. Layers with no direction, or with d' below
/// --min-separation (default 2; 0 keeps every layer with a direction), are
/// left out of the file: in project mode a direction is normalised, so a
/// layer where the sets barely differ would lose a full-strength component
/// along what is mostly noise. Fails, writing nothing, if no layer is left.
/// Ends with a suggested --layers for tool/control_vector_probe.dart: the
/// layers from which d' stays at or above 3 to the last, or a note that no
/// such trailing range exists.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';

const _usage =
    'usage: dart run tool/control_vector_extract.dart <bundle> <a.txt> '
    '<b.txt> --out=vector.safetensors [--min-separation=2] [--system=prompt] '
    '[--threads=4] [--gpu]';

/// The d' from which on the suggested layer range starts.
const _suggestedSeparation = 3.0;

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
  final unknown = options.keys.toSet().difference({
    'out',
    'system',
    'threads',
    'min-separation',
  });
  final threads = int.tryParse(options['threads'] ?? '4');
  final minSeparation = double.tryParse(options['min-separation'] ?? '2');
  if (positional.length != 3 ||
      options['out'] == null ||
      unknown.isNotEmpty ||
      threads == null ||
      threads <= 0 ||
      minSeparation == null ||
      !minSeparation.isFinite ||
      minSeparation < 0) {
    stderr.writeln(_usage);
    exitCode = 64;
    return;
  }
  final [bundle, pathA, pathB] = positional;
  final setA = _prompts(pathA), setB = _prompts(pathB);
  if (setA.isEmpty || setB.isEmpty) {
    stderr.writeln('both prompt files need at least one non-empty line');
    exitCode = 64;
    return;
  }
  if (minSeparation > 0 && (setA.length < 2 || setB.length < 2)) {
    stderr.writeln(
      "d' needs at least two prompts per set; add prompts or pass "
      '--min-separation=0',
    );
    exitCode = 64;
    return;
  }

  NativeKernels.load(null, threads: threads);
  final tokenizer = Lfm3Tokenizer.load(bundle);
  final model = await _load(bundle, gpu: gpu);
  try {
    final c = model.config;
    final h = c.hiddenSize;
    final system = options['system'];

    /// The residual leaving each layer at each prompt's last token, by
    /// layer then prompt. Kept in memory: prompts × layers × hidden floats
    /// (about 200 KB per prompt for LFM3-1.7B).
    Future<List<List<Float32List>>> residuals(List<String> prompts) async {
      final out = [for (var l = 0; l < c.numLayers; l++) <Float32List>[]];
      for (final (i, prompt) in prompts.indexed) {
        final ids = tokenizer.encode(
          renderChat([
            if (system != null) {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt},
          ], enableThinking: false),
        );
        if (ids.length > model.maxSeq) {
          throw StateError('prompt ${i + 1} has ${ids.length} tokens');
        }
        await model.reset();
        await model.forward(
          ids,
          computeLogits: false,
          capture: (slot, first, rows, count) {
            if (slot == 0 || first + count != ids.length) return;
            final row = (count - 1) * h;
            out[slot - 1].add(Float32List.fromList(rows.sublist(row, row + h)));
          },
        );
        stderr.write('\r${i + 1}/${prompts.length}');
      }
      stderr.writeln();
      return out;
    }

    stderr.writeln('set A: ${setA.length} prompts');
    final rowsA = await residuals(setA);
    stderr.writeln('set B: ${setB.length} prompts');
    final rowsB = await residuals(setB);
    final layers = [
      for (var l = 0; l < c.numLayers; l++)
        LayerSeparation.measure(rowsA[l], rowsB[l]),
    ];

    print(
      'layer  |direction|  mean |residual|  ratio  held-out A (sd)'
      '  held-out B (sd)      d\'  cos(previous)',
    );
    String f3(double? v) => v == null ? '-' : v.toStringAsFixed(3);
    for (var l = 0; l < c.numLayers; l++) {
      final s = layers[l];
      final all = [...rowsA[l], ...rowsB[l]];
      final residual = all.map(_norm).reduce((a, b) => a + b) / all.length;
      final previous = l == 0 ? null : layers[l - 1];
      final cos = s.norm == 0 || previous == null || previous.norm == 0
          ? '-'
          : (_dot(s.direction, previous.direction) / (s.norm * previous.norm))
                .toStringAsFixed(3);
      final dPrime = s.separation?.toStringAsFixed(2) ?? '-';
      print(
        '${'$l'.padLeft(5)}  ${s.norm.toStringAsFixed(3).padLeft(11)}  '
        '${residual.toStringAsFixed(3).padLeft(15)}  '
        '${(s.norm / residual).toStringAsFixed(3).padLeft(5)}  '
        '${'${f3(s.meanA)} (${f3(s.sdA)})'.padLeft(15)}  '
        '${'${f3(s.meanB)} (${f3(s.sdB)})'.padLeft(15)}  '
        '${dPrime.padLeft(6)}  $cos',
      );
    }

    // A layer whose output both sets share at the last token (a short
    // convolution that only sees the identical generation prompt, say) has
    // no direction, and one where the sets overlap has a direction that is
    // mostly noise: project mode would remove a full-strength component
    // along it from every token. Both stay out of the file.
    final zero = [
      for (var l = 0; l < c.numLayers; l++)
        if (layers[l].norm == 0) l,
    ];
    final weak = [
      for (var l = 0; l < c.numLayers; l++)
        if (layers[l].norm > 0 && !layers[l].passes(minSeparation)) l,
    ];
    if (zero.isNotEmpty) print('no direction (left out): ${zero.join(',')}');
    if (weak.isNotEmpty) {
      print(
        "d' below $minSeparation (left out): "
        '${[for (final l in weak) '$l (${layers[l].separation?.toStringAsFixed(2) ?? '-'})'].join(', ')}',
      );
    }
    final kept = {
      for (var l = 0; l < c.numLayers; l++)
        if (layers[l].passes(minSeparation)) l: layers[l].direction,
    };
    if (kept.isEmpty) {
      stderr.writeln(
        "no layer separates the sets with d' >= $minSeparation; nothing "
        'written (lower --min-separation, or use more or more distinct '
        'prompts)',
      );
      exitCode = 1;
      return;
    }
    ControlVector(kept, hiddenSize: h).save(options['out']!);
    final first = LayerSeparation.passingFrom(layers, _suggestedSeparation);
    if (first != null) {
      print(
        "suggested: --layers=$first- (d' stays at or above "
        '$_suggestedSeparation from layer $first on; a starting point, '
        'not a guarantee)',
      );
    } else {
      print(
        "no suggestion: no trailing range of layers keeps d' at or above "
        '$_suggestedSeparation; pick a range from the d\' column',
      );
    }
    print('wrote ${kept.length} layers to ${options['out']}');
  } finally {
    await model.close();
  }
}

/// The prompts of [path]: one per line in a text file, or the `prompt`
/// fields of a `.jsonl` file (readEvalRows).
List<String> _prompts(String path) => [
  for (final row in readEvalRows(path)) row.prompt,
];

/// The bundle's Q4 or Q8 weights, on the CPU kernels or with [gpu] on the
/// GPU (Metal or Vulkan).
Future<Lfm3Model> _load(String bundle, {required bool gpu}) {
  final weights = [WeightFormat.q4, WeightFormat.q8].firstWhere(
    (w) => File('$bundle/lfm3-${w.name}.cache').existsSync(),
    orElse: () => throw ArgumentError('$bundle has no lfm3-q4/q8.cache'),
  );
  return Lfm3Model.load(
    bundle,
    maxSeq: 2048,
    weights: weights,
    q8CachePath: '$bundle/lfm3-${weights.name}.cache',
    gpu: gpu,
  );
}

double _dot(Float32List a, Float32List b) {
  var sum = 0.0;
  for (var j = 0; j < a.length; j++) {
    sum += a[j] * b[j];
  }
  return sum;
}

double _norm(Float32List v) => math.sqrt(_dot(v, v));
