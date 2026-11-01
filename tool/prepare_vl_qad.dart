/// Import the official VL QAD GGUF using the existing text importer, and audit
/// that every Q4 matrix's nibbles/scales is preserved. Validation-only layout.
library;

import 'dart:convert';
import 'dart:io';
import 'package:crypto/crypto.dart';
import 'package:dianoia/lab.dart';

Future<void> main(List<String> args) async {
  final ggufPath = args[0],
      vision = args[1],
      backbone = args[2],
      evidence = args[3];
  NativeKernels.load(null, threads: 4);
  final text = Directory('$vision.text-config')..createSync();
  final config =
      jsonDecode(File('$vision/config.json').readAsStringSync()) as Map;
  File(
    '${text.path}/config.json',
  ).writeAsStringSync(jsonEncode(config['text_config']));
  File('$vision/tokenizer.json').copySync('${text.path}/tokenizer.json');
  importGguf(ggufPath, text.path, backbone);
  File('$backbone/vl_backbone.json').writeAsStringSync(
    jsonEncode({
      'version': 1,
      'source_key': Q8Cache.keyFor(vision),
      'backbone_key': Q8Cache.keyFor(backbone),
    }),
  );
  final gguf = GgufFile.open(ggufPath);
  final cache = Q8Cache.open(
    '$backbone/lfm3-q4.cache',
    Q8Cache.keyFor(backbone, weights: 'q4'),
  )!;
  final audit = <String, Object?>{};
  for (final t in gguf.tensors.values.where((t) => t.type == GgufType.q4_0)) {
    final match = RegExp(r'^blk\.(\d+)\.(.+)\.weight$').firstMatch(t.name)!;
    final suffix = <String, String>{
      'ffn_gate': 'feed_forward.w1',
      'ffn_up': 'feed_forward.w3',
      'ffn_down': 'feed_forward.w2',
      'shortconv.in_proj': 'conv.in_proj.weight',
      'shortconv.out_proj': 'conv.out_proj.weight',
      'attn_q': 'self_attn.q_proj',
      'attn_k': 'self_attn.k_proj',
      'attn_v': 'self_attn.v_proj',
      'attn_gate': 'self_attn.gate_proj',
      'attn_output': 'self_attn.o_proj',
    }[match[2]]!;
    final name = 'model.layers.${match[1]}.$suffix';
    final source = splitQ4_0(gguf.bytes(t), t.dims[1], t.dims[0]);
    final target = cache.matrix(name, t.dims[1], t.dims[0])!;
    final qs = sha256.convert(source.qsBytes).toString();
    final scales = sha256.convert(source.scaleBytes).toString();
    if (qs != sha256.convert(target.qsBytes).toString() ||
        scales != sha256.convert(target.scaleBytes).toString()) {
      throw StateError('import changed Q4 payload: ${t.name}');
    }
    audit[t.name] = {
      'cache_name': name,
      'nibbles_sha256': qs,
      'scales_sha256': scales,
    };
  }
  cache.close();
  final files = <String, Object?>{};
  for (final root in [vision, backbone]) {
    for (final file in Directory(root).listSync().whereType<File>()) {
      files[file.path] = {
        'bytes': file.lengthSync(),
        'sha256': (await sha256.bind(file.openRead()).first).toString(),
      };
    }
  }
  File(evidence).writeAsStringSync(
    '${const JsonEncoder.withIndent('  ').convert({'q4_matrices_byte_identical': audit.length, 'q4_payloads': audit, 'embedding_policy': 'GGUF Q6_K embedding is converted to Q8 for tied head AND input lookup by the existing dianoia GGUF importer.', 'files': files})}\n',
  );
  stdout.writeln('Verified ${audit.length} unchanged Q4 matrices.');
}
