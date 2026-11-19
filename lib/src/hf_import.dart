/// Turning a Hugging Face text checkpoint of another family into this
/// engine's model directory.
///
/// The engine reads tensors under LFM3's names (model.dart), so a Qwen3
/// checkpoint's tensors are renamed, and its shards merged, into one
/// `model.safetensors` with the bytes unchanged:
///
///     input_layernorm           -> operator_norm
///     post_attention_layernorm  -> ffn_norm
///     mlp.gate_proj / up_proj / down_proj -> feed_forward.w1 / w3 / w2
///     model.norm                -> model.embedding_norm
///
/// Attention projections, q_norm, k_norm and the embedding already share
/// LFM3's names. The result is a model directory like an LFM3 checkpoint's:
/// `run --weights=q8|q4` builds its quantized cache and `pack` makes the
/// bundle.
library;

import 'dart:convert';
import 'dart:io';

import 'config.dart';
import 'safetensors.dart';

/// The engine's name for the Qwen3 tensor [name].
String qwen3TensorName(String name) {
  if (name == 'model.norm.weight') return 'model.embedding_norm.weight';
  if (name == 'model.embed_tokens.weight') return name;
  final layer = RegExp(r'^model\.layers\.(\d+)\.(.+)$').firstMatch(name);
  if (layer == null) {
    throw FormatException('unexpected Qwen3 tensor $name');
  }
  final rest = switch (layer[2]!) {
    'input_layernorm.weight' => 'operator_norm.weight',
    'post_attention_layernorm.weight' => 'ffn_norm.weight',
    'mlp.gate_proj.weight' => 'feed_forward.w1.weight',
    'mlp.up_proj.weight' => 'feed_forward.w3.weight',
    'mlp.down_proj.weight' => 'feed_forward.w2.weight',
    final attention
        when RegExp(
          r'^self_attn\.([qkvo]_proj|[qk]_norm)\.weight$',
        ).hasMatch(attention) =>
      attention,
    _ => throw FormatException('unexpected Qwen3 tensor $name'),
  };
  return 'model.layers.${layer[1]}.$rest';
}

/// Files copied unchanged when they exist.
const _sideFiles = [
  'config.json',
  'generation_config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'LICENSE',
];

/// Writes the engine's model directory for the checkpoint in [hfDir] to
/// [outDir], which must not hold a model.safetensors yet.
void importHfCheckpoint(String hfDir, String outDir) {
  final config = TextModelConfig.load(hfDir);
  final rename = switch (config.architecture) {
    ModelArchitecture.qwen3 => qwen3TensorName,
    ModelArchitecture.lfm3 => throw ArgumentError.value(
      hfDir,
      'hfDir',
      'is an LFM3 checkpoint, which the engine reads as it is',
    ),
  };
  final target = File('$outDir/model.safetensors');
  if (target.existsSync()) {
    throw StateError('${target.path} exists');
  }
  final sources = [
    for (final name in _shardNames(hfDir)) SafetensorsFile.open('$hfDir/$name'),
  ];
  try {
    Directory(outDir).createSync(recursive: true);
    // A tied checkpoint may still carry lm_head; the engine uses the
    // embedding instead.
    SafetensorsFile.writeMerged(
      target.path,
      sources,
      keep: (name, _) => name != 'lm_head.weight',
      rename: rename,
    );
  } finally {
    for (final source in sources) {
      source.close();
    }
  }
  for (final name in _sideFiles) {
    final file = File('$hfDir/$name');
    if (file.existsSync()) file.copySync('$outDir/$name');
  }
}

/// The checkpoint's safetensors files: model.safetensors, or the shards
/// its index names, in order.
List<String> _shardNames(String hfDir) {
  if (File('$hfDir/model.safetensors').existsSync()) {
    return const ['model.safetensors'];
  }
  final index = File('$hfDir/model.safetensors.index.json');
  if (!index.existsSync()) {
    throw ArgumentError.value(hfDir, 'hfDir', 'holds no safetensors weights');
  }
  final map =
      (jsonDecode(index.readAsStringSync()) as Map)['weight_map'] as Map;
  return {...map.values.cast<String>()}.toList()..sort();
}
