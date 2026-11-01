import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';

import 'synthetic_tokenizer.dart';
import 'synthetic_vision.dart';

/// Builds a tiny complete VL checkpoint and its Q8 language bundle at runtime.
Future<void> prepareVisionFixture(Directory root, {int hiddenSize = 32}) async {
  final h = hiddenSize, hd = hiddenSize ~/ 4, kv = hiddenSize ~/ 2;
  final source = Directory('${root.path}/model')..createSync();
  final oracle =
      jsonDecode(
            File('test/fixtures/vision/encoder_oracle.json').readAsStringSync(),
          )
          as Map<String, Object?>;
  final config = oracle['config'] as Map<String, Object?>;
  config.addAll({
    'image_token_id': 263,
    'video_token_id': 264,
    'vision_start_token_id': 261,
    'vision_end_token_id': 262,
  });
  final text = config['text_config'] as Map<String, Object?>;
  text.addAll({
    'hidden_size': h,
    'intermediate_size': 2 * h,
    'num_attention_heads': 4,
    'num_key_value_heads': 2,
    'head_dim': hd,
    'vocab_size': 272,
    'eos_token_id': 258,
    'num_hidden_layers': 3,
    'layer_types': ['conv', 'full_attention', 'sliding_attention'],
  });
  File('${source.path}/config.json').writeAsStringSync(jsonEncode(config));
  writeSyntheticTokenizer(source.path);
  File(
    '${source.path}/LICENSE',
  ).writeAsStringSync('Synthetic test weights generated at runtime.\n');
  final processor =
      jsonDecode(
            File(
              'test/fixtures/vision/processor_config.json',
            ).readAsStringSync(),
          )
          as Map<String, Object?>;
  (processor['image_processor'] as Map<String, Object?>).addAll({
    'patch_size': 2,
    'size': {'shortest_edge': 16, 'longest_edge': 4096},
  });
  File(
    '${source.path}/processor_config.json',
  ).writeAsStringSync(jsonEncode(processor));
  final shapes = oracle['shapes'] as Map<String, Object?>;
  shapes['model.multi_modal_projector.linear_2.weight'] = [h, 12];
  shapes['model.multi_modal_projector.linear_2.bias'] = [h];
  final language = <String, List<int>>{
    'model.embed_tokens.weight': [272, h],
    'model.embedding_norm.weight': [h],
  };
  for (var i = 0; i < 3; i++) {
    final p = 'model.layers.$i.';
    for (final norm in [
      'operator_norm',
      'post_operator_norm',
      'ffn_norm',
      'post_ffn_norm',
    ]) {
      language['$p$norm.weight'] = [h];
    }
    language['${p}feed_forward.w1.weight'] = [2 * h, h];
    language['${p}feed_forward.w3.weight'] = [2 * h, h];
    language['${p}feed_forward.w2.weight'] = [h, 2 * h];
    if (i == 0) {
      language['${p}conv.in_proj.weight'] = [3 * h, h];
      language['${p}conv.out_proj.weight'] = [h, h];
      language['${p}conv.conv.weight'] = [h, 1, 7];
    } else {
      final a = '${p}self_attn.';
      language['${a}q_proj.weight'] = [h, h];
      language['${a}k_proj.weight'] = [kv, h];
      language['${a}v_proj.weight'] = [kv, h];
      language['${a}o_proj.weight'] = [h, h];
      language['${a}gate_proj.weight'] = [4, h];
      language['${a}q_norm.weight'] = [hd];
      language['${a}k_norm.weight'] = [hd];
    }
  }
  for (final e in language.entries) {
    shapes[Lfm3VlConfig.languageTensorName(e.key)] = e.value;
  }
  writeSyntheticVisionWeights(
    '${source.path}/model.safetensors',
    shapes,
    bf16: true,
    suppressSpecialEmbeddings: true,
  );
  final out = '${root.path}/backbone';
  importVlBackbone(source.path, out);
  NativeKernels.load(null, threads: 2);
  final model = await Lfm3Model.load(
    out,
    maxSeq: 512,
    weights: WeightFormat.q8,
    q8CachePath: '$out/lfm3-q8.cache',
  );
  await model.close();
}

bool fixtureBackboneIsMapped(Directory root) =>
    MappedFile.isMapped('${root.path}/backbone/lfm3-q8.cache');
