import 'dart:convert';
import 'dart:io';

import 'synthetic_tokenizer.dart';
import 'synthetic_vision.dart';

const _h = 32, _hd = 8, _heads = 4, _kvHeads = 2, _layers = 2, _vocab = 272;

/// Writes a tiny random Qwen3 checkpoint, in Hugging Face's layout and
/// with bf16 weights, to [dir].
void writeSyntheticQwen3(String dir) {
  final hf = Directory(dir)..createSync(recursive: true);
  File('${hf.path}/config.json').writeAsStringSync(
    jsonEncode({
      'model_type': 'qwen3',
      'hidden_act': 'silu',
      'hidden_size': _h,
      'intermediate_size': 2 * _h,
      'num_attention_heads': _heads,
      'num_key_value_heads': _kvHeads,
      'head_dim': _hd,
      'num_hidden_layers': _layers,
      'vocab_size': _vocab,
      'rms_norm_eps': 1e-6,
      'rope_theta': 1000000,
      'rope_scaling': null,
      'tie_word_embeddings': true,
      'attention_bias': false,
      'use_sliding_window': false,
      'eos_token_id': 1,
    }),
  );
  final shapes = <String, Object?>{
    'model.embed_tokens.weight': [_vocab, _h],
    'model.norm.weight': [_h],
  };
  for (var i = 0; i < _layers; i++) {
    final p = 'model.layers.$i.';
    shapes['${p}input_layernorm.weight'] = [_h];
    shapes['${p}post_attention_layernorm.weight'] = [_h];
    shapes['${p}self_attn.q_proj.weight'] = [_heads * _hd, _h];
    shapes['${p}self_attn.k_proj.weight'] = [_kvHeads * _hd, _h];
    shapes['${p}self_attn.v_proj.weight'] = [_kvHeads * _hd, _h];
    shapes['${p}self_attn.o_proj.weight'] = [_h, _heads * _hd];
    shapes['${p}self_attn.q_norm.weight'] = [_hd];
    shapes['${p}self_attn.k_norm.weight'] = [_hd];
    shapes['${p}mlp.gate_proj.weight'] = [2 * _h, _h];
    shapes['${p}mlp.up_proj.weight'] = [2 * _h, _h];
    shapes['${p}mlp.down_proj.weight'] = [_h, 2 * _h];
  }
  writeSyntheticVisionWeights(
    '${hf.path}/model.safetensors',
    shapes,
    bf16: true,
  );
  writeSyntheticTokenizer(hf.path);
  File('${hf.path}/LICENSE').writeAsStringSync('Synthetic test weights.\n');
}
