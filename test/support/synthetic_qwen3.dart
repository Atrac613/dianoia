import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';

import 'gguf_fixture.dart';
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

/// GGUF names of the Hugging Face tensors in a Qwen3 layer.
const _ggufNames = {
  'input_layernorm': 'attn_norm',
  'post_attention_layernorm': 'ffn_norm',
  'self_attn.q_proj': 'attn_q',
  'self_attn.k_proj': 'attn_k',
  'self_attn.v_proj': 'attn_v',
  'self_attn.o_proj': 'attn_output',
  'self_attn.q_norm': 'attn_q_norm',
  'self_attn.k_norm': 'attn_k_norm',
  'mlp.gate_proj': 'ffn_gate',
  'mlp.up_proj': 'ffn_up',
  'mlp.down_proj': 'ffn_down',
};

/// Writes the checkpoint [writeSyntheticQwen3] made in [hfDir] to [path] as
/// llama.cpp lays out a Qwen3 Q8_0 GGUF: matrices in Q8_0, vectors in F32.
void writeSyntheticQwen3Gguf(String hfDir, String path) {
  final st = SafetensorsFile.open('$hfDir/model.safetensors');
  try {
    final tensors = <(String, List<int>, GgufType, Uint8List)>[];
    for (final MapEntry(key: name, value: info) in st.tensors.entries) {
      final values = st.readF32(name, info.shape);
      final layer = RegExp(
        r'^model\.layers\.(\d+)\.(.+)\.weight$',
      ).firstMatch(name);
      final ggufName = switch (name) {
        'model.embed_tokens.weight' => 'token_embd.weight',
        'model.norm.weight' => 'output_norm.weight',
        _ => 'blk.${layer![1]}.${_ggufNames[layer[2]]!}.weight',
      };
      tensors.add(
        info.shape.length == 1
            ? (ggufName, info.shape, GgufType.f32, values.buffer.asUint8List())
            : (ggufName, info.shape, GgufType.q8_0, quantizeQ8_0(values)),
      );
    }
    writeTestGguf(path, {
      'general.architecture': 'qwen3',
      'qwen3.block_count': _layers,
      'qwen3.embedding_length': _h,
      'qwen3.attention.head_count': _heads,
      'qwen3.attention.head_count_kv': _kvHeads,
      'qwen3.attention.key_length': _hd,
      'qwen3.rope.freq_base': 1000000.0,
      'qwen3.attention.layer_norm_rms_epsilon': 1e-6,
    }, tensors);
  } finally {
    st.close();
  }
}

/// ggml's quantize_row_q8_0, with the half scale rounded toward zero.
Uint8List quantizeQ8_0(Float32List values) {
  final out = Uint8List(values.length ~/ 32 * 34);
  final view = ByteData.sublistView(out);
  for (var b = 0; b < values.length ~/ 32; b++) {
    var max = 0.0;
    for (var k = 0; k < 32; k++) {
      max = values[32 * b + k].abs() > max ? values[32 * b + k].abs() : max;
    }
    final half = _toHalf(max / 127);
    view.setUint16(34 * b, half, Endian.little);
    final d = halfToDouble(half);
    for (var k = 0; k < 32; k++) {
      final q = d == 0 ? 0 : (values[32 * b + k] / d).round().clamp(-127, 127);
      view.setInt8(34 * b + 2 + k, q);
    }
  }
  return out;
}

int _toHalf(double v) {
  final bits = (ByteData(4)..setFloat32(0, v)).getUint32(0);
  final exponent = (bits >> 23 & 0xff) - 127 + 15;
  if (exponent <= 0) return 0;
  return (bits >> 16 & 0x8000) | exponent << 10 | (bits >> 13 & 0x3ff);
}
