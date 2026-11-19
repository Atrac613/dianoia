// A tiny synthetic Qwen3 checkpoint, imported and run in float32: prompt
// reuse across a diverging conversation gives the logits of a fresh
// prefill, and keeps every common token since the model has no conv state.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/synthetic_vision.dart';

const _h = 32, _hd = 8, _heads = 4, _kvHeads = 2, _layers = 2, _vocab = 64;

String _writeQwen3(Directory root) {
  final hf = Directory('${root.path}/hf')..createSync();
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
  final out = '${root.path}/model';
  importHfCheckpoint(hf.path, out);
  return out;
}

void main() {
  late Directory root;
  late String dir;
  late TextModel model;

  setUpAll(() async {
    root = Directory.systemTemp.createTempSync('dianoia-qwen3-model-');
    dir = _writeQwen3(root);
    model = await TextModel.load(dir, maxSeq: 64, maxChunk: 4);
  });
  tearDownAll(() async {
    await model.close();
    root.deleteSync(recursive: true);
  });

  test('loads as ungated attention without sandwich norms', () {
    expect(model.config.architecture, ModelArchitecture.qwen3);
    expect(model.config.attentionOutputGate, isFalse);
    expect(model.config.sandwichNorm, isFalse);
  });

  test('a diverging prompt reuses every common token', () async {
    final first = [5, 9, 12, 7, 30, 31, 2, 8, 40, 41, 42];
    final second = [5, 9, 12, 7, 30, 31, 3, 8, 40, 50];
    await model.reset();
    await model.prefill(first, checkpoints: [2]);
    expect(await model.prefill(second, checkpoints: [2]), 6);
    final reused = Float32List.fromList(model.logits);

    // The same chunks from scratch: the first prefill's up to the
    // divergence, then the rest.
    await model.reset();
    await model.forward(second.sublist(0, 2), computeLogits: false);
    await model.forward(second.sublist(2, 6), computeLogits: false);
    await model.forward(second.sublist(6));
    expect(model.logits, orderedEquals(reused));
    expect(reused.any((v) => v != 0), isTrue);
  });
}
