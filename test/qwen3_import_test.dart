import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:dianoia/src/model_loading.dart'
    show autoPrefersCpu, defaultContextLength;
import 'package:test/test.dart';

/// Qwen3-0.6B's config.json, shortened to two layers.
Map<String, Object?> _qwen3Config() => {
  'architectures': ['Qwen3ForCausalLM'],
  'attention_bias': false,
  'bos_token_id': 151643,
  'eos_token_id': 151645,
  'head_dim': 128,
  'hidden_act': 'silu',
  'hidden_size': 1024,
  'intermediate_size': 3072,
  'model_type': 'qwen3',
  'num_attention_heads': 16,
  'num_hidden_layers': 2,
  'num_key_value_heads': 8,
  'rms_norm_eps': 1e-06,
  'rope_scaling': null,
  'rope_theta': 1000000,
  'sliding_window': null,
  'tie_word_embeddings': true,
  'use_sliding_window': false,
  'vocab_size': 151936,
};

void main() {
  test('auto runs Qwen3 on the CPU on Android only', () {
    final qwen3 = TextModelConfig.fromJson(_qwen3Config());
    final vision =
        jsonDecode(File('test/fixtures/vision/config.json').readAsStringSync())
            as Map<String, Object?>;
    final lfm3 = TextModelConfig.fromJson(
      vision['text_config'] as Map<String, Object?>,
    );
    expect(autoPrefersCpu(qwen3, android: true), isTrue);
    expect(autoPrefersCpu(qwen3, android: false), isFalse);
    expect(autoPrefersCpu(lfm3, android: true), isFalse);
  });

  test('Qwen3 gets 4096 tokens of context by default, LFM3 2048', () {
    final vision =
        jsonDecode(File('test/fixtures/vision/config.json').readAsStringSync())
            as Map<String, Object?>;
    final lfm3 = TextModelConfig.fromJson(
      vision['text_config'] as Map<String, Object?>,
    );
    expect(
      defaultContextLength(TextModelConfig.fromJson(_qwen3Config())),
      4096,
    );
    expect(defaultContextLength(lfm3), 2048);
    // Vulkan keeps Qwen3's cache in float32, twice the size.
    expect(
      defaultContextLength(
        TextModelConfig.fromJson(_qwen3Config()),
        halfCache: false,
      ),
      2048,
    );
  });

  group('Qwen3 config', () {
    test('every layer is ungated full attention without sandwich norms', () {
      final c = TextModelConfig.fromJson(_qwen3Config());
      expect(c.architecture, ModelArchitecture.qwen3);
      expect(c.layerTypes, [LayerType.fullAttention, LayerType.fullAttention]);
      expect(c.attentionOutputGate, isFalse);
      expect(c.sandwichNorm, isFalse);
      expect(c.convKernel, 0);
      expect(c.fullRope.theta, 1000000);
      expect(c.fullRope.partialRotaryFactor, 1);
      expect(c.normEps, 1e-6);
      expect(c.eosTokenIds, [151645]);
      expect(c.kvDim, 1024);
    });

    test('load adds generation_config.json end tokens', () {
      final dir = Directory.systemTemp.createTempSync('dianoia-qwen3-cfg-');
      addTearDown(() => dir.deleteSync(recursive: true));
      File(
        '${dir.path}/config.json',
      ).writeAsStringSync(jsonEncode(_qwen3Config()));
      expect(TextModelConfig.load(dir.path).eosTokenIds, [151645]);
      File('${dir.path}/generation_config.json').writeAsStringSync(
        jsonEncode({
          'eos_token_id': [151645, 151643],
        }),
      );
      final c = TextModelConfig.load(dir.path);
      expect(c.eosTokenIds, [151645, 151643]);
      expect(c.eosTokenId, 151645);
      expect(c.isEos(151643), isTrue);
      expect(c.isEos(151644), isFalse);
    });

    test('end tokens may come from generation_config.json alone', () {
      final json = _qwen3Config()..remove('eos_token_id');
      expect(
        () => TextModelConfig.fromJson(json),
        throwsA(isA<FormatException>()),
      );
      expect(
        TextModelConfig.fromJson(
          json,
          extraEosTokenIds: [151645, 151643],
        ).eosTokenIds,
        [151645, 151643],
      );
    });

    test('rope_theta may sit in rope_parameters', () {
      final json = _qwen3Config()
        ..remove('rope_theta')
        ..['rope_parameters'] = {'rope_type': 'default', 'rope_theta': 5e5};
      expect(TextModelConfig.fromJson(json).fullRope.theta, 5e5);
    });

    for (final (key, value) in [
      ('tie_word_embeddings', false),
      ('attention_bias', true),
      ('use_sliding_window', true),
      ('rope_scaling', {'rope_type': 'yarn', 'factor': 4.0}),
      ('hidden_act', 'gelu'),
    ]) {
      test('$key: $value is rejected', () {
        final json = _qwen3Config()..[key] = value;
        expect(
          () => TextModelConfig.fromJson(json),
          throwsA(isA<FormatException>()),
        );
      });
    }

    test('an unknown model_type is rejected', () {
      final json = _qwen3Config()..['model_type'] = 'llama';
      expect(
        () => TextModelConfig.fromJson(json),
        throwsA(isA<FormatException>()),
      );
    });
  });

  group('qwen3TensorName', () {
    test('maps to the engine names', () {
      const cases = {
        'model.embed_tokens.weight': 'model.embed_tokens.weight',
        'model.norm.weight': 'model.embedding_norm.weight',
        'model.layers.3.input_layernorm.weight':
            'model.layers.3.operator_norm.weight',
        'model.layers.3.post_attention_layernorm.weight':
            'model.layers.3.ffn_norm.weight',
        'model.layers.3.mlp.gate_proj.weight':
            'model.layers.3.feed_forward.w1.weight',
        'model.layers.3.mlp.up_proj.weight':
            'model.layers.3.feed_forward.w3.weight',
        'model.layers.3.mlp.down_proj.weight':
            'model.layers.3.feed_forward.w2.weight',
        'model.layers.3.self_attn.q_proj.weight':
            'model.layers.3.self_attn.q_proj.weight',
        'model.layers.3.self_attn.k_norm.weight':
            'model.layers.3.self_attn.k_norm.weight',
        'model.layers.3.self_attn.o_proj.weight':
            'model.layers.3.self_attn.o_proj.weight',
      };
      cases.forEach((from, to) => expect(qwen3TensorName(from), to));
    });

    test('rejects tensors the engine would not read', () {
      for (final name in [
        'model.layers.0.self_attn.q_proj.bias',
        'model.layers.0.mlp.experts.0.up_proj.weight',
        'model.rotary_emb.inv_freq',
      ]) {
        expect(() => qwen3TensorName(name), throwsFormatException);
      }
    });
  });

  group('importHfCheckpoint', () {
    late Directory root;
    setUp(() => root = Directory.systemTemp.createTempSync('dianoia-hf-'));
    tearDown(() => root.deleteSync(recursive: true));

    Float32List values(int n, double base) =>
        Float32List.fromList([for (var i = 0; i < n; i++) base + i]);

    test('merges shards, renames tensors and drops a tied lm_head', () {
      final hf = Directory('${root.path}/hf')..createSync();
      File(
        '${hf.path}/config.json',
      ).writeAsStringSync(jsonEncode(_qwen3Config()));
      File('${hf.path}/tokenizer.json').writeAsStringSync('{}');
      writeSafetensorsF32('${hf.path}/model-00001-of-00002.safetensors', {
        'model.embed_tokens.weight': ([2, 3], values(6, 0)),
        'model.layers.0.input_layernorm.weight': ([3], values(3, 10)),
      });
      writeSafetensorsF32('${hf.path}/model-00002-of-00002.safetensors', {
        'model.norm.weight': ([3], values(3, 20)),
        'lm_head.weight': ([2, 3], values(6, 0)),
      });
      File('${hf.path}/model.safetensors.index.json').writeAsStringSync(
        jsonEncode({
          'weight_map': {
            'model.embed_tokens.weight': 'model-00001-of-00002.safetensors',
            'model.layers.0.input_layernorm.weight':
                'model-00001-of-00002.safetensors',
            'model.norm.weight': 'model-00002-of-00002.safetensors',
            'lm_head.weight': 'model-00002-of-00002.safetensors',
          },
        }),
      );

      final out = '${root.path}/out';
      importHfCheckpoint(hf.path, out);

      final st = SafetensorsFile.open('$out/model.safetensors');
      addTearDown(st.close);
      expect(st.tensors.keys, {
        'model.embed_tokens.weight',
        'model.layers.0.operator_norm.weight',
        'model.embedding_norm.weight',
      });
      expect(
        st.readF32('model.layers.0.operator_norm.weight', [3]),
        values(3, 10),
      );
      expect(st.readF32('model.embedding_norm.weight', [3]), values(3, 20));
      expect(File('$out/config.json').existsSync(), isTrue);
      expect(File('$out/tokenizer.json').existsSync(), isTrue);
      expect(() => importHfCheckpoint(hf.path, out), throwsStateError);
    });
  });
}
