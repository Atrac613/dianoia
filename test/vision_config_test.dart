import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

Map<String, Object?> fixture() =>
    jsonDecode(File('test/fixtures/vision/config.json').readAsStringSync())
        as Map<String, Object?>;

void main() {
  test(
    'official VL config exposes language, vision and projector dimensions',
    () {
      final c = Lfm3VlConfig.load('test/fixtures/vision');
      expect(c.text.hiddenSize, 2048);
      expect(c.text.numLayers, 24);
      expect(c.vision.hiddenSize, 1024);
      expect(c.vision.numLayers, 14);
      expect(c.vision.headDim, 64);
      expect(
        c.vision.layerTypes.where((t) => t == VisionLayerType.conv),
        hasLength(8),
      );
      expect(c.projectorInputSize, 4096);
      expect(c.projectorHiddenSize, 4096);
      expect(c.imageTokenId, 125030);
      expect(c.vision.patchSize, 16);
      expect(c.vision.temporalPatchSize, 2);
      expect(c.vision.spatialMergeSize, 2);
    },
  );

  test('text loader continues to reject a VL checkpoint', () {
    expect(
      () => Lfm3Config.load('test/fixtures/vision'),
      throwsFormatException,
    );
  });

  test(
    'language tensor mapping preserves layer names and rejects other towers',
    () {
      expect(
        Lfm3VlConfig.languageTensorName('model.embed_tokens.weight'),
        'model.language_model.embed_tokens.weight',
      );
      expect(
        Lfm3VlConfig.languageTensorName('model.layers.0.operator_norm.weight'),
        'model.language_model.layers.0.operator_norm.weight',
      );
      expect(
        () => Lfm3VlConfig.languageTensorName('model.vision_tower.norm.weight'),
        throwsArgumentError,
      );
    },
  );

  test(
    'unsupported projector options and invalid special tokens fail early',
    () {
      for (final change in <Map<String, Object?>>[
        {'model_type': 'lfm3_omni'},
        {'projector_bias': false},
        {'projector_hidden_act': 'silu'},
        {'downsample_factor': 3},
        {'image_token_id': 128000},
        {'image_token_id': 125029},
        {'projector_hidden_size': 0},
        {'projector_layer_norm_eps': double.nan},
      ]) {
        expect(
          () => Lfm3VlConfig.fromJson(fixture()..addAll(change)),
          throwsFormatException,
        );
      }
    },
  );

  test('invalid vision dimensions and operators fail early', () {
    for (final change in <Map<String, Object?>>[
      {'num_attention_heads': 15},
      {'num_key_value_heads': 3},
      {'hidden_size': 32},
      {'patch_size': 0},
      {'conv_kernel_size': 2},
      {'num_hidden_layers': 13},
      {
        'layer_types': ['sliding_attention'],
      },
      {
        'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000},
      },
    ]) {
      final json = fixture();
      (json['vision_config'] as Map<String, Object?>).addAll(change);
      expect(() => Lfm3VlConfig.fromJson(json), throwsFormatException);
    }
  });
}
