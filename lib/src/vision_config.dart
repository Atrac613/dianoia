/// LFM3-VL checkpoint metadata, separate from the text-only model loader.
library;

import 'dart:convert';
import 'dart:io';

import 'config.dart';

enum VisionLayerType { conv, attention }

/// Validated configuration for the LFM3 vision encoder and projector.
/// Parsing this metadata does not load weights or enable image inference.
final class Lfm3VlConfig {
  Lfm3VlConfig._(
    this.text,
    this.vision,
    this.imageTokenId,
    this.videoTokenId,
    this.visionStartTokenId,
    this.visionEndTokenId,
    this.downsampleFactor,
    this.projectorHiddenSize,
    this.projectorNormEps,
  );

  factory Lfm3VlConfig.load(String modelDir) => Lfm3VlConfig.fromJson(
    jsonDecode(File('$modelDir/config.json').readAsStringSync())
        as Map<String, Object?>,
  );

  factory Lfm3VlConfig.fromJson(Map<String, Object?> json) {
    final fields = _Fields(json, 'config.json');
    fields.requireValue('model_type', 'lfm3_vl');
    return Lfm3VlConfig._parse(
      fields,
      fields.get<int>('vision_start_token_id'),
      fields.get<int>('vision_end_token_id'),
    );
  }

  /// LFM3-Omni's config.json carries the same text, vision and projector
  /// sections, but not the span marker IDs: the tokenizer defines those.
  /// Audio sections are validated by the audio loaders, not here.
  factory Lfm3VlConfig.fromOmniJson(
    Map<String, Object?> json, {
    required int visionStartTokenId,
    required int visionEndTokenId,
  }) {
    final fields = _Fields(json, 'config.json');
    fields.requireValue('model_type', 'lfm3_omni');
    return Lfm3VlConfig._parse(fields, visionStartTokenId, visionEndTokenId);
  }

  factory Lfm3VlConfig._parse(_Fields fields, int start, int end) {
    fields.requireValue('tie_word_embeddings', true);
    fields.requireValue('projector_bias', true);
    fields.requireValue('projector_hidden_act', 'gelu');
    final text = Lfm3Config.fromJson(fields.get('text_config'));
    final vision = Lfm3VisionConfig._parse(fields.get('vision_config'));
    final downsample = fields.positiveInt('downsample_factor');
    if (downsample != vision.spatialMergeSize) {
      throw const FormatException('projector and vision spatial merge differ');
    }
    final tokenIds = [
      fields.get<int>('image_token_id'),
      fields.get<int>('video_token_id'),
      start,
      end,
    ];
    if (tokenIds.toSet().length != tokenIds.length ||
        tokenIds.any((id) => id < 0 || id >= text.vocabSize)) {
      throw const FormatException(
        'vision token IDs must be distinct vocabulary entries',
      );
    }
    return Lfm3VlConfig._(
      text,
      vision,
      tokenIds[0],
      tokenIds[1],
      tokenIds[2],
      tokenIds[3],
      downsample,
      fields.positiveInt('projector_hidden_size'),
      fields.positiveDouble('projector_layer_norm_eps'),
    );
  }

  final Lfm3Config text;
  final Lfm3VisionConfig vision;
  final int imageTokenId, videoTokenId, visionStartTokenId, visionEndTokenId;
  final int downsampleFactor, projectorHiddenSize;
  final double projectorNormEps;

  int get projectorInputSize =>
      vision.hiddenSize * downsampleFactor * downsampleFactor;

  /// Checkpoint names differ from standalone text checkpoints.
  static const languagePrefix = 'model.language_model.';
  static const visionPrefix = 'model.vision_tower.';
  static const projectorPrefix = 'model.multi_modal_projector.';

  /// Maps an engine language tensor name to the original VL checkpoint name.
  /// Never accepts a vision/projector tensor as a language tensor.
  static String languageTensorName(String name) {
    if (name != 'model.embed_tokens.weight' &&
        name != 'model.embedding_norm.weight' &&
        !RegExp(r'^model\.layers\.\d+\.').hasMatch(name)) {
      throw ArgumentError.value(name, 'name', 'not an engine language tensor');
    }
    return '$languagePrefix${name.substring('model.'.length)}';
  }
}

final class Lfm3VisionConfig {
  Lfm3VisionConfig._(
    this.hiddenSize,
    this.intermediateSize,
    this.numHeads,
    this.numKvHeads,
    this.inChannels,
    this.patchSize,
    this.temporalPatchSize,
    this.spatialMergeSize,
    this.convKernelSize,
    this.normEps,
    this.ropeTheta,
    this.layerTypes,
  );

  factory Lfm3VisionConfig._parse(Map<String, Object?> json) {
    final f = _Fields(json, 'vision_config');
    f.requireValue('model_type', 'lfm3_vl_vision');
    final hidden = f.positiveInt('hidden_size');
    final heads = f.positiveInt('num_attention_heads');
    final kvHeads = f.positiveInt('num_key_value_heads');
    if (hidden % heads != 0 ||
        heads % kvHeads != 0 ||
        (hidden ~/ heads) % 4 != 0) {
      throw const FormatException(
        'vision attention dimensions do not support axial RoPE',
      );
    }
    final rope = _Fields(f.get('rope_parameters'), 'vision rope_parameters');
    rope.requireValue('rope_type', 'axial');
    final layers = [
      for (final type in f.get<List<Object?>>('layer_types'))
        switch (type) {
          'conv' => VisionLayerType.conv,
          'full_attention' => VisionLayerType.attention,
          _ => throw FormatException('unsupported vision layer type $type'),
        },
    ];
    if (layers.length != f.positiveInt('num_hidden_layers')) {
      throw const FormatException('vision layer_types length mismatch');
    }
    final kernel = f.positiveInt('conv_kernel_size');
    if (kernel.isEven) {
      throw const FormatException(
        'vision convolution needs an odd kernel size',
      );
    }
    return Lfm3VisionConfig._(
      hidden,
      f.positiveInt('intermediate_size'),
      heads,
      kvHeads,
      f.positiveInt('in_channels'),
      f.positiveInt('patch_size'),
      f.positiveInt('temporal_patch_size'),
      f.positiveInt('spatial_merge_size'),
      kernel,
      f.positiveDouble('norm_eps'),
      rope.positiveDouble('rope_theta'),
      List.unmodifiable(layers),
    );
  }

  final int hiddenSize, intermediateSize, numHeads, numKvHeads, inChannels;
  final int patchSize, temporalPatchSize, spatialMergeSize, convKernelSize;
  final double normEps, ropeTheta;
  final List<VisionLayerType> layerTypes;
  int get headDim => hiddenSize ~/ numHeads;
  int get numLayers => layerTypes.length;
}

final class _Fields {
  _Fields(this.json, this.path);
  final Map<String, Object?> json;
  final String path;

  T get<T>(String key) {
    final value = json[key];
    if (value is! T) {
      throw FormatException('$path.$key: expected $T, got $value');
    }
    return value;
  }

  void requireValue(String key, Object expected) {
    if (json[key] != expected) {
      throw FormatException('$path.$key: expected $expected, got ${json[key]}');
    }
  }

  int positiveInt(String key) {
    final value = get<int>(key);
    if (value <= 0) throw FormatException('$path.$key must be positive');
    return value;
  }

  double positiveDouble(String key) {
    final value = get<num>(key).toDouble();
    if (!value.isFinite || value <= 0) {
      throw FormatException('$path.$key must be finite and positive');
    }
    return value;
  }
}
