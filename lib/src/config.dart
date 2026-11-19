/// The subset of a text model's config.json the engine implements.
///
/// The engine runs one family of decoder layers: a mixer (a gated short
/// convolution or grouped-query attention with QK-norm and RoPE) and a
/// SwiGLU FFN, each behind an RMSNorm and optionally followed by another
/// (sandwich norms). [TextModelConfig.architecture] names the checkpoint
/// family whose config.json and tensor names were read.
library;

import 'dart:convert';
import 'dart:io';

enum LayerType { conv, fullAttention, slidingAttention }

/// The checkpoint families the engine reads.
enum ModelArchitecture { lfm3, qwen3 }

final class RopeSpec {
  const RopeSpec(this.theta, this.partialRotaryFactor);

  final double theta;

  /// Fraction of head_dim whose pairs rotate; the rest get zero frequency.
  final double partialRotaryFactor;
}

/// Older name of [TextModelConfig], from when LFM3 was the only family.
typedef Lfm3Config = TextModelConfig;

final class TextModelConfig {
  TextModelConfig._({
    required this.architecture,
    required this.hiddenSize,
    required this.intermediateSize,
    required this.numHeads,
    required this.numKvHeads,
    required this.headDim,
    required this.vocabSize,
    required this.convKernel,
    required this.slidingWindow,
    required this.normEps,
    required this.layerTypes,
    required this.fullRope,
    required this.slidingRope,
    required this.sandwichNorm,
    required this.attentionOutputGate,
    required this.eosTokenIds,
  });

  factory TextModelConfig.load(String modelDir) {
    return TextModelConfig.fromJson(
      jsonDecode(File('$modelDir/config.json').readAsStringSync())
          as Map<String, Object?>,
    );
  }

  /// Parses a standalone text configuration, including a VL text_config.
  factory TextModelConfig.fromJson(Map<String, Object?> json) {
    T field<T>(String key) {
      final value = json[key];
      if (value is! T) {
        throw FormatException('config.json: $key is $value, expected $T');
      }
      return value;
    }

    // Fail loudly on anything this engine does not implement rather than
    // producing plausible but wrong text.
    switch (field<String>('model_type')) {
      case 'lfm3':
        break;
      case 'qwen3':
        return _qwen3(json, field);
      case final type:
        throw FormatException('config.json: unsupported model_type $type');
    }
    if (field<bool>('tie_word_embeddings') != true ||
        field<bool>('conv_bias') != false) {
      throw const FormatException(
        'config.json: only tied embeddings and bias-free convs are supported',
      );
    }

    final rope = field<Map<String, Object?>>('rope_parameters');
    RopeSpec ropeFor(String layerType) {
      final spec = rope[layerType] as Map<String, Object?>;
      if (spec['rope_type'] != 'default') {
        throw FormatException('config.json: unsupported rope for $layerType');
      }
      return RopeSpec(
        (spec['rope_theta'] as num).toDouble(),
        ((spec['partial_rotary_factor'] ?? 1.0) as num).toDouble(),
      );
    }

    final layerTypes = [
      for (final type in field<List<Object?>>('layer_types'))
        switch (type) {
          'conv' => LayerType.conv,
          'full_attention' => LayerType.fullAttention,
          'sliding_attention' => LayerType.slidingAttention,
          _ => throw FormatException('config.json: unknown layer type $type'),
        },
    ];
    if (layerTypes.length != field<int>('num_hidden_layers')) {
      throw const FormatException('config.json: layer_types length mismatch');
    }

    final eos = json['eos_token_id'];
    return TextModelConfig._(
      architecture: ModelArchitecture.lfm3,
      hiddenSize: field<int>('hidden_size'),
      intermediateSize: field<int>('intermediate_size'),
      numHeads: field<int>('num_attention_heads'),
      numKvHeads: field<int>('num_key_value_heads'),
      headDim: field<int>('head_dim'),
      vocabSize: field<int>('vocab_size'),
      convKernel: field<int>('conv_L_cache'),
      slidingWindow: field<int>('sliding_window'),
      normEps: field<num>('norm_eps').toDouble(),
      layerTypes: layerTypes,
      fullRope: ropeFor('full_attention'),
      slidingRope: ropeFor('sliding_attention'),
      sandwichNorm: field<bool>('use_sandwich_norm'),
      attentionOutputGate: true,
      eosTokenIds: List.unmodifiable(
        eos is List ? eos.cast<int>() : [eos as int],
      ),
    );
  }

  /// Qwen3 dense: every layer full attention with QK-norm, no output gate
  /// and no sandwich norms.
  static TextModelConfig _qwen3(
    Map<String, Object?> json,
    T Function<T>(String key) field,
  ) {
    if (field<bool>('tie_word_embeddings') != true) {
      throw const FormatException(
        'config.json: only tied embeddings are supported',
      );
    }
    if (json['attention_bias'] == true ||
        json['use_sliding_window'] == true ||
        json['rope_scaling'] != null) {
      throw const FormatException(
        'config.json: attention bias, sliding windows and rope scaling are '
        'not supported',
      );
    }
    if (json['hidden_act'] case final act? when act != 'silu') {
      throw FormatException('config.json: unsupported hidden_act $act');
    }
    // Newer transformers move rope_theta into rope_parameters.
    final ropeParameters = json['rope_parameters'];
    final theta = ropeParameters is Map
        ? ropeParameters['rope_theta']
        : json['rope_theta'];
    if (ropeParameters is Map &&
        (ropeParameters['rope_type'] ?? 'default') != 'default') {
      throw const FormatException('config.json: unsupported rope_type');
    }
    if (theta is! num) {
      throw const FormatException('config.json: rope_theta is missing');
    }
    final rope = RopeSpec(theta.toDouble(), 1);
    final eos = json['eos_token_id'];
    return TextModelConfig._(
      architecture: ModelArchitecture.qwen3,
      hiddenSize: field<int>('hidden_size'),
      intermediateSize: field<int>('intermediate_size'),
      numHeads: field<int>('num_attention_heads'),
      numKvHeads: field<int>('num_key_value_heads'),
      headDim: field<int>('head_dim'),
      vocabSize: field<int>('vocab_size'),
      convKernel: 0,
      slidingWindow: 0,
      normEps: field<num>('rms_norm_eps').toDouble(),
      layerTypes: List.filled(
        field<int>('num_hidden_layers'),
        LayerType.fullAttention,
        growable: false,
      ),
      fullRope: rope,
      slidingRope: rope,
      sandwichNorm: false,
      attentionOutputGate: false,
      eosTokenIds: List.unmodifiable(
        eos is List ? eos.cast<int>() : [eos as int],
      ),
    );
  }

  final ModelArchitecture architecture;
  final int hiddenSize;
  final int intermediateSize;
  final int numHeads;
  final int numKvHeads;
  final int headDim;
  final int vocabSize;
  final int convKernel;
  final int slidingWindow;
  final double normEps;
  final List<LayerType> layerTypes;
  final RopeSpec fullRope;
  final RopeSpec slidingRope;
  final bool sandwichNorm;

  /// Whether attention layers scale each head's output by
  /// sigmoid(gate_proj(x)) before o_proj.
  final bool attentionOutputGate;

  /// Tokens that end a reply, the first being config.json's own.
  final List<int> eosTokenIds;

  /// The first of [eosTokenIds].
  int get eosTokenId => eosTokenIds.first;

  bool isEos(int token) => eosTokenIds.contains(token);

  int get numLayers => layerTypes.length;
  int get kvDim => numKvHeads * headDim;
}
