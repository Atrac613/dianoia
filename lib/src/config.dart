/// The subset of LFM3's config.json the engine implements.
library;

import 'dart:convert';
import 'dart:io';

enum LayerType { conv, fullAttention, slidingAttention }

final class RopeSpec {
  const RopeSpec(this.theta, this.partialRotaryFactor);

  final double theta;

  /// Fraction of head_dim whose pairs rotate; the rest get zero frequency.
  final double partialRotaryFactor;
}

final class Lfm3Config {
  Lfm3Config._({
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
    required this.eosTokenId,
  });

  factory Lfm3Config.load(String modelDir) {
    final json =
        jsonDecode(File('$modelDir/config.json').readAsStringSync())
            as Map<String, Object?>;

    T field<T>(String key) {
      final value = json[key];
      if (value is! T) {
        throw FormatException('config.json: $key is $value, expected $T');
      }
      return value;
    }

    // Fail loudly on anything this engine does not implement rather than
    // producing plausible but wrong text.
    if (field<String>('model_type') != 'lfm3') {
      throw const FormatException('config.json: model_type is not lfm3');
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
    return Lfm3Config._(
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
      eosTokenId: eos is List ? eos.first as int : eos as int,
    );
  }

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
  final int eosTokenId;

  int get numLayers => layerTypes.length;
  int get kvDim => numKvHeads * headDim;
}
