/// Pinned official Q4_0 sources and validated audio encoder tensor layout.
library;

import 'download.dart';
import 'gguf.dart';

/// Shared source pins for research preparation and AudioGgufSpec installation.
abstract final class Lfm3AudioQ4Source {
  static const repository = 'LiquidAI/LFM3-Audio-2B-GGUF';
  static const revision = '02355c1a141e464a064f2ab2c276638dbcb7f38a';
  static const _base = 'https://huggingface.co/$repository/resolve/$revision';
  static const language = RemoteFile(
    'LFM3-Audio-2B-Q4_0.gguf',
    '$_base/LFM3-Audio-2B-Q4_0.gguf',
    size: 1049613856,
    sha256: '112e4acb79be79dcd8df5749495614817e51c064cd9ca576aead02b64b2b7584',
  );
  static const audio = RemoteFile(
    'mmproj-LFM3-Audio-2B-Q4_0.gguf',
    '$_base/mmproj-LFM3-Audio-2B-Q4_0.gguf',
    size: 161393408,
    sha256: '467a1ee5512a3faba79bf04e789a838959cc48c72a802bf6d9cb951a77736cfc',
  );
  static const files = [language, audio];
  static const downloadBytes = 1211007264;
  static const metadataRevision = '0295e4130594c06800fd25451d97139ac2da1aff';
  static const _metadataBase =
      'https://huggingface.co/LiquidAI/LFM3-Audio-2B/resolve/$metadataRevision';
  static const config = RemoteFile(
    'audio_config.json',
    '$_metadataBase/config.json',
    size: 6243,
    sha256: '1eb2a8349a1ec9eda2ed8e8bb1a8788007ae2ae9baa38a7d959743c5379d30d8',
  );
  static const tokenizer = RemoteFile(
    'tokenizer.json',
    '$_metadataBase/tokenizer.json',
    size: 17908634,
    sha256: '41927bd314d17d7d7fadb44e10cd0919f7b71528d7741596d8cbd15c6945ec44',
  );
  static const license = RemoteFile(
    'LICENSE',
    '$_metadataBase/LICENSE',
    size: 10574,
    sha256: '4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c',
  );
  static const metadataFiles = [config, tokenizer, license];
}

/// Input encoder configuration read from an LFM3-Audio Q4_0 mmproj GGUF.
/// Does not treat the LFM2 encoder as a language or LFM3 graph.
final class Lfm3AudioEncoderConfig {
  Lfm3AudioEncoderConfig._(
    this.hiddenSize,
    this.intermediateSize,
    this.numHeads,
    this.numKvHeads,
    this.projectionSize,
    this.convChannels,
    this.melBins,
    this.window,
    this.normEps,
    this.convLayers,
  );

  factory Lfm3AudioEncoderConfig.fromGguf(GgufFile file) {
    final m = file.metadata;
    if (m['general.architecture'] != 'clip' ||
        m['clip.audio.projector_type'] != 'lfm3a' ||
        m['clip.has_audio_encoder'] != 1) {
      throw const FormatException('expected an LFM3-Audio encoder GGUF');
    }
    int positive(String name) {
      final v = m['clip.audio.$name'];
      if (v is! int || v <= 0) {
        throw FormatException('invalid clip.audio.$name');
      }
      return v;
    }

    final h = positive('embedding_length');
    final f = positive('feed_forward_length');
    final heads = positive('attention.head_count');
    final kvHeads = positive('attention.head_count_kv');
    final mel = positive('num_mel_bins');
    final count = positive('block_count');
    final eps = m['clip.audio.attention.layer_norm_epsilon'];
    final ds = file.tensors['a.conv1d.0.weight'];
    if (h % heads != 0 ||
        heads % kvHeads != 0 ||
        (h ~/ heads).isOdd ||
        h % 32 != 0 ||
        f % 32 != 0 ||
        mel != 128 ||
        eps is! num ||
        !eps.isFinite ||
        eps <= 0 ||
        ds == null ||
        ds.dims.length != 4 ||
        ds.dims[3] <= 0 ||
        ds.dims[3] % 4 != 0) {
      throw const FormatException('unsupported LFM3-Audio encoder dimensions');
    }
    final conv = <bool>[];
    for (var i = 0; i < count; i++) {
      final p = 'a.blk.$i.';
      final isConv = file.tensors.containsKey('${p}conv_dw.weight');
      final isAttn = file.tensors.containsKey('${p}attn_q.weight');
      if (isConv == isAttn) {
        throw FormatException('ambiguous audio encoder layer $i');
      }
      conv.add(isConv);
    }
    final c = Lfm3AudioEncoderConfig._(
      h,
      f,
      heads,
      kvHeads,
      positive('projection_dim'),
      ds.dims[3],
      mel,
      positive('window_size'),
      eps.toDouble(),
      List.unmodifiable(conv),
    );
    // Checking every consumed tensor before allocating matrices makes malformed
    // layouts fail atomically. Generation tensors are deliberately not loaded.
    for (final e in c.tensorShapes.entries) {
      final type = c.isQuantized(e.key) ? GgufType.q4_0 : GgufType.f32;
      final tensor = file.tensors[e.key];
      if (tensor == null ||
          tensor.type != type ||
          tensor.dims.join(',') != e.value.reversed.join(',')) {
        throw FormatException('unsupported audio encoder tensor ${e.key}');
      }
    }
    final expected = c.tensorShapes.keys.toSet();
    for (final name in file.tensors.keys) {
      if ((name.startsWith('a.') && !name.startsWith('a.gen.')) ||
          name.startsWith('mm.a.')) {
        if (!expected.contains(name)) {
          throw FormatException('unexpected audio encoder tensor $name');
        }
      }
    }
    return c;
  }

  final int hiddenSize, intermediateSize, numHeads, numKvHeads;
  final int projectionSize, convChannels, melBins, window;
  final double normEps;
  final List<bool> convLayers;
  int get headDim => hiddenSize ~/ numHeads;
  int get numLayers => convLayers.length;
  static const ropeTheta = 1000000.0;

  bool isQuantized(String name) =>
      name.endsWith('.weight') &&
      (name == 'a.pre_encode.out.weight' ||
          name == 'mm.a.mlp.1.weight' ||
          name == 'mm.a.mlp.3.weight' ||
          (name.startsWith('a.blk.') &&
              !name.contains('norm') &&
              !name.contains('.ln') &&
              !name.contains('conv_dw')));

  /// Shapes in row-major order; singleton convolution/bias axes are retained.
  Map<String, List<int>> get tensorShapes {
    final h = hiddenSize, f = intermediateSize, ch = convChannels;
    final kv = numKvHeads * headDim;
    final result = <String, List<int>>{
      'a.conv1d.0.weight': [ch, 1, 3, 3],
      'a.conv1d.2.weight': [ch, 1, 3, 3],
      'a.conv1d.3.weight': [ch, ch, 1, 1],
      'a.conv1d.5.weight': [ch, 1, 3, 3],
      'a.conv1d.6.weight': [ch, ch, 1, 1],
      for (final i in [0, 2, 3, 5, 6]) 'a.conv1d.$i.bias': [ch, 1, 1],
      'a.pre_encode.out.weight': [h, ch * melBins ~/ 8],
      'a.pre_encode.out.bias': [h],
      'a.post_ln.weight': [h],
      'mm.a.mlp.0.weight': [h],
      'mm.a.mlp.0.bias': [h],
      'mm.a.mlp.1.weight': [projectionSize, h],
      'mm.a.mlp.1.bias': [projectionSize],
      'mm.a.mlp.3.weight': [projectionSize, projectionSize],
      'mm.a.mlp.3.bias': [projectionSize],
    };
    for (var i = 0; i < numLayers; i++) {
      final p = 'a.blk.$i.';
      result.addAll({
        '${p}ln1.weight': [h],
        '${p}ln2.weight': [h],
        '${p}ffn_gate.weight': [f, h],
        '${p}ffn_up.weight': [f, h],
        '${p}ffn_down.weight': [h, f],
        if (convLayers[i]) ...{
          '${p}conv_dw.weight': [h, 3],
          '${p}conv_pw1.weight': [3 * h, h],
          '${p}conv_pw2.weight': [h, h],
        } else ...{
          '${p}attn_q.weight': [h, h],
          '${p}attn_k.weight': [kv, h],
          '${p}attn_v.weight': [kv, h],
          '${p}attn_out.weight': [h, h],
          '${p}attn_q_norm.weight': [headDim],
          '${p}attn_k_norm.weight': [headDim],
        },
      });
    }
    return result;
  }
}
