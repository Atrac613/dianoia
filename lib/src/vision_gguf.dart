/// Import the paired official LFM3-VL QAD language and BF16 vision GGUFs.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'gguf.dart';
import 'native.dart';
import 'q8_cache.dart';
import 'quant.dart';
import 'vision_bundle.dart';
import 'vision_config.dart';
import 'vision_encoder.dart';

String _visionName(String name) {
  for (final e in const {
    'mm.0': 'linear_1',
    'mm.2': 'linear_2',
    'mm.input_norm': 'layer_norm',
  }.entries) {
    if (name.startsWith('${e.key}.')) {
      return 'model.multi_modal_projector.${e.value}${name.substring(e.key.length)}';
    }
  }
  if (name == 'v.embedding_norm.weight') {
    return 'model.vision_tower.embedding_norm.weight';
  }
  final match = RegExp(r'^v\.blk\.(\d+)\.(.+)\.weight$').firstMatch(name);
  const names = {
    'ln1': 'operator_norm',
    'ln2': 'ffn_norm',
    'ffn_gate': 'feed_forward.w1',
    'ffn_up': 'feed_forward.w3',
    'ffn_down': 'feed_forward.w2',
    'conv2d_in_proj': 'conv.in_proj',
    'conv2d_out_proj': 'conv.out_proj',
    'conv2d_spatial': 'conv.spatial_conv',
    'attn_q': 'self_attn.q_proj',
    'attn_k': 'self_attn.k_proj',
    'attn_v': 'self_attn.v_proj',
    'attn_out': 'self_attn.out_proj',
    'attn_q_norm': 'self_attn.q_layernorm',
    'attn_k_norm': 'self_attn.k_layernorm',
  };
  final suffix = match == null ? null : names[match[2]];
  if (suffix == null) throw FormatException('unsupported VL GGUF tensor $name');
  return 'model.vision_tower.layers.${match![1]}.$suffix.weight';
}

/// Writes only the paired mmproj's vision/projector tensors to [output].
/// BF16/F32 payloads are unchanged; split F16 temporal patches widen exactly.
void importVlVisionGguf(String path, Lfm3VlConfig config, String output) {
  final gguf = GgufFile.open(path);
  try {
    if (gguf.metadata['clip.vision.projector_type'] != 'lfm3vl') {
      throw const FormatException('expected an LFM3-VL projector GGUF');
    }
    final c = config.vision;
    if (c.temporalPatchSize != 2 || c.inChannels != 3) {
      throw const FormatException('unsupported VL patchifier layout');
    }
    final expected = lfm3VisionTensorShapes(config);
    final entries = <String, (GgufTensor, List<int>)>{};
    for (final tensor in gguf.tensors.values) {
      if (tensor.name.startsWith('v.patch_embd.weight')) continue;
      final name = _visionName(tensor.name);
      final shape = tensor.dims.reversed.toList();
      if (entries.containsKey(name) ||
          expected[name]?.join(',') != shape.join(',') ||
          ![GgufType.f32, GgufType.bf16].contains(tensor.type)) {
        throw FormatException('unsupported vision tensor ${tensor.name}');
      }
      entries[name] = (tensor, shape);
    }
    const patchName = 'model.vision_tower.patchifier.weight';
    if (entries.length != expected.length - 1 ||
        expected.keys.any((n) => n != patchName && !entries.containsKey(n))) {
      throw const FormatException('incomplete vision GGUF');
    }
    final slices = [
      for (final name in ['v.patch_embd.weight', 'v.patch_embd.weight.1'])
        ByteData.sublistView(
          gguf.bytes(
            gguf.tensor(name, GgufType.f16, [
              c.patchSize,
              c.patchSize,
              3,
              c.hiddenSize,
            ]),
          ),
        ),
    ];
    final area = c.patchSize * c.patchSize;
    final patch = Float32List(c.hiddenSize * 3 * 2 * area);
    var at = 0;
    for (var row = 0; row < c.hiddenSize; row++) {
      for (var channel = 0; channel < 3; channel++) {
        for (final slice in slices) {
          for (var i = 0; i < area; i++) {
            patch[at++] = halfToDouble(
              slice.getUint16(
                ((row * 3 + channel) * area + i) * 2,
                Endian.little,
              ),
            );
          }
        }
      }
    }
    final header = <String, Object?>{};
    var offset = 0;
    for (final e in entries.entries) {
      final (tensor, shape) = e.value;
      header[e.key] = {
        'dtype': tensor.type == GgufType.f32 ? 'F32' : 'BF16',
        'shape': shape,
        'data_offsets': [offset, offset + tensor.bytes],
      };
      offset += tensor.bytes;
    }
    header[patchName] = {
      'dtype': 'F32',
      'shape': expected[patchName],
      'data_offsets': [offset, offset + patch.lengthInBytes],
    };
    final json = utf8.encode(jsonEncode(header));
    final padded = (json.length + 7) ~/ 8 * 8;
    final prefix = ByteData(8)..setUint64(0, padded, Endian.little);
    final file = File(output).openSync(mode: FileMode.write);
    try {
      file.writeFromSync(prefix.buffer.asUint8List());
      file.writeFromSync(json);
      file.writeFromSync(
        Uint8List(padded - json.length)..fillRange(0, padded - json.length, 32),
      );
      for (final e in entries.values) {
        file.writeFromSync(gguf.bytes(e.$1));
      }
      file.writeFromSync(patch.buffer.asUint8List());
    } finally {
      file.closeSync();
    }
  } finally {
    gguf.close();
  }
}

/// Imports into a portable bundle, retaining Q4 blocks and using the validated
/// Q6_K-to-Q8 embedding/head adapter. Intermediate files are always removed.
Future<void> importVlQadBundle(
  String language,
  String projector,
  String metadata,
  String output,
) async {
  if (Directory(output).existsSync()) throw StateError('output already exists');
  Directory(output).parent.createSync(recursive: true);
  final temp = Directory(output).parent.createTempSync('.dianoia-vl-qad-');
  try {
    if (!NativeKernels.isLoaded) NativeKernels.load(null);
    final vision = Directory('${temp.path}/vision')..createSync();
    final text = Directory('${temp.path}/text')..createSync();
    final backbone = '${temp.path}/language';
    final config = Lfm3VlConfig.load(metadata);
    for (final name in [
      'config.json',
      'processor_config.json',
      'tokenizer.json',
      'LICENSE',
    ]) {
      File('$metadata/$name').copySync('${vision.path}/$name');
    }
    importVlVisionGguf(projector, config, '${vision.path}/model.safetensors');
    final json =
        jsonDecode(File('$metadata/config.json').readAsStringSync()) as Map;
    File(
      '${text.path}/config.json',
    ).writeAsStringSync(jsonEncode(json['text_config']));
    File('$metadata/tokenizer.json').copySync('${text.path}/tokenizer.json');
    importGguf(language, text.path, backbone);
    File('$backbone/vl_backbone.json').writeAsStringSync(
      jsonEncode({
        'version': 1,
        'source_key': Q8Cache.keyFor(vision.path),
        'backbone_key': Q8Cache.keyFor(backbone),
      }),
    );
    await packVlBundle(
      vision.path,
      backbone,
      output,
      weights: WeightFormat.q4,
      useCacheEmbedding: true,
    );
  } finally {
    temp.deleteSync(recursive: true);
  }
}
