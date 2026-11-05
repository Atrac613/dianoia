/// Import the paired official LFM3-VL QAD language and BF16 vision GGUFs.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'gguf.dart';
import 'native.dart';
import 'vision_backbone.dart';
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

/// The two F16 temporal patch slices, interleaved per channel into the
/// checkpoint's [hidden, 3 * 2 * patch * patch] float32 patchifier.
Float32List _patchifier(GgufFile gguf, Lfm3VisionConfig c) {
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
  return patch;
}

void _checkVisionGguf(GgufFile gguf, Lfm3VlConfig config) {
  if (gguf.metadata['clip.vision.projector_type'] != 'lfm3vl') {
    throw const FormatException('expected an LFM3-VL projector GGUF');
  }
  final c = config.vision;
  if (c.temporalPatchSize != 2 || c.inChannels != 3) {
    throw const FormatException('unsupported VL patchifier layout');
  }
}

/// Vision and projector weights of an LFM3-VL or LFM3-Omni mmproj, widened
/// to float32 under checkpoint names for [Lfm3VisionEncoder.fromWeights].
/// F32, F16, BF16 and Q4_0 payloads convert exactly; audio tensors in an
/// Omni mmproj are ignored, and every vision tensor must be present.
Map<String, Float32List> readVisionGgufWeights(
  GgufFile gguf,
  Lfm3VlConfig config,
) => _readVisionGguf(gguf, config, null);

Map<String, Float32List> _readVisionGguf(
  GgufFile gguf,
  Lfm3VlConfig config,
  Map<String, NativeMatrix>? native,
) {
  _checkVisionGguf(gguf, config);
  final expected = lfm3VisionTensorShapes(config);
  final out = <String, Float32List>{};
  bool seen(String name) =>
      out.containsKey(name) || (native?.containsKey(name) ?? false);
  for (final tensor in gguf.tensors.values) {
    final source = tensor.name;
    if (source.startsWith('v.patch_embd.weight') ||
        !(source.startsWith('v.') || source.startsWith('mm.')) ||
        source.startsWith('mm.a.')) {
      continue;
    }
    final name = _visionName(source);
    final shape = tensor.dims.reversed.toList();
    if (seen(name) || expected[name]?.join(',') != shape.join(',')) {
      throw FormatException('unsupported vision tensor $source');
    }
    final bytes = gguf.bytes(tensor);
    if (native != null && tensor.type == GgufType.q4_0 && shape.length == 2) {
      native[name] = NativeQ4Matrix.fromGgufBlocks(bytes, shape[0], shape[1]);
      continue;
    }
    out[name] = switch (tensor.type) {
      GgufType.f32 => Float32List.fromList(
        Float32List.sublistView(Uint8List.fromList(bytes)),
      ),
      GgufType.f16 => _widen16(bytes, halfToDouble),
      GgufType.bf16 => _widen16(
        bytes,
        (bits) => (ByteData(4)..setUint32(0, bits << 16)).getFloat32(0),
      ),
      GgufType.q4_0 => dequantizeQ4_0(bytes, tensor.values),
      _ => throw FormatException('unsupported vision tensor type $source'),
    };
  }
  const patchName = 'model.vision_tower.patchifier.weight';
  if (out.length + (native?.length ?? 0) != expected.length - 1 ||
      expected.keys.any((n) => n != patchName && !seen(n))) {
    throw const FormatException('incomplete vision GGUF');
  }
  out[patchName] = _patchifier(gguf, config.vision);
  return out;
}

Float32List _widen16(Uint8List bytes, double Function(int bits) convert) {
  final data = ByteData.sublistView(bytes);
  final out = Float32List(bytes.length ~/ 2);
  for (var i = 0; i < out.length; i++) {
    out[i] = convert(data.getUint16(i * 2, Endian.little));
  }
  return out;
}

/// Loads the CPU vision encoder straight from an mmproj GGUF, without
/// writing an intermediate safetensors file. Q4_0 linear weights stay native
/// Q4 blocks, multiplied with float inputs, unless [nativeQ4] is false; then
/// every weight is widened to float32 on the Dart heap.
Lfm3VisionEncoder loadVisionEncoderGguf(
  String path,
  Lfm3VlConfig config, {
  bool nativeQ4 = true,
}) {
  if (nativeQ4 && !NativeKernels.isLoaded) NativeKernels.load(null);
  final gguf = GgufFile.open(path);
  final native = <String, NativeMatrix>{};
  try {
    final weights = _readVisionGguf(gguf, config, nativeQ4 ? native : null);
    return Lfm3VisionEncoder.fromWeights(config, weights, native: native);
  } catch (_) {
    for (final m in native.values) {
      m.free();
    }
    rethrow;
  } finally {
    gguf.close();
  }
}

/// Writes only the paired mmproj's vision/projector tensors to [output].
/// BF16/F32 payloads are unchanged; split F16 temporal patches widen exactly.
void importVlVisionGguf(String path, Lfm3VlConfig config, String output) {
  final gguf = GgufFile.open(path);
  try {
    _checkVisionGguf(gguf, config);
    final c = config.vision;
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
    final patch = _patchifier(gguf, c);
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
    writeVlBackboneIdentity(vision.path, backbone);
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
