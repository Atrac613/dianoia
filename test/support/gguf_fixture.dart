/// Tiny valid GGUFs generated at runtime; no downloaded weights in unit tests.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:dianoia/lab.dart';

void _number(BytesBuilder b, int n, int width) {
  final data = ByteData(width);
  if (width == 4) {
    data.setUint32(0, n, Endian.little);
  } else {
    data.setUint64(0, n, Endian.little);
  }
  b.add(data.buffer.asUint8List());
}

void _string(BytesBuilder b, String s) {
  final bytes = utf8.encode(s);
  _number(b, bytes.length, 8);
  b.add(bytes);
}

void writeTestGguf(
  String path,
  Map<String, Object> metadata,
  List<(String, List<int>, GgufType, Uint8List)> tensors,
) {
  final header = BytesBuilder()..add(utf8.encode('GGUF'));
  _number(header, 3, 4);
  _number(header, tensors.length, 8);
  _number(header, metadata.length, 8);
  for (final e in metadata.entries) {
    _string(header, e.key);
    _number(
      header,
      e.value is String
          ? 8
          : e.value is double
          ? 6
          : 4,
      4,
    );
    if (e.value is String) {
      _string(header, e.value as String);
    } else if (e.value is double) {
      final data = ByteData(4)..setFloat32(0, e.value as double, Endian.little);
      header.add(data.buffer.asUint8List());
    } else {
      _number(header, e.value as int, 4);
    }
  }
  final payload = BytesBuilder();
  for (final (name, shape, type, bytes) in tensors) {
    _string(header, name);
    _number(header, shape.length, 4);
    for (final n in shape.reversed) {
      _number(header, n, 8);
    }
    _number(header, type.id, 4);
    _number(header, payload.length, 8);
    payload.add(bytes);
    payload.add(Uint8List((-payload.length) % 32));
  }
  header.add(Uint8List((-header.length) % 32));
  final file = File(path).openSync(mode: FileMode.write);
  try {
    file.writeFromSync(header.takeBytes());
    file.writeFromSync(payload.takeBytes());
  } finally {
    file.closeSync();
  }
}

/// The synthetic language is deliberately zero-valued, including its valid
/// Q6_K embedding blocks; it tests installation/layout rather than accuracy.
void writeVisionGgufFixture(String model, String language, String projector) {
  final config = Lfm3VlConfig.load(model);
  final c = config.text;
  final languageTensors = <(String, List<int>, GgufType, Uint8List)>[];
  void tensor(String name, List<int> shape, GgufType type) {
    final n = shape.fold<int>(1, (a, b) => a * b);
    final data = Uint8List(n ~/ type.blockValues * type.blockBytes);
    if (type == GgufType.f32 && name.contains('norm')) {
      Float32List.sublistView(data).fillRange(0, n, 1);
    }
    languageTensors.add((name, shape, type, data));
  }

  tensor('token_embd.weight', [c.vocabSize, c.hiddenSize], GgufType.q6K);
  tensor('output_norm.weight', [c.hiddenSize], GgufType.f32);
  for (var i = 0; i < c.numLayers; i++) {
    final g = 'blk.$i.';
    for (final n in [
      'attn_norm',
      'post_attention_norm',
      'ffn_norm',
      'post_ffw_norm',
    ]) {
      tensor('$g$n.weight', [c.hiddenSize], GgufType.f32);
    }
    for (final n in ['ffn_gate', 'ffn_up']) {
      tensor('$g$n.weight', [c.intermediateSize, c.hiddenSize], GgufType.q4_0);
    }
    tensor('${g}ffn_down.weight', [
      c.hiddenSize,
      c.intermediateSize,
    ], GgufType.q4_0);
    if (c.layerTypes[i] == LayerType.conv) {
      tensor('${g}shortconv.conv.weight', [
        c.hiddenSize,
        c.convKernel,
      ], GgufType.f32);
      tensor('${g}shortconv.in_proj.weight', [
        3 * c.hiddenSize,
        c.hiddenSize,
      ], GgufType.q4_0);
      tensor('${g}shortconv.out_proj.weight', [
        c.hiddenSize,
        c.hiddenSize,
      ], GgufType.q4_0);
    } else {
      for (final n in ['attn_q_norm', 'attn_k_norm']) {
        tensor('$g$n.weight', [c.headDim], GgufType.f32);
      }
      for (final (n, rows) in [
        ('attn_q', c.hiddenSize),
        ('attn_k', c.kvDim),
        ('attn_v', c.kvDim),
        ('attn_gate', c.numHeads),
      ]) {
        tensor('$g$n.weight', [rows, c.hiddenSize], GgufType.q4_0);
      }
      tensor('${g}attn_output.weight', [
        c.hiddenSize,
        c.hiddenSize,
      ], GgufType.q4_0);
    }
  }
  writeTestGguf(language, {
    'general.architecture': 'lfm3',
    'lfm3.block_count': c.numLayers,
  }, languageTensors);
  final st = SafetensorsFile.open('$model/model.safetensors');
  final vision = <(String, List<int>, GgufType, Uint8List)>[];
  try {
    const v = 'model.vision_tower.', p = 'model.multi_modal_projector.';
    const suffixes = {
      'operator_norm': 'ln1',
      'ffn_norm': 'ln2',
      'feed_forward.w1': 'ffn_gate',
      'feed_forward.w3': 'ffn_up',
      'feed_forward.w2': 'ffn_down',
      'conv.in_proj': 'conv2d_in_proj',
      'conv.out_proj': 'conv2d_out_proj',
      'conv.spatial_conv': 'conv2d_spatial',
      'self_attn.q_proj': 'attn_q',
      'self_attn.k_proj': 'attn_k',
      'self_attn.v_proj': 'attn_v',
      'self_attn.out_proj': 'attn_out',
      'self_attn.q_layernorm': 'attn_q_norm',
      'self_attn.k_layernorm': 'attn_k_norm',
    };
    for (final e in lfm3VisionTensorShapes(config).entries) {
      String name;
      if (e.key == '${v}patchifier.weight') continue;
      if (e.key == '${v}embedding_norm.weight') {
        name = 'v.embedding_norm.weight';
      } else if (e.key.startsWith(p)) {
        name = e.key
            .replaceFirst(p, 'mm.')
            .replaceFirst('linear_1', '0')
            .replaceFirst('linear_2', '2')
            .replaceFirst('layer_norm', 'input_norm');
      } else {
        final m = RegExp(
          r'^model\.vision_tower\.layers\.(\d+)\.(.+)\.weight$',
        ).firstMatch(e.key)!;
        name = 'v.blk.${m[1]}.${suffixes[m[2]]!}.weight';
      }
      final values = st.readF32(e.key, e.value);
      vision.add((name, e.value, GgufType.f32, values.buffer.asUint8List()));
    }
    final c = config.vision;
    for (final name in ['v.patch_embd.weight', 'v.patch_embd.weight.1']) {
      final bytes = Uint8List(c.hiddenSize * 3 * c.patchSize * c.patchSize * 2);
      final bits = ByteData.sublistView(bytes);
      for (var i = 0; i < bytes.length; i += 2) {
        bits.setUint16(i, name.endsWith('.1') ? 0xc100 : 0x3e00, Endian.little);
      }
      vision.add((
        name,
        [c.hiddenSize, 3, c.patchSize, c.patchSize],
        GgufType.f16,
        bytes,
      ));
    }
  } finally {
    st.close();
  }
  writeTestGguf(projector, {
    'general.architecture': 'clip',
    'clip.vision.projector_type': 'lfm3vl',
  }, vision);
}
