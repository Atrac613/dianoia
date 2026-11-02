import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

// Three small layers exercise conv, global attention and sliding attention.
Map<String, List<int>> _shapes() {
  final result = <String, List<int>>{
    'model.embed_tokens.weight': [8, 4],
    'model.embedding_norm.weight': [4],
  };
  for (var i = 0; i < 3; i++) {
    final p = 'model.layers.$i.';
    for (final norm in [
      'operator_norm',
      'post_operator_norm',
      'ffn_norm',
      'post_ffn_norm',
    ]) {
      result['$p$norm.weight'] = [4];
    }
    result['${p}feed_forward.w1.weight'] = [8, 4];
    result['${p}feed_forward.w3.weight'] = [8, 4];
    result['${p}feed_forward.w2.weight'] = [4, 8];
    if (i == 0) {
      result['${p}conv.in_proj.weight'] = [12, 4];
      result['${p}conv.out_proj.weight'] = [4, 4];
      result['${p}conv.conv.weight'] = [4, 1, 7];
    } else {
      for (final name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']) {
        result['${p}self_attn.$name.weight'] = [4, 4];
      }
      result['${p}self_attn.gate_proj.weight'] = [1, 4];
      result['${p}self_attn.q_norm.weight'] = [4];
      result['${p}self_attn.k_norm.weight'] = [4];
    }
  }
  return result;
}

void _writeWeights(
  String path,
  Map<String, List<int>> shapes, {
  bool vl = false,
}) {
  final header = <String, Object?>{};
  final data = BytesBuilder();
  var offset = 0;
  var seed = 0;
  for (final entry in shapes.entries) {
    final count = entry.value.fold(1, (int a, b) => a * b);
    final bytes = ByteData(count * 2);
    for (var i = 0; i < count; i++) {
      final value = entry.key.contains('norm')
          ? 1.0
          : (++seed % 11 - 5) * 0.03125;
      final bits = ByteData(4)..setFloat32(0, value, Endian.little);
      bytes.setUint16(
        i * 2,
        bits.getUint32(0, Endian.little) >> 16,
        Endian.little,
      );
    }
    header[vl ? Lfm3VlConfig.languageTensorName(entry.key) : entry.key] = {
      'dtype': 'BF16',
      'shape': entry.value,
      'data_offsets': [offset, offset + count * 2],
    };
    data.add(bytes.buffer.asUint8List());
    offset += count * 2;
  }
  if (vl) {
    header['model.vision_tower.test.weight'] = {
      'dtype': 'F32',
      'shape': [1],
      'data_offsets': [offset, offset + 4],
    };
    data.add(Uint8List(4));
  }
  final json = utf8.encode(jsonEncode(header));
  final padded = (json.length + 7) ~/ 8 * 8;
  final prefix = ByteData(8)..setUint64(0, padded, Endian.little);
  File(path).writeAsBytesSync([
    ...prefix.buffer.asUint8List(),
    ...json,
    ...List.filled(padded - json.length, 32),
    ...data.takeBytes(),
  ]);
}

void main() {
  late Directory root, source;
  late Map<String, Object?> json;
  setUp(() {
    root = Directory.systemTemp.createTempSync('dianoia_vl_backbone');
    source = Directory('${root.path}/source')..createSync();
    json =
        jsonDecode(File('test/fixtures/vision/config.json').readAsStringSync())
            as Map<String, Object?>;
    (json['text_config'] as Map<String, Object?>).addAll({
      'hidden_size': 4,
      'intermediate_size': 8,
      'num_attention_heads': 1,
      'num_key_value_heads': 1,
      'head_dim': 4,
      'vocab_size': 8,
      'num_hidden_layers': 3,
      'layer_types': ['conv', 'full_attention', 'sliding_attention'],
      'eos_token_id': 4,
    });
    json.addAll({
      'image_token_id': 0,
      'video_token_id': 1,
      'vision_start_token_id': 2,
      'vision_end_token_id': 3,
    });
    File('${source.path}/config.json').writeAsStringSync(jsonEncode(json));
    File('${source.path}/tokenizer.json').writeAsStringSync('{}');
    _writeWeights('${source.path}/model.safetensors', _shapes(), vl: true);
  });
  tearDown(() => root.deleteSync(recursive: true));

  test(
    'imports raw language bytes and runs all three layer types identically',
    () async {
      final reference = Directory('${root.path}/reference')..createSync();
      File(
        '${reference.path}/config.json',
      ).writeAsStringSync(jsonEncode(json['text_config']));
      _writeWeights('${reference.path}/model.safetensors', _shapes());
      final out = '${root.path}/imported';
      importVlBackbone(source.path, out);
      final provenance =
          jsonDecode(File('$out/vl_backbone.json').readAsStringSync()) as Map;
      expect(provenance['source_key'], Q8Cache.keyFor(source.path));
      expect(provenance['backbone_key'], Q8Cache.keyFor(out));
      final expected = SafetensorsFile.open(
        '${reference.path}/model.safetensors',
      );
      final actual = SafetensorsFile.open('$out/model.safetensors');
      try {
        expect(actual.tensors.keys.toSet(), expected.tensors.keys.toSet());
        for (final entry in _shapes().entries) {
          expect(
            actual.readBf16Bits(entry.key, entry.value),
            orderedEquals(expected.readBf16Bits(entry.key, entry.value)),
          );
        }
      } finally {
        expected.close();
        actual.close();
      }
      expect(
        File('$out/vl_config.json').readAsStringSync(),
        File('${source.path}/config.json').readAsStringSync(),
      );
      final a = await Lfm3Model.load(reference.path, maxSeq: 8, maxChunk: 2);
      final b = await Lfm3Model.load(out, maxSeq: 8, maxChunk: 2);
      try {
        await a.forward([5, 6, 7]);
        await b.forward([5, 6, 7]);
        expect(b.logits.every((v) => v.isFinite), isTrue);
        expect(b.logits, orderedEquals(a.logits));
      } finally {
        await a.close();
        await b.close();
      }
    },
  );

  test('missing or wrong-shaped weights never publish output', () {
    for (final shapes in [
      _shapes()..remove('model.layers.1.self_attn.q_proj.weight'),
      _shapes()..['model.layers.2.self_attn.k_proj.weight'] = [2, 4],
    ]) {
      _writeWeights('${source.path}/model.safetensors', shapes, vl: true);
      expect(
        () => importVlBackbone(source.path, '${root.path}/out'),
        throwsFormatException,
      );
      expect(Directory('${root.path}/out').existsSync(), isFalse);
      expect(
        root.listSync().where((f) => f.path.contains('.dianoia-vl-')),
        isEmpty,
      );
    }
  });

  test('truncated data is rejected and existing output is preserved', () {
    final file = File(
      '${source.path}/model.safetensors',
    ).openSync(mode: FileMode.append);
    file.truncateSync(file.lengthSync() - 8);
    file.closeSync();
    expect(
      () => importVlBackbone(source.path, '${root.path}/out'),
      throwsFormatException,
    );
    expect(Directory('${root.path}/out').existsSync(), isFalse);
    final out = Directory('${root.path}/out')..createSync();
    File('${out.path}/keep').writeAsStringSync('keep');
    expect(
      () => importVlBackbone(source.path, out.path),
      throwsA(isA<FileSystemException>()),
    );
    expect(File('${out.path}/keep').readAsStringSync(), 'keep');
  });
}
