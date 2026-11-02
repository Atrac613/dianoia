/// Imports the language backbone of an LFM3-VL checkpoint without changing
/// its arithmetic. Vision encoder and projector weights stay in the source.
library;

import 'dart:convert';
import 'dart:io';

import 'config.dart';
import 'q8_cache.dart';
import 'native.dart';
import 'quant.dart';
import 'safetensors.dart';
import 'vision_config.dart';

/// Writes a standalone language checkpoint into a new [outDir].
///
/// The source must be a full BF16 LFM3-VL checkpoint with model.safetensors
/// and tokenizer.json. Language tensor names become the engine's canonical
/// names, with raw bytes unchanged. The original VL config is retained as
/// vl_config.json for subsequent encoder integration. This does not enable
/// image chat or substitute the text-chat model's weights for VL weights.
///
/// Required tensor names, shapes, dtypes and ranges are checked before any
/// output is published. Output must not exist; a failed copy is cleaned up.
void importVlBackbone(String modelDir, String outDir) {
  final configFile = File('$modelDir/config.json');
  final sourceConfig = configFile.readAsStringSync();
  final json = jsonDecode(sourceConfig) as Map<String, Object?>;
  final config = Lfm3VlConfig.fromJson(json);
  final tokenizer = File('$modelDir/tokenizer.json');
  if (!tokenizer.existsSync()) {
    throw FileSystemException('VL checkpoint lacks tokenizer.json', modelDir);
  }
  final destination = Directory(outDir).absolute;
  if (FileSystemEntity.typeSync(destination.path, followLinks: false) !=
      FileSystemEntityType.notFound) {
    throw FileSystemException('output already exists', destination.path);
  }
  final source = File('$modelDir/model.safetensors');
  final st = SafetensorsFile.open(source.path);
  Directory? staging;
  try {
    final shapes = lfm3LanguageTensorShapes(config.text);
    final names = {
      for (final name in shapes.keys)
        Lfm3VlConfig.languageTensorName(name): name,
    };
    for (final entry in names.entries) {
      final info = st.tensors[entry.key];
      final expected = shapes[entry.value]!;
      if (info == null || info.shape.join(',') != expected.join(',')) {
        throw FormatException(
          'VL tensor ${entry.key}: expected shape $expected',
        );
      }
      if (info.dtype != 'BF16' &&
          (info.dtype != 'F32' || entry.value == 'model.embed_tokens.weight')) {
        throw FormatException(
          'VL tensor ${entry.key}: unsupported dtype ${info.dtype}',
        );
      }
      final bytes = info.numel * (info.dtype == 'BF16' ? 2 : 4);
      if (info.begin < 0 ||
          info.end - info.begin != bytes ||
          st.dataStart + info.end > source.lengthSync()) {
        throw FormatException('VL tensor ${entry.key}: invalid byte range');
      }
    }
    for (final name in st.tensors.keys) {
      if (name.startsWith(Lfm3VlConfig.languagePrefix) &&
          !names.containsKey(name)) {
        throw FormatException('unsupported VL language tensor $name');
      }
    }
    destination.parent.createSync(recursive: true);
    staging = destination.parent.createTempSync('.dianoia-vl-');
    st.writeSubset(
      '${staging.path}/model.safetensors',
      (name, _) => names.containsKey(name),
      rename: (name) => names[name]!,
    );
    File('${staging.path}/config.json').writeAsStringSync(
      '${const JsonEncoder.withIndent('  ').convert(json['text_config'])}\n',
    );
    File('${staging.path}/vl_config.json').writeAsStringSync(sourceConfig);
    tokenizer.copySync('${staging.path}/tokenizer.json');
    writeVlBackboneIdentity(modelDir, staging.path);
    staging.renameSync(destination.path);
    staging = null;
  } finally {
    st.close();
    if (staging != null) staging.deleteSync(recursive: true);
  }
}

/// Canonical language tensor names and shapes, shared by import and packing.
Map<String, List<int>> lfm3LanguageTensorShapes(Lfm3Config c) {
  final h = c.hiddenSize;
  final shapes = <String, List<int>>{
    'model.embed_tokens.weight': [c.vocabSize, h],
    'model.embedding_norm.weight': [h],
  };
  for (var i = 0; i < c.numLayers; i++) {
    final p = 'model.layers.$i.';
    for (final norm in [
      'operator_norm',
      'ffn_norm',
      if (c.sandwichNorm) ...['post_operator_norm', 'post_ffn_norm'],
    ]) {
      shapes['$p$norm.weight'] = [h];
    }
    for (final name in ['w1', 'w3']) {
      shapes['${p}feed_forward.$name.weight'] = [c.intermediateSize, h];
    }
    shapes['${p}feed_forward.w2.weight'] = [h, c.intermediateSize];
    if (c.layerTypes[i] == LayerType.conv) {
      shapes['${p}conv.in_proj.weight'] = [3 * h, h];
      shapes['${p}conv.out_proj.weight'] = [h, h];
      shapes['${p}conv.conv.weight'] = [h, 1, c.convKernel];
    } else {
      final a = '${p}self_attn.';
      shapes['${a}q_proj.weight'] = [c.numHeads * c.headDim, h];
      shapes['${a}k_proj.weight'] = [c.kvDim, h];
      shapes['${a}v_proj.weight'] = [c.kvDim, h];
      shapes['${a}gate_proj.weight'] = [c.numHeads, h];
      shapes['${a}o_proj.weight'] = [h, c.numHeads * c.headDim];
      shapes['${a}q_norm.weight'] = [c.headDim];
      shapes['${a}k_norm.weight'] = [c.headDim];
    }
  }
  return shapes;
}

/// Checks import fingerprints before pairing a VL encoder with its backbone.
/// Uses the same metadata-based identity policy as weight caches, not a full
/// tensor-payload checksum. The importer writes this provenance file.
void validateVlBackbone(String modelDir, String backboneDir) {
  final identity = jsonDecode(
    File('$backboneDir/vl_backbone.json').readAsStringSync(),
  );
  if (identity is! Map ||
      identity['version'] != 1 ||
      identity['source_key'] != Q8Cache.keyFor(modelDir) ||
      identity['backbone_key'] != Q8Cache.keyFor(backboneDir)) {
    throw ArgumentError('backbone provenance does not match the VL checkpoint');
  }
}

/// Records the pairing used by importers and portable bundle packing.
void writeVlBackboneIdentity(String modelDir, String backboneDir) {
  File('$backboneDir/vl_backbone.json').writeAsStringSync(
    jsonEncode({
      'version': 1,
      'source_key': Q8Cache.keyFor(modelDir),
      'backbone_key': Q8Cache.keyFor(backboneDir),
    }),
  );
}

/// Quantizes one matrix at a time without constructing an inference model.
/// Each native allocation is released immediately after it is written.
void buildVlBackboneCache(String backboneDir, {required WeightFormat weights}) {
  if (!weights.isNative) throw ArgumentError('expected Q8 or Q4');
  final shapes = lfm3LanguageTensorShapes(Lfm3Config.load(backboneDir));
  final tensors = <String, (String, List<int>)>{};
  for (final e in shapes.entries.where((e) => e.value.length == 2)) {
    final name =
        e.key.contains('.self_attn.') || e.key.contains('.feed_forward.')
        ? e.key.substring(0, e.key.length - '.weight'.length)
        : e.key;
    tensors[name] = (e.key, e.value);
  }
  String format(String name) =>
      name == 'model.embed_tokens.weight' ? 'q8' : weights.name;
  final st = SafetensorsFile.open('$backboneDir/model.safetensors');
  try {
    Q8Cache.writeEach(
      '$backboneDir/lfm3-${weights.name}.cache',
      Q8Cache.keyFor(backboneDir, weights: weights.name),
      [
        for (final e in tensors.entries)
          (e.key, format(e.key), e.value.$2[0], e.value.$2[1]),
      ],
      (name) {
        final (tensor, shape) = tensors[name]!;
        if (format(name) == 'q8') {
          return NativeQ8Matrix.quantizeRows(
            shape[0],
            shape[1],
            (first, count, out) => out.setRange(
              0,
              count * shape[1],
              st.readF32(tensor, shape, rowStart: first, rowEnd: first + count),
            ),
          );
        }
        return NativeQ4Matrix.quantize(
          st.readF32(tensor, shape),
          shape[0],
          shape[1],
        );
      },
      done: (m) => (m as NativeMatrix).free(),
    );
  } finally {
    st.close();
  }
}
