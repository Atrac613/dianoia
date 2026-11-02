/// Portable VL bundles containing only vision weights and a packed backbone.
library;

import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';

import 'config.dart';
import 'native.dart';
import 'q8_cache.dart';
import 'quant.dart';
import 'safetensors.dart';
import 'vision_backbone.dart';
import 'vision_config.dart';
import 'vision_encoder.dart';
import 'vision_image.dart';

const _manifest = 'vl_bundle.json';

List<String> _files(String weights) => [
  'config.json',
  'processor_config.json',
  'tokenizer.json',
  'LICENSE',
  'model.safetensors',
  'language/config.json',
  'language/tokenizer.json',
  'language/model.safetensors',
  'language/lfm3-$weights.cache',
  'language/vl_backbone.json',
];

Future<String> _digest(File file) async =>
    (await sha256.bind(file.openRead()).first).toString();

/// Packages a VL checkpoint and an existing Q8/Q4 language cache into [outDir].
/// No matrix is requantized. Vision tensors, small language tensors and BF16
/// token embeddings retain their original bytes. [useCacheEmbedding] explicitly
/// selects Q8 lookup for GGUF imports that have no BF16 embedding table. Only the cache header/key is
/// rebuilt for the packed layout. The full original checkpoint and imported
/// language matrices are unnecessary at runtime.
///
/// Writes to a sibling staging directory and publishes only when every file
/// and its SHA-256 manifest are complete. Never overwrites existing output.
Future<void> packVlBundle(
  String modelDir,
  String backboneDir,
  String outDir, {
  WeightFormat weights = WeightFormat.q8,
  bool useCacheEmbedding = false,
}) async {
  if (!weights.isNative) throw ArgumentError('VL packing needs Q8 or Q4');
  validateVlBackbone(modelDir, backboneDir);
  final config = Lfm3VlConfig.load(modelDir);
  final text = Lfm3Config.load(backboneDir);
  final processor = Lfm3ImageProcessor.load(modelDir);
  if (processor.patchSize != config.vision.patchSize ||
      processor.mergeSize != config.vision.spatialMergeSize ||
      processor.temporalPatchSize != config.vision.temporalPatchSize ||
      config.vision.inChannels != 3) {
    throw const FormatException('processor and vision configuration differ');
  }
  for (final name in [
    'config.json',
    'processor_config.json',
    'tokenizer.json',
    'LICENSE',
  ]) {
    if (!File('$modelDir/$name').existsSync()) {
      throw FileSystemException('VL checkpoint lacks $name', modelDir);
    }
  }
  final destination = Directory(outDir).absolute;
  if (FileSystemEntity.typeSync(destination.path, followLinks: false) !=
      FileSystemEntityType.notFound) {
    throw FileSystemException('output already exists', outDir);
  }
  final cache = Q8Cache.open(
    '$backboneDir/lfm3-${weights.name}.cache',
    Q8Cache.keyFor(backboneDir, weights: weights.name),
  );
  if (cache == null) {
    throw StateError('build a matching ${weights.name} cache before packing');
  }
  Directory? staging;
  try {
    final vision = SafetensorsFile.open('$modelDir/model.safetensors');
    try {
      final visionShapes = lfm3VisionTensorShapes(config);
      final sourceLength = File('$modelDir/model.safetensors').lengthSync();
      for (final entry in visionShapes.entries) {
        _checkTensor(vision, entry.key, entry.value, sourceLength);
      }
      final language = SafetensorsFile.open('$backboneDir/model.safetensors');
      try {
        final matrices = <String, NativeMatrix>{};
        final smallNames = <String>{};
        final languageLength = File(
          '$backboneDir/model.safetensors',
        ).lengthSync();
        for (final entry in lfm3LanguageTensorShapes(text).entries) {
          final name = entry.key, shape = entry.value;
          if (shape.length == 2) {
            final cachedName =
                name.contains('.self_attn.') || name.contains('.feed_forward.')
                ? name.substring(0, name.length - '.weight'.length)
                : name;
            _checkCachedMatrix(cache, cachedName, shape[0], shape[1]);
            matrices[cachedName] = cache.matrix(
              cachedName,
              shape[0],
              shape[1],
            )!;
          }
          if (name == 'model.embed_tokens.weight' && useCacheEmbedding) {
            if (language.tensors.containsKey(name) ||
                matrices[name]!.format != 'q8') {
              throw const FormatException(
                'cache embedding requires an absent BF16 table and Q8 head',
              );
            }
            continue;
          }
          if (shape.length != 2 || name == 'model.embed_tokens.weight') {
            _checkTensor(
              language,
              name,
              shape,
              languageLength,
              requireBf16: name == 'model.embed_tokens.weight',
            );
            smallNames.add(name);
          }
        }
        destination.parent.createSync(recursive: true);
        staging = destination.parent.createTempSync('.dianoia-vl-bundle-');
        final root = staging.path;
        Directory('$root/language').createSync();
        for (final name in [
          'config.json',
          'processor_config.json',
          'tokenizer.json',
          'LICENSE',
        ]) {
          File('$modelDir/$name').copySync('$root/$name');
        }
        File('$backboneDir/config.json').copySync('$root/language/config.json');
        File(
          '$modelDir/tokenizer.json',
        ).copySync('$root/language/tokenizer.json');
        vision.writeSubset(
          '$root/model.safetensors',
          (name, _) => visionShapes.containsKey(name),
        );
        language.writeSubset(
          '$root/language/model.safetensors',
          (name, _) => smallNames.contains(name),
        );
        Q8Cache.write(
          '$root/language/lfm3-${weights.name}.cache',
          Q8Cache.keyFor('$root/language', weights: weights.name),
          matrices,
        );
        writeVlBackboneIdentity(root, '$root/language');
        final files = <String, Object?>{};
        for (final name in _files(weights.name)) {
          final file = File('$root/$name');
          files[name] = {
            'bytes': file.lengthSync(),
            'sha256': await _digest(file),
          };
        }
        File('$root/$_manifest').writeAsStringSync(
          const JsonEncoder.withIndent('  ').convert({
            'version': 1,
            'kind': 'lfm3-vl',
            'weights': weights.name,
            if (useCacheEmbedding) 'input_embedding': 'q8-cache',
            'origin_source_key': Q8Cache.keyFor(modelDir),
            'files': files,
          }),
        );
      } finally {
        language.close();
      }
    } finally {
      vision.close();
    }
    staging.renameSync(destination.path);
    staging = null;
  } finally {
    cache.close();
    if (staging != null) staging.deleteSync(recursive: true);
  }
}

/// Validated version-one layout, shared by installers and runtime loaders.
final class VlBundleManifest {
  VlBundleManifest._(this.weights, this.files, this.inputEmbedding);

  final String weights;
  final String inputEmbedding;
  final Map<String, ({int bytes, String sha256})> files;

  factory VlBundleManifest.parse(String json) {
    final value = jsonDecode(json);
    if (value is! Map ||
        value['version'] != 1 ||
        value['kind'] != 'lfm3-vl' ||
        !['q8', 'q4'].contains(value['weights']) ||
        value['files'] is! Map) {
      throw const FormatException('unsupported VL bundle manifest');
    }
    final embedding = value['input_embedding'] ?? 'bf16';
    if (!['bf16', 'q8-cache'].contains(embedding)) {
      throw const FormatException('unsupported VL input embedding policy');
    }
    final weights = value['weights'] as String;
    final files = value['files'] as Map;
    final expected = _files(weights);
    if (files.length != expected.length ||
        expected.any((name) => !files.containsKey(name))) {
      throw const FormatException(
        'VL bundle file list differs from required layout',
      );
    }
    final checked = <String, ({int bytes, String sha256})>{};
    for (final name in expected) {
      final entry = files[name];
      if (entry is! Map ||
          entry['bytes'] is! int ||
          (entry['bytes'] as int) <= 0 ||
          entry['sha256'] is! String ||
          !RegExp(r'^[0-9a-f]{64}$').hasMatch(entry['sha256'] as String)) {
        throw FormatException('invalid VL bundle manifest entry: $name');
      }
      checked[name] = (
        bytes: entry['bytes'] as int,
        sha256: entry['sha256'] as String,
      );
    }
    return VlBundleManifest._(
      weights,
      Map.unmodifiable(checked),
      embedding as String,
    );
  }
}

/// Verifies all required bundle files and their SHA-256 before native loading.
/// The manifest itself must come from a trusted package/install source; these
/// checks detect incomplete or modified files, not manifest authenticity.
Future<void> validateVlBundle(String directory) async {
  final manifest = VlBundleManifest.parse(
    File('$directory/$_manifest').readAsStringSync(),
  );
  final other = manifest.weights == 'q8' ? 'q4' : 'q8';
  if (File('$directory/language/lfm3-$other.cache').existsSync()) {
    throw const FormatException(
      'VL bundle contains an unrecorded alternative cache',
    );
  }
  for (final entry in manifest.files.entries) {
    final file = File('$directory/${entry.key}');
    if (!file.existsSync() ||
        file.lengthSync() != entry.value.bytes ||
        await _digest(file) != entry.value.sha256) {
      throw FormatException(
        'VL bundle file failed integrity check: ${entry.key}',
      );
    }
  }
  validateVlBackbone(directory, '$directory/language');
  if (manifest.inputEmbedding == 'bf16') {
    final c = Lfm3Config.load('$directory/language');
    final st = SafetensorsFile.open('$directory/language/model.safetensors');
    try {
      _checkTensor(
        st,
        'model.embed_tokens.weight',
        [c.vocabSize, c.hiddenSize],
        File('$directory/language/model.safetensors').lengthSync(),
        requireBf16: true,
      );
    } finally {
      st.close();
    }
  }
  if (manifest.inputEmbedding == 'q8-cache') {
    final language = '$directory/language';
    final st = SafetensorsFile.open('$language/model.safetensors');
    try {
      if (st.tensors.containsKey('model.embed_tokens.weight')) {
        throw const FormatException(
          'cache embedding bundle also contains a BF16 table',
        );
      }
    } finally {
      st.close();
    }
    final c = Lfm3Config.load(language);
    final cache = Q8Cache.open(
      '$language/lfm3-${manifest.weights}.cache',
      Q8Cache.keyFor(language, weights: manifest.weights),
    );
    if (cache == null) {
      throw const FormatException('invalid cache embedding cache');
    }
    try {
      _checkCachedMatrix(
        cache,
        'model.embed_tokens.weight',
        c.vocabSize,
        c.hiddenSize,
      );
      if (cache
              .offsets('model.embed_tokens.weight', c.vocabSize, c.hiddenSize)!
              .format !=
          'q8') {
        throw const FormatException('input embedding cache must be Q8');
      }
    } finally {
      cache.close();
    }
  }
}

void _checkTensor(
  SafetensorsFile st,
  String name,
  List<int> shape,
  int fileLength, {
  bool requireBf16 = false,
}) {
  final info = st.tensors[name];
  if (info == null ||
      shape.any((n) => n <= 0) ||
      info.shape.join(',') != shape.join(',') ||
      (requireBf16
          ? info.dtype != 'BF16'
          : !['BF16', 'F32'].contains(info.dtype))) {
    throw FormatException(
      'cannot pack tensor $name with expected shape $shape',
    );
  }
  final bytes = info.numel * (info.dtype == 'BF16' ? 2 : 4);
  if (info.begin < 0 ||
      info.end - info.begin != bytes ||
      st.dataStart + info.end > fileLength) {
    throw FormatException('cannot pack tensor $name with invalid byte range');
  }
}

void _checkCachedMatrix(Q8Cache cache, String name, int rows, int cols) {
  final offsets = cache.offsets(name, rows, cols);
  if (rows <= 0 ||
      cols <= 0 ||
      cols % 32 != 0 ||
      offsets == null ||
      !['q8', 'q4'].contains(offsets.format)) {
    throw FormatException('cannot pack cache without $name [$rows, $cols]');
  }
  final qsBytes = rows * cols ~/ (offsets.format == 'q4' ? 2 : 1);
  final alignment = offsets.format == 'q8' ? 4 : 2;
  if (offsets.qs < 0 ||
      offsets.qs + qsBytes > cache.file.length ||
      offsets.scales < 0 ||
      offsets.scales % alignment != 0 ||
      offsets.end > cache.file.length) {
    throw FormatException(
      'cannot pack cache matrix $name with invalid byte range',
    );
  }
}
