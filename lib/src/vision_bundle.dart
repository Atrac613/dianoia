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
/// token embeddings retain their original bytes. Only the cache header/key is
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
        File('$root/language/vl_backbone.json').writeAsStringSync(
          jsonEncode({
            'version': 1,
            'source_key': Q8Cache.keyFor(root),
            'backbone_key': Q8Cache.keyFor('$root/language'),
          }),
        );
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

/// Verifies all required bundle files and their SHA-256 before native loading.
/// The manifest itself must come from a trusted package/install source; these
/// checks detect incomplete or modified files, not manifest authenticity.
Future<void> validateVlBundle(String directory) async {
  final value = jsonDecode(File('$directory/$_manifest').readAsStringSync());
  if (value is! Map ||
      value['version'] != 1 ||
      value['kind'] != 'lfm3-vl' ||
      !['q8', 'q4'].contains(value['weights']) ||
      value['files'] is! Map) {
    throw const FormatException('unsupported VL bundle manifest');
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
  final other = weights == 'q8' ? 'q4' : 'q8';
  if (File('$directory/language/lfm3-$other.cache').existsSync()) {
    throw const FormatException(
      'VL bundle contains an unrecorded alternative cache',
    );
  }
  for (final name in expected) {
    final entry = files[name];
    final file = File('$directory/$name');
    if (entry is! Map ||
        entry['bytes'] is! int ||
        (entry['bytes'] as int) <= 0 ||
        entry['sha256'] is! String ||
        !RegExp(r'^[0-9a-f]{64}$').hasMatch(entry['sha256'] as String) ||
        !file.existsSync() ||
        file.lengthSync() != entry['bytes'] ||
        await _digest(file) != entry['sha256']) {
      throw FormatException('VL bundle file failed integrity check: $name');
    }
  }
  validateVlBackbone(directory, '$directory/language');
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
