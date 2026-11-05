/// Preparing and verifying an installed LFM3-Omni Q4 bundle.
///
/// The bundle has the research directory's layout, so the lab tools accept
/// it as is: `language/` (the imported Q4 backbone and tokenizer), the
/// original mmproj, `omni_config.json` and `processor_config.json`.
library;

import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';

import 'audio_generation.dart';
import 'audio_gguf.dart';
import 'download.dart';
import 'gguf.dart';
import 'native.dart';
import 'omni_gguf.dart';
import 'tokenizer.dart';
import 'vision_gguf.dart';

const _manifestName = 'omni_bundle.json';

List<String> _payloads(String mmproj) => [
  'omni_config.json',
  'processor_config.json',
  mmproj,
  for (final name in [
    'config.json',
    'tokenizer.json',
    'model.safetensors',
    'lfm3-q4.cache',
    'LICENSE',
    'omni_source.json',
  ])
    'language/$name',
];

Map<String, Object> _source(List<RemoteFile> files) => {
  'version': 1,
  'files': {
    for (final f in files) f.name: {'size': f.size, 'sha256': f.sha256},
  },
};

Future<String> _hash(File file) async =>
    (await sha256.bind(file.openRead()).first).toString();

/// Builds an Omni bundle from verified [downloads] into unpublished
/// staging storage [output]; the caller publishes it atomically. Every
/// consumer of the mmproj (audio encoder, speech generator, vision tower) is
/// validated before the language GGUF is converted.
Future<void> buildOmniGgufBundle(
  String downloads,
  String language,
  String mmproj,
  List<RemoteFile> sources,
  String output,
) async {
  if (!NativeKernels.isLoaded) NativeKernels.load(null);
  final configPath = '$downloads/omni_config.json';
  final config = jsonDecode(File(configPath).readAsStringSync()) as Map;
  if (config['model_type'] != 'lfm3_omni') {
    throw const FormatException('expected an LFM3-Omni config');
  }
  final tokenizerDir = Directory('$output/.tokenizer')
    ..createSync(recursive: true);
  try {
    File(
      '$downloads/tokenizer.json',
    ).copySync('${tokenizerDir.path}/tokenizer.json');
    final vision = loadOmniVisionConfig(
      configPath,
      Lfm3Tokenizer.load(tokenizerDir.path),
    );
    final file = GgufFile.open('$downloads/$mmproj');
    try {
      final audio = Lfm3AudioEncoderConfig.fromGguf(file);
      if (audio.variant != Lfm3SpeechVariant.omni ||
          audio.projectionSize != vision.text.hiddenSize) {
        throw const FormatException('mmproj is not an LFM3-Omni projector');
      }
      readVisionGgufWeights(file, vision);
    } finally {
      file.close();
    }
    final generator = Lfm3AudioGenerator.load('$downloads/$mmproj');
    final hidden = generator.hiddenSize;
    generator.close();
    if (hidden != vision.text.hiddenSize) {
      throw const FormatException('speech and language dimensions differ');
    }
  } finally {
    tokenizerDir.deleteSync(recursive: true);
  }
  final metadata = Directory('$output/.metadata')..createSync();
  try {
    File(
      '${metadata.path}/config.json',
    ).writeAsStringSync(jsonEncode(config['text_config']));
    File(
      '$downloads/tokenizer.json',
    ).copySync('${metadata.path}/tokenizer.json');
    importGguf('$downloads/$language', metadata.path, '$output/language');
    File('$downloads/LICENSE').copySync('$output/language/LICENSE');
    for (final name in ['omni_config.json', 'processor_config.json', mmproj]) {
      File('$downloads/$name').copySync('$output/$name');
    }
    File('$output/language/omni_source.json').writeAsStringSync(
      jsonEncode({
        'repository': Lfm3OmniQ4Source.repository,
        'revision': Lfm3OmniQ4Source.revision,
        'metadata_revision': Lfm3OmniQ4Source.metadataRevision,
        'files': _source(sources)['files'],
        'embedding_adapter': 'Q6_K to Q8',
        'scope': 'Q4 omni input, speech and text output',
      }),
    );
    File('$output/$_manifestName').writeAsStringSync(
      jsonEncode({
        'source': _source(sources),
        'mmproj': mmproj,
        'files': {
          for (final name in _payloads(mmproj))
            name: {
              'size': File('$output/$name').lengthSync(),
              'sha256': await _hash(File('$output/$name')),
            },
        },
      }),
    );
  } finally {
    metadata.deleteSync(recursive: true);
  }
}

Map _manifest(String directory, List<RemoteFile> sources, String mmproj) {
  final m =
      jsonDecode(File('$directory/$_manifestName').readAsStringSync()) as Map;
  if (jsonEncode(m['source']) != jsonEncode(_source(sources)) ||
      m['mmproj'] != mmproj) {
    throw const FormatException('Omni bundle source differs');
  }
  final files = m['files'] as Map;
  final expected = _payloads(mmproj);
  if (files.length != expected.length || !expected.every(files.containsKey)) {
    throw const FormatException('Omni bundle files differ');
  }
  // Files copied unchanged must still be the pinned sources.
  for (final f in sources) {
    final name = switch (f.name) {
      'tokenizer.json' || 'LICENSE' => 'language/${f.name}',
      _ => f.name,
    };
    if (!files.containsKey(name)) continue;
    if (files[name]['size'] != f.size || files[name]['sha256'] != f.sha256) {
      throw FormatException('Omni bundle source payload differs: $name');
    }
  }
  for (final name in expected) {
    final f = File('$directory/$name'), spec = files[name] as Map;
    if (spec['size'] is! int ||
        spec['size'] <= 0 ||
        !f.existsSync() ||
        f.lengthSync() != spec['size'] ||
        spec['sha256'] is! String ||
        !RegExp(r'^[a-f0-9]{64}$').hasMatch(spec['sha256'] as String)) {
      throw FormatException('Omni bundle payload differs: $name');
    }
  }
  return files;
}

/// A quick presence and source check. Loading must also call
/// [validateOmniBundle].
bool omniBundleIsInstalled(
  String directory,
  List<RemoteFile> sources,
  String mmproj,
) {
  try {
    _manifest(directory, sources, mmproj);
    return true;
  } on Object {
    return false;
  }
}

/// Full installed payload verification, cancellable between file chunks.
Future<void> validateOmniBundle(
  String directory,
  List<RemoteFile> sources,
  String mmproj, {
  void Function()? check,
}) async {
  final files = _manifest(directory, sources, mmproj);
  for (final name in files.keys.cast<String>()) {
    final chunks = File('$directory/$name').openRead().map((chunk) {
      check?.call();
      return chunk;
    });
    if ((await sha256.bind(chunks).first).toString() != files[name]['sha256']) {
      throw FormatException('Omni bundle hash differs: $name');
    }
  }
  check?.call();
}
