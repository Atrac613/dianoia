/// Preparing and verifying an installed Q4 audio bundle, without audio devices.
library;

import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';

import 'audio_gguf.dart';
import 'audio_generation.dart';
import 'config.dart';
import 'download.dart';
import 'gguf.dart';
import 'native.dart';

Map<String, Object> audioBundleSource(List<RemoteFile> files) => {
  'version': 1,
  'files': {
    for (final f in files) f.name: {'size': f.size, 'sha256': f.sha256},
  },
};

List<String> _payloads(String audio) => [
  'audio_config.json',
  audio,
  for (final name in [
    'config.json',
    'tokenizer.json',
    'model.safetensors',
    'lfm3-q4.cache',
    'LICENSE',
    'audio_source.json',
  ])
    'language/$name',
];

/// Source downloads have already passed size and SHA-256 verification.
/// [output] is unpublished staging storage; the caller publishes atomically.
Future<void> buildAudioGgufBundle(
  String downloads,
  String language,
  String audio,
  List<RemoteFile> sources,
  String output,
) async {
  if (!NativeKernels.isLoaded) NativeKernels.load(null);
  final file = GgufFile.open('$downloads/$audio');
  late final Lfm3AudioEncoderConfig encoder;
  try {
    encoder = Lfm3AudioEncoderConfig.fromGguf(file);
  } finally {
    file.close();
  }
  // Validate the output graph as well as the input encoder before conversion.
  final generator = Lfm3AudioGenerator.load('$downloads/$audio');
  final hidden = generator.hiddenSize;
  generator.close();
  final config =
      jsonDecode(File('$downloads/audio_config.json').readAsStringSync())
          as Map;
  if (config['model_type'] != 'lfm3_audio' || config['format_version'] != 1) {
    throw const FormatException('expected LFM3-Audio format version 1');
  }
  final text = Lfm3Config.fromJson(
    (config['lfm'] as Map).cast<String, Object?>(),
  );
  if (encoder.projectionSize != text.hiddenSize || hidden != text.hiddenSize) {
    throw const FormatException(
      'Audio and language projection dimensions differ',
    );
  }
  final root = Directory(output)..createSync(recursive: true);
  final metadata = Directory('$output/.metadata')..createSync();
  try {
    File(
      '${metadata.path}/config.json',
    ).writeAsStringSync(jsonEncode(config['lfm']));
    File(
      '$downloads/tokenizer.json',
    ).copySync('${metadata.path}/tokenizer.json');
    importGguf('$downloads/$language', metadata.path, '$output/language');
    File('$downloads/LICENSE').copySync('$output/language/LICENSE');
    File('$downloads/audio_config.json').copySync('$output/audio_config.json');
    File('$downloads/$audio').copySync('$output/$audio');
    File('$output/language/audio_source.json').writeAsStringSync(
      jsonEncode({
        'repository': Lfm3AudioQ4Source.repository,
        'revision': Lfm3AudioQ4Source.revision,
        'metadata_revision': Lfm3AudioQ4Source.metadataRevision,
        'files': audioBundleSource(sources)['files'],
        'embedding_adapter': 'Q6_K to Q8',
        'scope': 'Q4 audio input and speech synthesis',
      }),
    );
    final entries = <String, Object>{};
    for (final name in _payloads(audio)) {
      final f = File('${root.path}/$name');
      entries[name] = {
        'size': f.lengthSync(),
        'sha256': (await sha256.bind(f.openRead()).first).toString(),
      };
    }
    File('$output/audio_bundle.json').writeAsStringSync(
      jsonEncode({
        'source': audioBundleSource(sources),
        'audio': audio,
        'files': entries,
      }),
    );
  } finally {
    metadata.deleteSync(recursive: true);
  }
}

/// A quick presence/source check. Loading must also call [validateAudioBundle].
bool audioBundleIsInstalled(
  String directory,
  List<RemoteFile> sources,
  String audio,
) {
  try {
    _manifest(directory, sources, audio);
    return true;
  } on Object {
    return false;
  }
}

Map _manifest(String directory, List<RemoteFile> sources, String audio) {
  final m =
      jsonDecode(File('$directory/audio_bundle.json').readAsStringSync())
          as Map;
  if (jsonEncode(m['source']) != jsonEncode(audioBundleSource(sources)) ||
      m['audio'] != audio) {
    throw const FormatException('Audio bundle source differs');
  }
  final files = m['files'] as Map;
  final expected = _payloads(audio);
  if (files.length != expected.length || !expected.every(files.containsKey)) {
    throw const FormatException('Audio bundle files differ');
  }
  for (final f in sources.where(
    (f) =>
        f.name == audio ||
        ['audio_config.json', 'tokenizer.json', 'LICENSE'].contains(f.name),
  )) {
    final name = f.name == audio || f.name == 'audio_config.json'
        ? f.name
        : 'language/${f.name}';
    if (files[name]['size'] != f.size || files[name]['sha256'] != f.sha256) {
      throw FormatException('Audio bundle source payload differs: $name');
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
      throw FormatException('Audio bundle payload differs: $name');
    }
  }
  return files;
}

/// Full installed payload verification, cancellable between file chunks.
Future<void> validateAudioBundle(
  String directory,
  List<RemoteFile> sources,
  String audio, {
  void Function()? check,
}) async {
  final files = _manifest(directory, sources, audio);
  for (final name in files.keys.cast<String>()) {
    final chunks = File('$directory/$name').openRead().map((chunk) {
      check?.call();
      return chunk;
    });
    if ((await sha256.bind(chunks).first).toString() != files[name]['sha256']) {
      throw FormatException('Audio bundle hash differs: $name');
    }
  }
  check?.call();
}
