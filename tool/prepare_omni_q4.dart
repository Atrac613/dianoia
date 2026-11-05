/// Download the pinned official LFM3-Omni Q4_0 pair and prepare a research
/// language bundle. The mmproj stays as published: the audio encoder, speech
/// generator and vision encoder all load from it.
/// Usage: dart run tool/prepare_omni_q4.dart [build/omni-reference]
library;

import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';

Future<void> main(List<String> args) async {
  if (args.length > 1) {
    throw ArgumentError('usage: prepare_omni_q4.dart [directory]');
  }
  final dir = Directory(args.isEmpty ? 'build/omni-reference' : args.single)
    ..createSync(recursive: true);
  final out = '${dir.path}/language';
  if (Directory(out).existsSync()) {
    throw StateError(
      '$out already exists; choose another preparation directory',
    );
  }
  final client = HttpClient();
  try {
    for (final file in [
      ...Lfm3OmniQ4Source.metadataFiles,
      ...Lfm3OmniQ4Source.files,
    ]) {
      stdout.writeln('Verify/download ${file.name} (${file.size} bytes)');
      await downloadFile(client, file, '${dir.path}/${file.name}');
    }
  } finally {
    client.close(force: true);
  }
  NativeKernels.load(null);
  final mmprojPath = '${dir.path}/${Lfm3OmniQ4Source.mmproj.name}';
  final configPath = '${dir.path}/${Lfm3OmniQ4Source.config.name}';
  final config = jsonDecode(File(configPath).readAsStringSync()) as Map;
  if (config['model_type'] != 'lfm3_omni') {
    throw const FormatException('expected an LFM3-Omni config');
  }
  // Validate every consumer of the mmproj before converting the language.
  final mmproj = GgufFile.open(mmprojPath);
  try {
    final audio = Lfm3AudioEncoderConfig.fromGguf(mmproj);
    if (audio.variant != Lfm3SpeechVariant.omni) {
      throw const FormatException('mmproj lacks the Omni vision tower');
    }
    final tokenizer = Lfm3Tokenizer.load(dir.path);
    readVisionGgufWeights(mmproj, loadOmniVisionConfig(configPath, tokenizer));
  } finally {
    mmproj.close();
  }
  Lfm3AudioGenerator.load(mmprojPath).close();
  final temp = dir.createTempSync('.prepare-');
  try {
    final metadata = Directory('${temp.path}/metadata')..createSync();
    File(
      '${metadata.path}/config.json',
    ).writeAsStringSync(jsonEncode(config['text_config']));
    File(
      '${dir.path}/tokenizer.json',
    ).copySync('${metadata.path}/tokenizer.json');
    final staged = '${temp.path}/language';
    importGguf(
      '${dir.path}/${Lfm3OmniQ4Source.language.name}',
      metadata.path,
      staged,
    );
    File('${dir.path}/LICENSE').copySync('$staged/LICENSE');
    File('$staged/omni_source.json').writeAsStringSync(
      '${jsonEncode({
        'repository': Lfm3OmniQ4Source.repository,
        'revision': Lfm3OmniQ4Source.revision,
        'metadata_revision': Lfm3OmniQ4Source.metadataRevision,
        'files': {
          for (final f in [...Lfm3OmniQ4Source.files, ...Lfm3OmniQ4Source.metadataFiles]) f.name: {'size': f.size, 'sha256': f.sha256},
        },
        'embedding_adapter': 'Q6_K to Q8',
        'scope': 'research omni input; not an app installation receipt',
      })}\n',
    );
    Directory(staged).renameSync(out);
  } finally {
    temp.deleteSync(recursive: true);
  }
  stdout.writeln(
    'Prepared $out; audio and vision load from the original mmproj GGUF.',
  );
}
