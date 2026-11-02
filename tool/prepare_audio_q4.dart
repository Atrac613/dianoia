/// Download the pinned official Q4_0 pair and prepare a research language bundle.
/// Usage: dart run tool/prepare_audio_q4.dart [build/audio-reference]
library;

import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';

Future<void> main(List<String> args) async {
  if (args.length > 1) {
    throw ArgumentError('usage: prepare_audio_q4.dart [directory]');
  }
  final dir = Directory(args.isEmpty ? 'build/audio-reference' : args.single)
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
      ...Lfm3AudioQ4Source.metadataFiles,
      ...Lfm3AudioQ4Source.files,
    ]) {
      stdout.writeln('Verify/download ${file.name} (${file.size} bytes)');
      await downloadFile(client, file, '${dir.path}/${file.name}');
    }
  } finally {
    client.close(force: true);
  }
  NativeKernels.load(null);
  final audio = GgufFile.open('${dir.path}/${Lfm3AudioQ4Source.audio.name}');
  try {
    Lfm3AudioEncoderConfig.fromGguf(audio);
  } finally {
    audio.close();
  }
  final config =
      jsonDecode(File('${dir.path}/audio_config.json').readAsStringSync())
          as Map;
  if (config['model_type'] != 'lfm3_audio' || config['format_version'] != 1) {
    throw const FormatException('expected LFM3-Audio format version 1');
  }
  final temp = dir.createTempSync('.prepare-');
  try {
    final metadata = Directory('${temp.path}/metadata')..createSync();
    File(
      '${metadata.path}/config.json',
    ).writeAsStringSync(jsonEncode(config['lfm']));
    File(
      '${dir.path}/tokenizer.json',
    ).copySync('${metadata.path}/tokenizer.json');
    final staged = '${temp.path}/language';
    importGguf(
      '${dir.path}/${Lfm3AudioQ4Source.language.name}',
      metadata.path,
      staged,
    );
    File('${dir.path}/LICENSE').copySync('$staged/LICENSE');
    File('$staged/audio_source.json').writeAsStringSync(
      '${jsonEncode({
        'repository': Lfm3AudioQ4Source.repository,
        'revision': Lfm3AudioQ4Source.revision,
        'metadata_revision': Lfm3AudioQ4Source.metadataRevision,
        'files': {
          for (final f in [...Lfm3AudioQ4Source.files, ...Lfm3AudioQ4Source.metadataFiles]) f.name: {'size': f.size, 'sha256': f.sha256},
        },
        'embedding_adapter': 'Q6_K to Q8',
        'scope': 'research audio input; not an app installation receipt',
      })}\n',
    );
    Directory(staged).renameSync(out);
  } finally {
    temp.deleteSync(recursive: true);
  }
  stdout.writeln('Prepared $out; audio encoder uses the original mmproj GGUF.');
}
