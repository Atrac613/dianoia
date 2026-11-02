/// Research ASR on a prepared official Q4_0 pair and mono 16 kHz f32le PCM.
/// Usage: `dart run tool/audio_q4_transcribe.dart <directory> <pcm.f32>`
///        [English|Japanese|...] [--gpu]
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';

Future<void> main(List<String> args) async {
  final positional = args.where((a) => a != '--gpu').toList();
  if (positional.length < 2 || positional.length > 3) {
    throw ArgumentError(
      'usage: audio_q4_transcribe.dart <directory> <pcm.f32> [language] [--gpu]',
    );
  }
  final dir = positional[0], path = positional[1];
  final file = File(path);
  final size = file.lengthSync();
  if (size <= 0 || size % 4 != 0 || size > 30 * 16000 * 4) {
    throw ArgumentError(
      'expected 0 < duration <= 30 seconds of mono 16 kHz f32le PCM',
    );
  }
  final bytes = ByteData.sublistView(file.readAsBytesSync());
  final pcm = Float32List(size ~/ 4);
  for (var i = 0; i < pcm.length; i++) {
    pcm[i] = bytes.getFloat32(i * 4, Endian.little);
  }
  NativeKernels.load(null);
  final sw = Stopwatch()..start();
  final features = Lfm3AudioFrontend().extract(pcm, samplingRate: 16000);
  final frontendSeconds = sw.elapsedMicroseconds / 1e6;
  final encoder = Lfm3AudioEncoder.load('$dir/${Lfm3AudioQ4Source.audio.name}');
  late Float32List projected;
  final encodeTimer = Stopwatch()..start();
  try {
    projected = encoder.encode(features);
  } finally {
    encoder.close();
  }
  final encoderSeconds = encodeTimer.elapsedMicroseconds / 1e6;
  final bundle = '$dir/language';
  final tokenizer = Lfm3Tokenizer.load(bundle);
  final prompt = prepareAudioTranscriptionPrompt(
    tokenizer: tokenizer,
    audioTokens: features.encoderTokens,
    hiddenSize: encoder.config.projectionSize,
    language: positional.length == 3 ? positional[2] : null,
  );
  final gpu = args.contains('--gpu');
  final model = await Lfm3Model.load(
    bundle,
    maxSeq: 4096,
    weights: WeightFormat.q4,
    q8CachePath: '$bundle/lfm3-q4.cache',
    gpu: gpu,
  );
  final ids = <int>[];
  var stopReason = 'max_new_tokens';
  final prefillTimer = Stopwatch()..start();
  double firstTokenSeconds = 0;
  try {
    if (prompt.ids.length + 128 > model.maxSeq) {
      throw ArgumentError('audio prompt exceeds context');
    }
    await prompt.prefill(model, projected);
    firstTokenSeconds = prefillTimer.elapsedMicroseconds / 1e6;
    for (var i = 0; i < 128; i++) {
      final next = prompt.greedyToken(model.logits);
      if (prompt.stopTokenIds.contains(next)) {
        stopReason = 'eos';
        break;
      }
      ids.add(next);
      if (i + 1 < 128) await model.step(next);
    }
  } finally {
    await model.close();
  }
  stdout.writeln(
    const JsonEncoder.withIndent('  ').convert({
      'model': 'LFM3-Audio-2B-Q4_0',
      'source_revision': Lfm3AudioQ4Source.revision,
      'backend': gpu ? 'gpu' : 'cpu',
      'audio_seconds': pcm.length / 16000,
      'audio_tokens': features.encoderTokens,
      'prompt_tokens': prompt.ids.length,
      'generated_ids': ids,
      'text': tokenizer.decode(ids),
      'stop_reason': stopReason,
      'frontend_seconds': frontendSeconds,
      'encoder_seconds': encoderSeconds,
      'language_prefill_seconds': firstTokenSeconds,
      'language_prefill_decode_seconds': prefillTimer.elapsedMicroseconds / 1e6,
    }),
  );
}
