/// Generate speech without opening audio devices. Output is an ordinary WAV.
/// Usage: `dart run tool/audio_q4_tts.dart <prepared-dir> <text> <output.wav>`
///        [--language=ja] [--voice=female] [--seed=0] [--max-frames=375] [--gpu]
library;

import 'dart:convert';
import 'dart:async';
import 'dart:io';
import 'package:dianoia/lab.dart';

Future<void> main(List<String> args) async {
  final pos = <String>[], options = <String, String>{};
  var gpu = false;
  for (final arg in args) {
    if (arg == '--gpu') {
      gpu = true;
    } else if (arg.startsWith('--') && arg.contains('=')) {
      final at = arg.indexOf('=');
      options[arg.substring(2, at)] = arg.substring(at + 1);
    } else {
      pos.add(arg);
    }
  }
  if (pos.length != 3 ||
      options.keys.any(
        (k) => ![
          'language',
          'voice',
          'seed',
          'max-frames',
          'temperature',
        ].contains(k),
      )) {
    throw ArgumentError(
      'usage: audio_q4_tts.dart <prepared-dir> <text> <output.wav> [--language=ja] [--voice=female] [--gpu]',
    );
  }
  final dir = pos[0];
  NativeKernels.load(null);
  final generator = Lfm3AudioGenerator.load(
    '$dir/${Lfm3AudioQ4Source.audio.name}',
  );
  Lfm3Model? model;
  StreamSubscription<ProcessSignal>? interrupt;
  try {
    model = await Lfm3Model.load(
      '$dir/language',
      maxSeq: 2048,
      weights: WeightFormat.q4,
      q8CachePath: '$dir/language/lfm3-q4.cache',
      gpu: gpu,
    );
    final synth = AudioSynthesizer(
      model,
      Lfm3Tokenizer.load('$dir/language'),
      generator,
    );
    interrupt = ProcessSignal.sigint.watch().listen((_) => synth.cancel());
    final watch = Stopwatch()..start();
    final result = await synth.synthesize(
      pos[1],
      language: options['language'] ?? 'en',
      voice: options['voice'] ?? 'female',
      seed: int.parse(options['seed'] ?? '0'),
      maxFrames: int.parse(options['max-frames'] ?? '375'),
      temperature: double.parse(options['temperature'] ?? '1'),
    );
    final output = File(pos[2]);
    output.parent.createSync(recursive: true);
    output.writeAsBytesSync(encodeSpeechWav(result.samples));
    stdout.writeln(
      const JsonEncoder.withIndent('  ').convert({
        'output': output.absolute.path,
        'sample_rate': result.sampleRate,
        'frames': result.frames,
        'audio_seconds': result.seconds,
        'stop_reason': result.stopReason,
        'wall_seconds': watch.elapsedMicroseconds / 1e6,
        'peak': result.samples.fold(0.0, (m, x) => x.abs() > m ? x.abs() : m),
        'clipped_samples': result.samples.where((x) => x.abs() > 1).length,
      }),
    );
    if (result.stopReason != 'eos') exitCode = 2;
  } finally {
    await interrupt?.cancel();
    generator.close();
    await model?.close();
  }
}
