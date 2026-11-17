/// The device benchmark's `speech` mode: text to speech with an LFM3-Omni
/// or LFM3-Audio bundle, its generator with float and with int8 product
/// inputs in turn (Lfm3AudioGenerator.load's int8Inputs), so heat affects
/// both alike.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:dianoia/lab.dart';

const _defaultTexts = [
  (
    'en',
    'Good morning. The meeting starts at ten, so please bring the report.',
  ),
  ('ja', 'おはようございます。会議は十時に始まりますので、資料をお持ちください。'),
];

/// Synthesizes each text of [config] (paths relative to [docs]) with both
/// generators in ABBA order and writes the record, finished or not, to
/// `bench/speech-latest.json` after every run, with each waveform as
/// `bench/speech-<language>-<inputs>-<run>.wav`.
Future<Map<String, Object?>> speechCheck(
  Map<String, Object?> config,
  String docs,
  void Function(String) log,
) async {
  final output = File('$docs/bench/speech-latest.json');
  output.parent.createSync(recursive: true);
  final runs = <Map<String, Object?>>[];
  final record = <String, Object?>{
    'started': DateTime.now().toIso8601String(),
    'bundle': config['bundle'],
    'finished': false,
    'runs': runs,
  };
  void write() => output.writeAsStringSync(
    const JsonEncoder.withIndent('  ').convert(record),
  );
  write();
  final dir = '$docs/${config['bundle'] as String}';
  final gpu = config['backend'] != 'cpu';
  NativeKernels.load(null, threads: (config['threads'] as num?)?.toInt() ?? 4);
  final model = await Lfm3Model.load(
    '$dir/language',
    maxSeq: 2048,
    weights: WeightFormat.q4,
    q8CachePath: '$dir/language/lfm3-q4.cache',
    gpu: gpu,
  );
  final tokenizer = Lfm3Tokenizer.load('$dir/language');
  final generators = {
    for (final int8 in [false, true])
      int8: Lfm3AudioGenerator.load(speechMmprojIn(dir), int8Inputs: int8),
  };
  record['backend'] = gpu ? 'gpu' : 'cpu';
  try {
    final texts = [
      for (final t in (config['texts'] as List?) ?? const [])
        ((t as Map)['language'] as String, t['text'] as String),
    ];
    for (final (language, text) in texts.isEmpty ? _defaultTexts : texts) {
      for (final (i, int8) in [false, true, true, false].indexed) {
        final synth = AudioSynthesizer(model, tokenizer, generators[int8]!);
        final thermal = GpuForward.thermalState;
        final clock = Stopwatch()..start();
        final result = await synth.synthesize(
          text,
          language: language,
          voice: 'female',
          seed: 0,
        );
        final wall = clock.elapsedMicroseconds / 1e6;
        final inputs = int8 ? 'int8' : 'float';
        File(
          '$docs/bench/speech-$language-$inputs-$i.wav',
        ).writeAsBytesSync(encodeSpeechWav(result.samples));
        runs.add({
          'language': language,
          'inputs': inputs,
          'run': i,
          'frames': result.frames,
          'audio_seconds': result.seconds,
          'wall_seconds': wall,
          'real_time_factor': wall / math.max(result.seconds, 1e-9),
          'stop_reason': result.stopReason,
          'thermal_before': thermal,
          'thermal_after': GpuForward.thermalState,
          'footprint_mb': physFootprint() >> 20,
        });
        log(
          'speech $language $inputs: ${result.seconds.toStringAsFixed(2)} s '
          'of audio in ${wall.toStringAsFixed(2)} s, thermal '
          '$thermal→${GpuForward.thermalState}',
        );
        write();
      }
    }
    record['finished'] = true;
  } catch (e) {
    record['error'] = '$e';
    record['finished'] = true;
  } finally {
    for (final g in generators.values) {
      g.close();
    }
    await model.close();
    write();
  }
  return record;
}
