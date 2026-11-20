/// The device benchmark's `speech` mode: text to speech with an LFM3-Omni
/// or LFM3-Audio bundle, its generator with float and with int8 product
/// inputs in turn (Lfm3AudioGenerator.load's int8Inputs), so heat affects
/// both alike.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';

const _defaultTexts = [
  (
    'en',
    'Good morning. The meeting starts at ten, so please bring the report.',
  ),
  ('ja', 'おはようございます。会議は十時に始まりますので、資料をお持ちください。'),
];

/// Milliseconds per speech frame of each part, over 30 frames: the depth
/// decoder's sampling (sampleFrame: three depth passes and the sampler), one
/// depth pass by stage (capture copies included), the codec and the
/// language model's step that feeds a frame back.
Future<Map<String, Object?>> _profileFrames(
  TextModel model,
  BpeTokenizer tokenizer,
  Lfm3AudioGenerator g,
  void Function(String) log,
) async {
  const frames = 30;
  final random = math.Random(7);
  final hiddens = [
    for (var i = 0; i < frames; i++)
      Float32List.fromList([
        for (var j = 0; j < g.hiddenSize; j++) math.sin(i * 0.3 + j * 0.01),
      ]),
  ];
  final output = tokenizer.addedTokenId('<|audio_out|>')!;
  await model.reset();
  await model.forward(tokenizer.encode('Hello.'));
  final codec = g.startCodecStream();
  final history = <List<int>>[];
  final clock = Stopwatch();
  var sample = 0, codecUs = 0, lm = 0;
  final stages = <String, int>{};
  try {
    for (final hidden in hiddens) {
      clock
        ..reset()
        ..start();
      final codes = g.sampleFrame(hidden, random, history: history);
      sample += clock.elapsedMicroseconds;
      final audio = [for (final c in codes) c == g.eosCode ? 0 : c];
      history.add(audio);
      clock
        ..reset()
        ..start();
      codec.add(audio);
      codecUs += clock.elapsedMicroseconds;
      clock
        ..reset()
        ..start();
      await model.forward([output], embeddingOverrides: {0: g.feedback(audio)});
      lm += clock.elapsedMicroseconds;
      var last = 0;
      clock
        ..reset()
        ..start();
      g.depthLogits(
        hidden,
        List.filled(g.codebooks, -1),
        only: const [0, 1, 2, 3],
        capture: (stage, _) {
          clock.stop();
          stages[stage] =
              (stages[stage] ?? 0) + clock.elapsedMicroseconds - last;
          last = clock.elapsedMicroseconds;
          clock.start();
        },
      );
    }
  } finally {
    codec.close();
  }
  double ms(int us) => us / frames / 1000;
  final result = {
    'sample_frame_ms': ms(sample),
    'codec_ms': ms(codecUs),
    'language_step_ms': ms(lm),
    'depth_pass_ms': {for (final e in stages.entries) e.key: ms(e.value)},
  };
  log('speech profile ${g.int8Inputs ? 'int8' : 'float'}: $result');
  return result;
}

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
  final model = await TextModel.load(
    '$dir/language',
    maxSeq: 2048,
    weights: WeightFormat.q4,
    q8CachePath: '$dir/language/lfm3-q4.cache',
    gpu: gpu,
  );
  final tokenizer = BpeTokenizer.load('$dir/language');
  final generators = {
    for (final int8 in [false, true])
      int8: Lfm3AudioGenerator.load(speechMmprojIn(dir), int8Inputs: int8),
  };
  record['backend'] = gpu ? 'gpu' : 'cpu';
  record['threads'] = NativeKernels.instance.threads;
  try {
    if (config['profile'] == true) {
      record['profile'] = {
        for (final (i, int8) in [true, false, true, false].indexed)
          '${int8 ? 'int8' : 'float'}_$i': await _profileFrames(
            model,
            tokenizer,
            generators[int8]!,
            log,
          ),
      };
      write();
      if (config['profile_only'] == true) {
        record['finished'] = true;
        return record;
      }
    }
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
