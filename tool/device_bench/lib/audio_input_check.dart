/// Measures paced, prerecorded audio input through the streaming ASR path.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';

Future<Map<String, Object?>> audioInputCheck(
  Map<String, Object?> config,
  String docs,
  void Function(String) log,
) async {
  const sampleRate = 16000, chunkSamples = 1280;
  final dir = '$docs/${config['bundle'] as String}';
  final data = ByteData.sublistView(
    File('$docs/${config['clip'] as String}').readAsBytesSync(),
  );
  if (data.lengthInBytes == 0 ||
      data.lengthInBytes % 4 != 0 ||
      data.lengthInBytes > 30 * sampleRate * 4) {
    throw ArgumentError('Expected up to 30 seconds of mono 16 kHz f32le PCM');
  }
  final pcm = Float32List(data.lengthInBytes ~/ 4);
  for (var i = 0; i < pcm.length; i++) {
    pcm[i] = data.getFloat32(i * 4, Endian.little);
  }
  final durationUs = pcm.length * 1000000 ~/ sampleRate;
  final threads = (config['threads'] as num?)?.toInt() ?? 4;
  final gpu = config['backend'] != 'cpu';
  final nativeDownsampling = config['native_downsampling'] != false;
  final compareAttention = config['compare_attention'] != false;
  final output = File('$docs/bench/audio-input-latest.json');
  output.parent.createSync(recursive: true);
  final runs = <Map<String, Object?>>[];
  final record = <String, Object?>{
    'run_id': config['run_id'],
    'bundle': config['bundle'],
    'clip': config['clip'],
    'audio_us': durationUs,
    'chunk_ms': 80,
    'backend': gpu ? 'gpu' : 'cpu',
    'threads': threads,
    'native_downsampling': nativeDownsampling,
    'runs': runs,
    'finished': false,
  };
  void write() => output.writeAsStringSync(jsonEncode(record));
  write();

  NativeKernels.load(null, threads: threads);
  final model = await TextModel.load(
    '$dir/language',
    maxSeq: 2048,
    weights: WeightFormat.q4,
    q8CachePath: '$dir/language/lfm3-q4.cache',
    gpu: gpu,
  );
  Lfm3AudioEncoder? encoder;
  try {
    final tokenizer = Lfm3Tokenizer.load('$dir/language');
    final audioEncoder = encoder = Lfm3AudioEncoder.load(
      speechMmprojIn(dir),
      nativeDownsampling: nativeDownsampling,
    );
    // Warm both paths, then alternate order to reduce warmup and thermal bias.
    final attentionModes = compareAttention
        ? [false, true, false, true, true, false, false, true]
        : [true, true, true, true];
    for (final (index, native) in attentionModes.indexed) {
      NativeKernels.load(null, threads: threads, attention: native);
      final thermalBefore = GpuForward.thermalState;
      final frontend = Lfm3AudioFrontend().startStream();
      final stream = audioEncoder.startStream();
      final input = await AudioTranscriptionPrefill.start(
        model: model,
        tokenizer: tokenizer,
        hiddenSize: audioEncoder.config.projectionSize,
        variant: audioEncoder.config.variant,
        language: config['language'] as String? ?? 'English',
      );
      final clock = Stopwatch()..start();
      final chunks = <Map<String, int>>[];
      var downsampleUs = 0, encodeStart = 0;
      // The first capture follows downsampling and includes the stream's
      // input handling and the capture copy. Later captures are ignored.
      void capture(String stage, Float32List _) {
        if (stage == 'downsample') {
          downsampleUs += clock.elapsedMicroseconds - encodeStart;
        }
      }

      try {
        for (var at = 0; at < pcm.length; at += chunkSamples) {
          final end = math.min(at + chunkSamples, pcm.length);
          final arrival = end * 1000000 ~/ sampleRate;
          final remaining = arrival - clock.elapsedMicroseconds;
          if (remaining > 0) {
            await Future<void>.delayed(Duration(microseconds: remaining));
          }
          final start = clock.elapsedMicroseconds;
          final features = frontend.add(
            Float32List.sublistView(pcm, at, end),
            samplingRate: sampleRate,
          );
          encodeStart = clock.elapsedMicroseconds;
          final rows = stream.add(features, capture: capture);
          final lmStart = clock.elapsedMicroseconds;
          await input.addProjected(rows);
          final done = clock.elapsedMicroseconds;
          chunks.add({
            'arrival_to_start_us': start - arrival,
            'frontend_us': encodeStart - start,
            'encoder_us': lmStart - encodeStart,
            'language_us': done - lmStart,
            'work_us': done - start,
          });
        }

        final tailStart = clock.elapsedMicroseconds;
        final lastFeatures = frontend.finish();
        encodeStart = clock.elapsedMicroseconds;
        final lastRows = stream.add(lastFeatures, capture: capture);
        encodeStart = clock.elapsedMicroseconds;
        final tailRows = stream.finish(capture: capture);
        await input.addProjected(lastRows);
        await input.addProjected(tailRows);
        await input.finish();
        final first = input.greedyToken(model.logits);
        final firstTokenUs = clock.elapsedMicroseconds - durationUs;
        final ids = <int>[];
        var id = first;
        while (ids.length < 64 && !input.stopTokenIds.contains(id)) {
          ids.add(id);
          await model.step(id);
          id = input.greedyToken(model.logits);
        }
        runs.add({
          'native_attention': native,
          'warmup': index < (compareAttention ? 2 : 1),
          'thermal_before': thermalBefore,
          'thermal_after': GpuForward.thermalState,
          'end_to_first_token_us': firstTokenUs,
          'tail_to_first_token_us': firstTokenUs + durationUs - tailStart,
          'downsample_and_stream_setup_us': downsampleUs,
          'audio_tokens': input.audioTokens,
          'chunks': chunks,
          'ids': ids,
          'text': tokenizer.decode(ids),
        });
        write();
        log(
          'Audio input ${index + 1}: end to first token ${firstTokenUs ~/ 1000} ms',
        );
      } finally {
        input.close();
        stream.close();
      }
    }
    record['finished'] = true;
    write();
    return record;
  } finally {
    encoder?.close();
    await model.close();
  }
}
