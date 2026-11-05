/// Research ASR on a prepared official Q4_0 pair and mono 16 kHz f32le PCM.
/// Usage: `dart run tool/audio_q4_transcribe.dart <directory> <pcm.f32>`
///        [English|Japanese|...] [--gpu] [--chunk-ms=80]
/// Chunk mode simulates unpaced arriving audio; it does not capture a microphone.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';

T _timed<T>(Stopwatch timer, T Function() action) {
  timer.start();
  try {
    return action();
  } finally {
    timer.stop();
  }
}

Future<T> _timedAsync<T>(Stopwatch timer, Future<T> Function() action) async {
  timer.start();
  try {
    return await action();
  } finally {
    timer.stop();
  }
}

Future<void> main(List<String> args) async {
  final positional = <String>[];
  var gpu = false;
  int? chunkMs;
  for (final arg in args) {
    if (arg == '--gpu') {
      gpu = true;
    } else if (arg.startsWith('--chunk-ms=')) {
      final value = int.tryParse(arg.substring('--chunk-ms='.length));
      if (value == null || value < 1 || value > 1000 || chunkMs != null) {
        throw ArgumentError(
          'specify --chunk-ms once, with an integer from 1 to 1000',
        );
      }
      chunkMs = value;
    } else if (arg.startsWith('--')) {
      throw ArgumentError('unknown option $arg');
    } else {
      positional.add(arg);
    }
  }
  if (positional.length < 2 || positional.length > 3) {
    throw ArgumentError(
      'usage: audio_q4_transcribe.dart <directory> <pcm.f32> [language] [--gpu] [--chunk-ms=80]',
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
  final encoder = Lfm3AudioEncoder.load(speechMmprojIn(dir));
  final omni = encoder.config.variant == Lfm3SpeechVariant.omni;
  final bundle = '$dir/language';
  Lfm3Model? model;
  AudioTranscriptionPrefill? input;
  Lfm3AudioEncoderStream? stream;
  final frontendTimer = Stopwatch(),
      encoderTimer = Stopwatch(),
      languageTimer = Stopwatch();
  final ids = <int>[];
  var stopReason = 'max_new_tokens';
  late int audioTokens, promptTokens;
  var beforeFinish = 0, peakCacheBytes = 0, inputChunks = 0;
  late double inputProcessingSeconds, prefillSeconds;
  late String text;
  try {
    final tokenizer = Lfm3Tokenizer.load(bundle);
    model = await Lfm3Model.load(
      bundle,
      maxSeq: 4096,
      weights: WeightFormat.q4,
      q8CachePath: '$bundle/lfm3-q4.cache',
      gpu: gpu,
    );
    final language = positional.length == 3 ? positional[2] : null;
    final frontend = Lfm3AudioFrontend();
    final inputTimer = Stopwatch()..start();
    late int Function(Float32List) greedy;
    late Set<int> stops;
    if (chunkMs == null) {
      final features = _timed(
        frontendTimer,
        () => frontend.extract(pcm, samplingRate: 16000),
      );
      final projected = _timed(encoderTimer, () => encoder.encode(features));
      final prompt = prepareAudioTranscriptionPrompt(
        tokenizer: tokenizer,
        audioTokens: features.encoderTokens,
        hiddenSize: encoder.config.projectionSize,
        language: language,
        variant: encoder.config.variant,
      );
      if (prompt.ids.length + 128 > model.maxSeq) {
        throw ArgumentError('audio prompt exceeds context');
      }
      await _timedAsync(languageTimer, () => prompt.prefill(model!, projected));
      audioTokens = features.encoderTokens;
      promptTokens = prompt.ids.length;
      greedy = prompt.greedyToken;
      stops = prompt.stopTokenIds;
      inputChunks = 1;
    } else {
      final features = frontend.startStream();
      final currentStream = encoder.startStream();
      stream = currentStream;
      final currentInput = await _timedAsync(
        languageTimer,
        () => AudioTranscriptionPrefill.start(
          model: model!,
          tokenizer: tokenizer,
          hiddenSize: encoder.config.projectionSize,
          language: language,
          variant: encoder.config.variant,
        ),
      );
      input = currentInput;
      final chunkSamples = chunkMs * 16;
      for (var at = 0; at < pcm.length; at += chunkSamples) {
        final frame = _timed(
          frontendTimer,
          () => features.add(
            pcm.sublist(at, math.min(at + chunkSamples, pcm.length)),
            samplingRate: 16000,
          ),
        );
        final projected = _timed(encoderTimer, () => currentStream.add(frame));
        peakCacheBytes = math.max(
          peakCacheBytes,
          currentStream.cachedStateBytes,
        );
        await _timedAsync(
          languageTimer,
          () => currentInput.addProjected(projected),
        );
        inputChunks++;
      }
      beforeFinish = currentInput.audioTokens;
      final tail = _timed(frontendTimer, features.finish);
      final complete = _timed(encoderTimer, () => currentStream.add(tail));
      peakCacheBytes = math.max(peakCacheBytes, currentStream.cachedStateBytes);
      await _timedAsync(
        languageTimer,
        () => currentInput.addProjected(complete),
      );
      final padded = _timed(encoderTimer, currentStream.finish);
      await _timedAsync(languageTimer, () => currentInput.addProjected(padded));
      await _timedAsync(languageTimer, currentInput.finish);
      audioTokens = currentInput.audioTokens;
      promptTokens = currentInput.promptTokens;
      greedy = currentInput.greedyToken;
      stops = currentInput.stopTokenIds;
    }
    inputTimer.stop();
    inputProcessingSeconds = inputTimer.elapsedMicroseconds / 1e6;
    prefillSeconds = languageTimer.elapsedMicroseconds / 1e6;
    languageTimer.start();
    for (var i = 0; i < 128; i++) {
      final next = greedy(model.logits);
      if (stops.contains(next)) {
        stopReason = 'eos';
        break;
      }
      ids.add(next);
      if (i + 1 < 128) await model.step(next);
    }
    languageTimer.stop();
    text = tokenizer.decode(ids);
  } finally {
    input?.close();
    stream?.close();
    encoder.close();
    await model?.close();
  }
  stdout.writeln(
    const JsonEncoder.withIndent('  ').convert({
      'model': omni ? 'LFM3-Omni-2B-Q4_0' : 'LFM3-Audio-2B-Q4_0',
      'source_revision': omni
          ? Lfm3OmniQ4Source.revision
          : Lfm3AudioQ4Source.revision,
      'backend': gpu ? 'gpu' : 'cpu',
      'input_mode': chunkMs == null ? 'batch' : 'streaming',
      'audio_seconds': pcm.length / 16000,
      'audio_tokens': audioTokens,
      'prompt_tokens': promptTokens,
      'input_chunks': inputChunks,
      if (chunkMs != null) ...{
        'chunk_ms': chunkMs,
        'audio_tokens_prefilled_before_finish': beforeFinish,
        'peak_encoder_cache_bytes': peakCacheBytes,
      },
      'generated_ids': ids,
      'text': text,
      'stop_reason': stopReason,
      'frontend_seconds': frontendTimer.elapsedMicroseconds / 1e6,
      'encoder_seconds': encoderTimer.elapsedMicroseconds / 1e6,
      'language_prefill_seconds': prefillSeconds,
      'input_processing_seconds': inputProcessingSeconds,
      'language_prefill_decode_seconds':
          languageTimer.elapsedMicroseconds / 1e6,
    }),
  );
}
