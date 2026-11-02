import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:dianoia/lab.dart';

import 'audio_backend.dart';
import 'model_worker_gate.dart';

/// Request/reply transport. One worker owns the audio and language models.
final class AudioIsolateBackend implements AudioBackend {
  AudioIsolateBackend({this.useGpu = true});
  final bool useGpu;
  final _pending = <int, Completer<Object?>>{};
  Future<void>? _starting, _closing;
  final _exited = Completer<void>();
  SendPort? _commands;
  var _next = 0, _closed = false;

  Future<void> _start() => _starting ??= _spawn();
  Future<void> _spawn() async {
    final lease = ModelWorkerLease.reserve();
    final events = ReceivePort(), errors = ReceivePort(), exit = ReceivePort();
    final ready = Completer<void>();
    var spawned = false;
    void fail(Object error) {
      if (!ready.isCompleted) ready.completeError(error);
      for (final c in _pending.values) {
        c.completeError(error);
      }
      _pending.clear();
    }

    events.listen((event) {
      if (event is SendPort) {
        _commands = event;
        ready.complete();
        return;
      }
      final reply = event as Map;
      final c = _pending.remove(reply['id']);
      if (c == null) return;
      if (reply['error'] case final String message) {
        c.completeError(StateError(message));
      } else {
        c.complete(reply['value']);
      }
    });
    errors.listen((error) => fail(StateError('Audio worker failed: $error')));
    exit.listen((_) {
      _commands = null;
      fail(StateError('Audio worker exited'));
      events.close();
      errors.close();
      exit.close();
      lease.release();
      if (!_exited.isCompleted) _exited.complete();
    });
    try {
      await lease.ready;
      if (_closed) throw StateError('Audio worker was closed');
      await Isolate.spawn(
        _runAudioWorker,
        (events.sendPort, useGpu),
        onError: errors.sendPort,
        onExit: exit.sendPort,
        errorsAreFatal: true,
      );
      spawned = true;
      await ready.future;
    } catch (_) {
      // If spawn itself failed, no isolate will deliver the exit event.
      if (!spawned) {
        events.close();
        errors.close();
        exit.close();
        lease.release();
        if (!_exited.isCompleted) _exited.complete();
      }
      rethrow;
    }
  }

  Future<Object?> _request(String op, [Object? value]) async {
    if (_closed && op != 'close') throw StateError('Audio worker is closed');
    final port = _commands;
    if (port == null) throw StateError('Audio worker is unavailable');
    final id = _next++;
    final c = _pending[id] = Completer<Object?>();
    port.send({'id': id, 'op': op, 'value': value});
    return c.future;
  }

  @override
  Future<String> load(String directory) async {
    await _start();
    return (await _request('load', directory)) as String;
  }

  @override
  Future<void> begin(String? language) async {
    await _request('begin', language);
  }

  @override
  Future<void> addPcm(Float32List samples) async {
    await _request('pcm', samples);
  }

  @override
  Future<String> finish() async => (await _request('finish')) as String;
  @override
  Future<Uint8List> synthesize(
    String text,
    String language,
    String voice,
  ) async =>
      (await _request('synthesize', {
            'text': text,
            'language': language,
            'voice': voice,
          }))
          as Uint8List;
  @override
  Future<void> cancel() async {
    if (!_closed && _commands != null) await _request('cancel');
  }

  @override
  Future<void> close() => _closing ??= _close();
  Future<void> _close() async {
    _closed = true;
    if (_starting == null) return;
    try {
      await _starting;
    } catch (_) {
      return;
    }
    try {
      await _request('close');
    } catch (_) {
      /* The exit event releases the lease. */
    }
    await _exited.future;
  }
}

Future<void> _runAudioWorker((SendPort, bool) args) async {
  final (out, useGpu) = args;
  final commands = ReceivePort();
  out.send(commands.sendPort);
  Lfm3Model? model;
  Lfm3AudioEncoder? encoder;
  Lfm3Tokenizer? tokenizer;
  Lfm3AudioGenerator? generator;
  AudioSynthesizer? synthesizer;
  String? directory;
  AudioTranscriptionPrefill? input;
  Lfm3AudioEncoderStream? stream;
  Lfm3AudioFeatureStream? features;
  var generation = 0, samples = 0, closing = false;
  Future<void> chain = Future<void>.value();
  void clear() {
    input?.close();
    stream?.close();
    input = null;
    stream = null;
    features = null;
    samples = 0;
  }

  void check(int version) {
    if (closing || version != generation) {
      throw StateError('Audio input was cancelled');
    }
  }

  commands.listen((message) {
    final request = message as Map;
    final op = request['op'] as String;
    if (op == 'cancel' || op == 'close') {
      generation++;
      synthesizer?.cancel();
    }
    if (op == 'close') closing = true;
    final version = generation;
    chain = chain.then((_) async {
      try {
        Object? result;
        if (op == 'cancel') {
          clear();
        } else if (op == 'close') {
          clear();
          generator?.close();
          encoder?.close();
          await model?.close();
        } else {
          check(version);
          switch (op) {
            case 'load':
              if (model != null) throw StateError('Model is already loaded');
              final dir = request['value'] as String;
              if (!NativeKernels.isAvailable) {
                throw UnsupportedError(
                  'Native inference is unavailable on this device',
                );
              }
              NativeKernels.load(null);
              await _verifyBundle(dir, () => check(version));
              encoder = Lfm3AudioEncoder.load(
                '$dir/${Lfm3AudioQ4Source.audio.name}',
              );
              tokenizer = Lfm3Tokenizer.load('$dir/language');
              Future<Lfm3Model> open(bool gpu) => Lfm3Model.load(
                '$dir/language',
                maxSeq: 2048,
                weights: WeightFormat.q4,
                q8CachePath: '$dir/language/lfm3-q4.cache',
                gpu: gpu,
              );
              var gpu = useGpu;
              try {
                model = await open(gpu);
              } catch (_) {
                if (!gpu) rethrow;
                gpu = false;
                model = await open(false);
              }
              check(version);
              directory = dir;
              result = gpu ? 'GPU' : 'CPU';
            case 'synthesize':
              if (model == null || tokenizer == null || directory == null) {
                throw StateError('Model is not loaded');
              }
              if (input != null) throw StateError('Recording is active');
              final options = request['value'] as Map;
              generator ??= Lfm3AudioGenerator.load(
                '$directory/${Lfm3AudioQ4Source.audio.name}',
              );
              final current = synthesizer = AudioSynthesizer(
                model!,
                tokenizer!,
                generator!,
              );
              final speech = await current.synthesize(
                options['text'] as String,
                language: options['language'] as String,
                voice: options['voice'] as String,
              );
              check(version);
              synthesizer = null;
              if (speech.stopReason != 'eos' || speech.samples.isEmpty) {
                throw StateError(
                  'Speech did not finish. Please try shorter text.',
                );
              }
              result = encodeSpeechWav(
                speech.samples,
                sampleRate: speech.sampleRate,
              );
            case 'begin':
              if (model == null || encoder == null || tokenizer == null) {
                throw StateError('Model is not loaded');
              }
              clear();
              input = await AudioTranscriptionPrefill.start(
                model: model!,
                tokenizer: tokenizer!,
                hiddenSize: encoder!.config.projectionSize,
                language: request['value'] as String?,
              );
              check(version);
              stream = encoder!.startStream();
              features = Lfm3AudioFrontend().startStream();
            case 'pcm':
              if (input == null || features == null || stream == null) {
                throw StateError('Recording is not active');
              }
              final pcm = request['value'] as Float32List;
              if (samples + pcm.length > 30 * 16000) {
                throw ArgumentError('Recording exceeds 30 seconds');
              }
              final rows = stream!.add(features!.add(pcm, samplingRate: 16000));
              await input!.addProjected(rows);
              check(version);
              samples += pcm.length;
            case 'finish':
              final current = input;
              if (current == null ||
                  stream == null ||
                  features == null ||
                  samples == 0) {
                throw StateError('No audio recorded');
              }
              await current.addProjected(stream!.add(features!.finish()));
              check(version);
              await current.addProjected(stream!.finish());
              check(version);
              await current.finish();
              check(version);
              final ids = <int>[];
              for (var i = 0; i < 128; i++) {
                check(version);
                final token = current.greedyToken(model!.logits);
                if (current.stopTokenIds.contains(token)) break;
                ids.add(token);
                if (i < 127) await model!.step(token);
              }
              result = tokenizer!.decode(ids);
              clear();
            default:
              throw ArgumentError('Unknown audio command $op');
          }
        }
        out.send({'id': request['id'], 'value': result});
      } catch (e) {
        synthesizer = null;
        clear();
        // A failed load must not retain an encoder when the caller retries.
        if (op == 'load') {
          encoder?.close();
          encoder = null;
          await model?.close();
          model = null;
        }
        out.send({'id': request['id'], 'error': '$e'});
      }
      if (op == 'close') {
        commands.close();
        Isolate.exit();
      }
    });
  });
}

/// Prepared research bundle provenance, checked before mapping any model.
Future<void> _verifyBundle(String dir, void Function() check) async {
  final receipt =
      jsonDecode(await File('$dir/language/audio_source.json').readAsString())
          as Map;
  if (receipt['repository'] != Lfm3AudioQ4Source.repository ||
      receipt['revision'] != Lfm3AudioQ4Source.revision ||
      receipt['metadata_revision'] != Lfm3AudioQ4Source.metadataRevision) {
    throw const FormatException(
      'Choose a prepared official LFM3-Audio Q4_0 model',
    );
  }
  for (final f in [
    ...Lfm3AudioQ4Source.files,
    ...Lfm3AudioQ4Source.metadataFiles,
  ]) {
    final entry = (receipt['files'] as Map)[f.name] as Map?;
    if (entry?['size'] != f.size || entry?['sha256'] != f.sha256) {
      throw const FormatException('Audio source receipt differs');
    }
  }
  for (final (file, spec) in [
    (File('$dir/${Lfm3AudioQ4Source.audio.name}'), Lfm3AudioQ4Source.audio),
    (File('$dir/language/tokenizer.json'), Lfm3AudioQ4Source.tokenizer),
    (File('$dir/audio_config.json'), Lfm3AudioQ4Source.config),
  ]) {
    check();
    if (await file.length() != spec.size) {
      throw FormatException('${spec.name} has the wrong size');
    }
    final chunks = file.openRead().map((chunk) {
      check();
      return chunk;
    });
    if ((await sha256.bind(chunks).first).toString() != spec.sha256) {
      throw FormatException('${spec.name} has the wrong hash');
    }
  }
  check();
  final full =
      jsonDecode(await File('$dir/audio_config.json').readAsString()) as Map;
  final text = jsonDecode(
    await File('$dir/language/config.json').readAsString(),
  );
  Object? canonical(Object? value) => value is Map
      ? {
          for (final key in (value.keys.cast<String>().toList()..sort()))
            key: canonical(value[key]),
        }
      : value is List
      ? value.map(canonical).toList()
      : value;
  if (jsonEncode(canonical(full['lfm'])) != jsonEncode(canonical(text))) {
    throw const FormatException('Audio language config differs');
  }
  final cache = Q8Cache.open(
    '$dir/language/lfm3-q4.cache',
    Q8Cache.keyFor('$dir/language', weights: 'q4'),
  );
  if (cache == null) {
    throw const FormatException(
      'Prepared audio language cache is missing or stale',
    );
  }
  cache.close();
  check();
}
