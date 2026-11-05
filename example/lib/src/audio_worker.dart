import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:dianoia/lab.dart';

import 'audio_backend.dart';
import 'model_worker_gate.dart';
import 'photo_conversation.dart';
import 'speech_model.dart';

/// Request/reply transport. One worker owns the audio and language models.
final class AudioIsolateBackend implements AudioBackend {
  AudioIsolateBackend({
    this.model = SpeechModel.audio,
    this.useGpu = true,
    this.mirror = const String.fromEnvironment('DIANOIA_MODEL_MIRROR'),
  });
  final SpeechModel model;
  final bool useGpu;
  final String mirror;
  void Function(InstallProgress)? _progress;
  void Function(String)? _replyText;
  void Function(String)? _liveText;
  final _pending = <int, Completer<Object?>>{};
  Future<void>? _starting, _closing;
  final _exited = Completer<void>();
  SendPort? _commands;
  var _next = 0, _closed = false;
  var _controlVersion = 0;

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
      if (reply['liveUpdate'] == true) {
        _liveText?.call(reply['text'] as String);
        return;
      }
      if (reply['textUpdate'] == true) {
        _replyText?.call(reply['text'] as String);
        return;
      }
      if (reply['progress'] == true) {
        _progress?.call(
          InstallProgress(
            InstallStage.values[reply['stage'] as int],
            bytes: reply['bytes'] as int,
            totalBytes: reply['total'] as int,
          ),
        );
        return;
      }
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
        (events.sendPort, model, useGpu, mirror),
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
  Future<String?> installed(String storeDirectory) async {
    await _start();
    return await _request('installed', storeDirectory) as String?;
  }

  @override
  Future<String> install(
    String storeDirectory,
    void Function(InstallProgress) onProgress,
  ) async {
    final version = _controlVersion;
    await _start();
    if (version != _controlVersion) {
      throw StateError('Installation was cancelled');
    }
    if (_progress != null) throw StateError('Installation is already running');
    _progress = onProgress;
    try {
      return await _request('install', storeDirectory) as String;
    } finally {
      _progress = null;
    }
  }

  @override
  Future<String> load(String directory) async {
    final version = _controlVersion;
    await _start();
    if (version != _controlVersion) {
      throw StateError('Model loading was cancelled');
    }
    return (await _request('load', directory)) as String;
  }

  @override
  Future<void> begin(
    String? language, {
    bool conversation = false,
    bool live = false,
    bool question = false,
    void Function(String)? onLiveText,
  }) async {
    if ([conversation, live, question].where((v) => v).length > 1) {
      throw ArgumentError('Choose one audio input mode');
    }
    _liveText = live ? onLiveText : null;
    try {
      await _request('begin', {
        'language': language,
        'conversation': conversation,
        'live': live,
        'question': question,
      });
    } catch (_) {
      _liveText = null;
      rethrow;
    }
  }

  @override
  Future<AudioVoiceReply> finishConversation(
    void Function(String) onText,
  ) async {
    if (_replyText != null) {
      throw StateError('Conversation is already generating');
    }
    _replyText = onText;
    try {
      return await _request('finishConversation') as AudioVoiceReply;
    } finally {
      _replyText = null;
    }
  }

  @override
  Future<AudioVoiceReply> askAboutPhoto(
    Uint8List? image, {
    String? text,
    required bool spoken,
    required bool speak,
    int previousTurns = 0,
    required void Function(String) onText,
  }) async {
    if (_replyText != null) throw StateError('A reply is already generating');
    _replyText = onText;
    try {
      return await _request('ask', {
            'image': image,
            'text': text,
            'spoken': spoken,
            'speak': speak,
            'previousTurns': previousTurns,
          })
          as AudioVoiceReply;
    } finally {
      _replyText = null;
    }
  }

  @override
  Future<void> acceptConversationReply() async {
    await _request('acceptReply');
  }

  @override
  Future<void> newConversation() async {
    await _request('newConversation');
  }

  @override
  Future<void> addPcm(Float32List samples) async {
    await _request('pcm', samples);
  }

  @override
  Future<String> finish() async {
    try {
      return await _request('finish') as String;
    } finally {
      _liveText = null;
    }
  }

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
    _controlVersion++;
    _liveText = null;
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

/// Text budget for a 30-second clip, reserved in the context before audio.
/// Reaching it fails instead of returning a silently truncated transcript.
const _maxTranscriptTokens = 256;

/// Image budget for questions about photos, as for the VL example chat.
const _maxPhotoTokens = 256;

Future<void> _runAudioWorker((SendPort, SpeechModel, bool, String) args) async {
  final (out, kind, useGpu, mirror) = args;
  final spec = mirror.isEmpty ? kind.spec : kind.spec.fromMirror(mirror);
  final mmprojName = switch (kind) {
    SpeechModel.audio => Lfm3AudioQ4Source.audio.name,
    SpeechModel.omni => Lfm3OmniQ4Source.mmproj.name,
  };
  Lfm3VisionEncoder? vision;
  PhotoConversation? photo;
  // Projected rows of a spoken question, kept for the following ask.
  final questionRows = <Float32List>[];
  var recordingQuestion = false;
  final commands = ReceivePort();
  out.send(commands.sendPort);
  Lfm3Model? model;
  Lfm3AudioEncoder? encoder;
  Lfm3Tokenizer? tokenizer;
  Lfm3AudioGenerator? generator;
  AudioSynthesizer? synthesizer;
  AudioConversationSession? conversation;
  Lfm3LiveTranscriber? live;
  var recordingConversation = false;
  String? directory;
  StreamSubscription<InstallProgress>? installation;
  Completer<void>? installDone;
  Future<void>? stoppingInstall;
  Future<void> cancelInstallation() async {
    final subscription = installation;
    installation = null;
    try {
      await subscription?.cancel();
    } finally {
      final done = installDone;
      if (done != null && !done.isCompleted) {
        done.completeError(StateError('Installation was cancelled'));
      }
    }
  }

  AudioTranscriptionPrefill? input;
  Lfm3AudioEncoderStream? stream;
  Lfm3AudioFeatureStream? features;
  var generation = 0, samples = 0, closing = false;
  Future<void> chain = Future<void>.value();
  void clear() {
    live?.close();
    live = null;
    input?.close();
    stream?.close();
    input = null;
    stream = null;
    features = null;
    samples = 0;
    recordingConversation = false;
    recordingQuestion = false;
  }

  Future<void> addProjected(Float32List rows) async {
    if (recordingQuestion) {
      if (rows.isNotEmpty) questionRows.add(rows);
    } else if (recordingConversation) {
      await conversation!.addProjected(rows);
    } else {
      await input!.addProjected(rows);
    }
  }

  void check(int version) {
    if (closing || version != generation) {
      throw StateError('Audio input was cancelled');
    }
  }

  commands.listen((message) {
    final request = message as Map;
    final op = request['op'] as String;
    // Acknowledgement is synchronous and ordered before a subsequent cancel.
    // A late reply that the UI discarded never becomes hidden conversation history.
    if (op == 'acceptReply') {
      try {
        conversation!.acceptReply();
        out.send({'id': request['id'], 'value': null});
      } catch (e) {
        out.send({'id': request['id'], 'error': '$e'});
      }
      return;
    }
    if (op == 'cancel' || op == 'close') {
      generation++;
      synthesizer?.cancel();
      conversation?.cancel();
      live?.close();
      stoppingInstall ??= cancelInstallation().whenComplete(
        () => stoppingInstall = null,
      );
      stoppingInstall!.ignore();
    }
    if (op == 'close') closing = true;
    final version = generation;
    chain = chain.then((_) async {
      try {
        Object? result;
        if (op == 'cancel') {
          await stoppingInstall;
          clear();
        } else if (op == 'close') {
          await stoppingInstall;
          clear();
          conversation?.close();
          generator?.close();
          encoder?.close();
          vision?.close();
          await model?.close();
        } else {
          check(version);
          switch (op) {
            case 'newConversation':
              conversation?.clear();
            case 'installed':
              final store = ModelStore(request['value'] as String);
              result = store.isInstalled(spec) ? store.bundlePath(spec) : null;
            case 'install':
              if (model != null) throw StateError('Model is already loaded');
              final store = ModelStore(request['value'] as String);
              final done = installDone = Completer<void>();
              done.future.ignore();
              try {
                installation = store
                    .install(spec)
                    .listen(
                      (p) {
                        out.send({
                          'progress': true,
                          'stage': p.stage.index,
                          'bytes': p.bytes,
                          'total': p.totalBytes,
                        });
                      },
                      onError: (Object e, StackTrace stack) {
                        if (!done.isCompleted) done.completeError(e, stack);
                      },
                      onDone: () {
                        if (!done.isCompleted) done.complete();
                      },
                      cancelOnError: true,
                    );
                await done.future;
                check(version);
                result = store.bundlePath(spec);
              } finally {
                await installation?.cancel();
                installation = null;
                installDone = null;
              }
            case 'load':
              if (model != null) throw StateError('Model is already loaded');
              final dir = request['value'] as String;
              if (!NativeKernels.isAvailable) {
                throw UnsupportedError(
                  'Native inference is unavailable on this device',
                );
              }
              NativeKernels.load(null);
              await switch (kind) {
                SpeechModel.audio => _verifyBundle(dir, () => check(version)),
                SpeechModel.omni => _verifyOmniBundle(
                  dir,
                  () => check(version),
                ),
              };
              encoder = Lfm3AudioEncoder.load('$dir/$mmprojName');
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
              generator ??= Lfm3AudioGenerator.load('$directory/$mmprojName');
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
              conversation?.cancel();
              final options = request['value'] as Map;
              recordingConversation = options['conversation'] as bool;
              recordingQuestion = options['question'] == true;
              final liveInput = options['live'] == true;
              if (recordingQuestion) {
                if (!kind.supportsPhotoQuestions) {
                  throw UnsupportedError('Photo questions need LFM3-Omni');
                }
                questionRows.clear();
              }
              if (liveInput && recordingConversation) {
                throw ArgumentError('Choose one audio input mode');
              }
              if (liveInput && !kind.supportsLiveTranscription) {
                throw UnsupportedError('LiveASR needs LFM3-Audio');
              }
              if (recordingQuestion) {
                // Rows are kept until the question is asked.
              } else if (liveInput) {
                if (options['language'] != null &&
                    options['language'] != 'English') {
                  throw ArgumentError('LiveASR supports English only');
                }
                live = await Lfm3LiveTranscriber.start(
                  model!,
                  tokenizer!,
                  encoder!,
                  onText: (text) {
                    out.send({'liveUpdate': true, 'text': text});
                  },
                );
              } else if (recordingConversation) {
                generator ??= Lfm3AudioGenerator.load('$directory/$mmprojName');
                conversation ??= AudioConversationSession(
                  model!,
                  tokenizer!,
                  generator!,
                );
                await conversation!.begin();
              } else {
                input = await AudioTranscriptionPrefill.start(
                  model: model!,
                  tokenizer: tokenizer!,
                  hiddenSize: encoder!.config.projectionSize,
                  language: options['language'] as String?,
                  reservedTokens: _maxTranscriptTokens,
                );
              }
              check(version);
              if (!liveInput) {
                stream = encoder!.startStream();
                features = Lfm3AudioFrontend().startStream();
              }
            case 'pcm':
              if (live == null &&
                  ((!recordingConversation &&
                          !recordingQuestion &&
                          input == null) ||
                      features == null ||
                      stream == null)) {
                throw StateError('Recording is not active');
              }
              final pcm = request['value'] as Float32List;
              if (live == null && samples + pcm.length > 30 * 16000) {
                throw ArgumentError('Recording exceeds 30 seconds');
              }
              if (live != null) {
                await live!.addPcm(pcm);
              } else {
                final rows = stream!.add(
                  features!.add(pcm, samplingRate: 16000),
                );
                await addProjected(rows);
              }
              check(version);
              samples += pcm.length;
            case 'finishConversation':
              if (!recordingConversation ||
                  stream == null ||
                  features == null ||
                  samples == 0) {
                throw StateError('No voice message recorded');
              }
              await addProjected(stream!.add(features!.finish()));
              check(version);
              await addProjected(stream!.finish());
              check(version);
              final reply = await conversation!.finish(
                deferCommit: true,
                onText: (text) {
                  out.send({'textUpdate': true, 'text': text});
                },
              );
              check(version);
              result = AudioVoiceReply(
                reply.text,
                encodeSpeechWav(reply.samples, sampleRate: reply.sampleRate),
                conversation!.turns + 1,
              );
              clear();
            case 'ask':
              if (model == null || tokenizer == null || directory == null) {
                throw StateError('Model is not loaded');
              }
              if (!kind.supportsPhotoQuestions) {
                throw UnsupportedError('Photo questions need LFM3-Omni');
              }
              final options = request['value'] as Map;
              final spoken = options['spoken'] as bool;
              final text = (options['text'] as String?)?.trim();
              // A new photo or a restart drops earlier questions; a follow-up
              // keeps exactly the answers the caller received.
              final previous = options['previousTurns'] as int;
              if (previous == 0) photo = null;
              if (photo == null && options['image'] == null) {
                throw StateError('Choose a photo first');
              }
              if (previous > 0) {
                (photo ?? (throw StateError('The photo conversation ended')))
                    .keep(previous);
              }
              if (spoken) {
                if (!recordingQuestion ||
                    stream == null ||
                    features == null ||
                    samples == 0) {
                  throw StateError('No spoken question recorded');
                }
                await addProjected(stream!.add(features!.finish()));
                await addProjected(stream!.finish());
                check(version);
              } else if (text == null || text.isEmpty) {
                throw ArgumentError('Ask a question by voice or text');
              }
              final config = loadOmniVisionConfig(
                '$directory/${Lfm3OmniQ4Source.config.name}',
                tokenizer!,
              );
              final hidden = model!.config.hiddenSize;
              if (photo == null) {
                vision ??= loadVisionEncoderGguf(
                  '$directory/$mmprojName',
                  config,
                );
                final rgb = decodeImageRgb(options['image'] as Uint8List);
                final image = Lfm3ImageProcessor.load(directory!)
                    .withMaxImageTokens(_maxPhotoTokens)
                    .prepareRgb(
                      rgb.pixels,
                      height: rgb.height,
                      width: rgb.width,
                    );
                final imageRows = await vision!.encodeAsync(
                  image,
                  checkCancelled: () => check(version),
                );
                photo = PhotoConversation(imageRows, image.imageTokens, hidden);
              }
              final chat = photo!;
              Float32List? audioRows;
              if (spoken) {
                audioRows = Float32List(
                  questionRows.fold(0, (n, r) => n + r.length),
                );
                var at = 0;
                for (final rows in questionRows) {
                  audioRows.setAll(at, rows);
                  at += rows.length;
                }
              }
              final speak = options['speak'] as bool;
              final mode = speak ? AudioReplyMode.mixed : AudioReplyMode.text;
              final prompt = prepareOmniPrompt(
                chat.messages(speak: speak, text: text, spoken: spoken),
                tokenizer: tokenizer!,
                config: config,
                imageTokenCounts: [chat.imageTokens],
                audioTokenCounts: chat.audioTokens(audioRows),
                mode: mode,
              );
              final maxSlots = speak ? 768 : 512;
              if (prompt.ids.length + maxSlots + 1 > model!.maxSeq) {
                throw StateError(
                  chat.turns.isEmpty
                      ? 'The question is too long for the context'
                      : 'The conversation about this photo is full. '
                            'Start a new conversation.',
                );
              }
              await prompt.prefill(
                model!,
                images: [chat.imageRows],
                audio: chat.audio(audioRows),
              );
              check(version);
              if (speak) {
                generator ??= Lfm3AudioGenerator.load('$directory/$mmprojName');
              }
              final reply = await decodeSpeechReply(
                model: model!,
                tokenizer: tokenizer!,
                generator: speak ? generator : null,
                variant: Lfm3SpeechVariant.omni,
                mode: mode,
                maxSlots: maxSlots,
                check: () => check(version),
                onText: (text) {
                  out.send({'textUpdate': true, 'text': text});
                },
                feed: (id, embedding, {required logits}) => model!.forward(
                  [id],
                  computeLogits: logits,
                  embeddingOverrides: {0: ?embedding},
                ),
              );
              check(version);
              if (reply.text.isNotEmpty) {
                chat.add(text: text, audio: audioRows, answer: reply.text);
              }
              result = AudioVoiceReply(
                reply.text,
                reply.samples.isEmpty
                    ? Uint8List(0)
                    : encodeSpeechWav(
                        reply.samples,
                        sampleRate: reply.sampleRate,
                      ),
                chat.turns.length,
              );
              clear();
              questionRows.clear();
            case 'finish':
              if (live != null) {
                result = await live!.finish();
                check(version);
                clear();
                break;
              }
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
              var stopped = false;
              for (var i = 0; i < _maxTranscriptTokens; i++) {
                check(version);
                final token = current.greedyToken(model!.logits);
                if (current.stopTokenIds.contains(token)) {
                  stopped = true;
                  break;
                }
                ids.add(token);
                if (i < _maxTranscriptTokens - 1) await model!.step(token);
              }
              if (!stopped) {
                throw StateError(
                  'Transcript reached the decoding limit. Please try a shorter recording.',
                );
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
        conversation?.cancel();
        clear();
        // A failed load must not retain an encoder when the caller retries.
        if (op == 'load') {
          encoder?.close();
          encoder = null;
          vision?.close();
          vision = null;
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
  final installed = File('$dir/audio_bundle.json').existsSync();
  if (installed) {
    await validateAudioBundle(
      dir,
      AudioGgufSpec.lfm3AudioQ4.files,
      Lfm3AudioQ4Source.audio.name,
      check: check,
    );
  }
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
  // The installed manifest pins these payloads to the source hashes and
  // validateAudioBundle has just hashed them; only research folders need it.
  for (final (file, spec) in [
    if (!installed) ...[
      (File('$dir/${Lfm3AudioQ4Source.audio.name}'), Lfm3AudioQ4Source.audio),
      (File('$dir/language/tokenizer.json'), Lfm3AudioQ4Source.tokenizer),
      (File('$dir/audio_config.json'), Lfm3AudioQ4Source.config),
    ],
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

/// Omni bundle provenance: an installed bundle is fully hashed against its
/// manifest; a prepared research folder must carry the pinned receipt and
/// the original mmproj and metadata.
Future<void> _verifyOmniBundle(String dir, void Function() check) async {
  final installed = File('$dir/omni_bundle.json').existsSync();
  if (installed) {
    await validateOmniBundle(
      dir,
      OmniGgufSpec.lfm3OmniQ4.files,
      Lfm3OmniQ4Source.mmproj.name,
      check: check,
    );
  }
  final receipt =
      jsonDecode(await File('$dir/language/omni_source.json').readAsString())
          as Map;
  if (receipt['repository'] != Lfm3OmniQ4Source.repository ||
      receipt['revision'] != Lfm3OmniQ4Source.revision ||
      receipt['metadata_revision'] != Lfm3OmniQ4Source.metadataRevision) {
    throw const FormatException(
      'Choose a prepared official LFM3-Omni Q4_0 model',
    );
  }
  for (final f in [
    ...Lfm3OmniQ4Source.files,
    ...Lfm3OmniQ4Source.metadataFiles,
  ]) {
    final entry = (receipt['files'] as Map)[f.name] as Map?;
    if (entry?['size'] != f.size || entry?['sha256'] != f.sha256) {
      throw const FormatException('Omni source receipt differs');
    }
  }
  for (final (file, spec) in [
    if (!installed) ...[
      (File('$dir/${Lfm3OmniQ4Source.mmproj.name}'), Lfm3OmniQ4Source.mmproj),
      (File('$dir/language/tokenizer.json'), Lfm3OmniQ4Source.tokenizer),
      (File('$dir/omni_config.json'), Lfm3OmniQ4Source.config),
      (File('$dir/processor_config.json'), Lfm3OmniQ4Source.processorConfig),
    ],
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
  final cache = Q8Cache.open(
    '$dir/language/lfm3-q4.cache',
    Q8Cache.keyFor('$dir/language', weights: 'q4'),
  );
  if (cache == null) {
    throw const FormatException(
      'Prepared Omni language cache is missing or stale',
    );
  }
  cache.close();
  check();
}
