import 'dart:async';
import 'dart:isolate';

import 'package:dianoia/dianoia.dart';

import 'live_translation.dart';

/// The instruction the translation model receives, in the system prompt
/// and before each sentence; a bare system prompt let the model answer the
/// sentence instead of translating it.
String translationSystemPrompt(String target) =>
    'You are a translator. Translate English into $target.';
String translationRequest(String text, String target) =>
    'Translate the following English into $target. '
    'Output only the translation.\n\n$text';

/// The model sometimes closes a reasoning block it was never asked for and
/// repeats the answer after it; keep only what follows.
String cleanTranslation(String text) {
  final end = text.lastIndexOf('</think>');
  final kept = end < 0 ? text : text.substring(end + '</think>'.length);
  return kept.replaceFirst('<think>', '').trim();
}

/// Enough room for one transcript sentence, its translation and the prompt.
const _contextLength = 512;

/// A second, text-only copy of the speech model's language backbone in its
/// own isolate, on the CPU kernels.
///
/// LiveASR keeps that backbone busy, and its state, so translation cannot
/// borrow it between audio blocks. The GPU backend is process-wide (one
/// command buffer at a time), so the copy runs on the CPU, beside the GPU
/// recognition instead of in turn with it. The weights are memory-mapped
/// files, so the copy shares their pages; its own memory is its context.
final class TranslationIsolateBackend implements TranslationBackend {
  TranslationIsolateBackend({this.threads = 4});
  final int threads;
  final _pending = <int, Completer<Object?>>{};
  final _onText = <int, void Function(String)>{};
  Future<void>? _starting, _closing;
  Future<String>? _loading;
  final _exited = Completer<void>();
  SendPort? _commands;
  var _next = 0, _closed = false;

  Future<void> _start() => _starting ??= _spawn();
  Future<void> _spawn() async {
    final events = ReceivePort(), exit = ReceivePort();
    final ready = Completer<void>();
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
      if (reply['textUpdate'] == true) {
        _onText[reply['id']]?.call(reply['text'] as String);
        return;
      }
      _onText.remove(reply['id']);
      final c = _pending.remove(reply['id']);
      if (c == null) return;
      if (reply['error'] case final String message) {
        c.completeError(StateError(message));
      } else {
        c.complete(reply['value']);
      }
    });
    exit.listen((_) {
      _commands = null;
      fail(StateError('Translation worker exited'));
      events.close();
      exit.close();
      if (!_exited.isCompleted) _exited.complete();
    });
    try {
      await Isolate.spawn(
        _runTranslationWorker,
        (events.sendPort, threads),
        onExit: exit.sendPort,
        errorsAreFatal: true,
      );
    } catch (_) {
      events.close();
      exit.close();
      if (!_exited.isCompleted) _exited.complete();
      rethrow;
    }
    await ready.future;
  }

  Future<Object?> _request(String op, [Object? value]) {
    if (_closed && op != 'close') {
      throw StateError('Translation worker is closed');
    }
    final port = _commands;
    if (port == null) throw StateError('Translation worker is unavailable');
    final id = _next++;
    final c = _pending[id] = Completer<Object?>();
    port.send({'id': id, 'op': op, 'value': value});
    return c.future;
  }

  @override
  Future<String> load(String languageDirectory) => _loading ??= () async {
    try {
      await _start();
      return await _request('load', languageDirectory) as String;
    } catch (_) {
      _loading = null;
      rethrow;
    }
  }();

  @override
  Future<String> translate(
    String text,
    String target, {
    void Function(String)? onText,
  }) async {
    final port = _commands;
    if (_closed || port == null) throw StateError('Translator is not loaded');
    final id = _next++;
    final c = _pending[id] = Completer<Object?>();
    if (onText != null) _onText[id] = onText;
    port.send({
      'id': id,
      'op': 'translate',
      'value': {'text': text, 'target': target},
    });
    return await c.future as String;
  }

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
      /* The isolate exits either way. */
    }
    await _exited.future;
  }
}

Future<void> _runTranslationWorker((SendPort, int) args) async {
  final (out, threads) = args;
  final commands = ReceivePort();
  out.send(commands.sendPort);
  LocalModel? model;
  String? loadedFrom;
  ChatSession? chat;
  String? chatTarget;
  // Cancel and close act at once; the rest run one at a time in order.
  var generation = 0;
  Future<void> chain = Future<void>.value();
  commands.listen((message) {
    final request = message as Map;
    final id = request['id'];
    final op = request['op'] as String;
    if (op == 'cancel' || op == 'close') generation++;
    final version = generation;
    chain = chain.then((_) async {
      try {
        Object? result;
        switch (op) {
          case 'cancel':
            break;
          case 'close':
            await model?.close();
            model = null;
          case 'load':
            final dir = request['value'] as String;
            if (model != null && loadedFrom != dir) {
              throw StateError('Another translation model is loaded');
            }
            model ??= await LocalModel.load(
              dir,
              backend: InferenceBackend.cpu,
              contextLength: _contextLength,
              threads: threads,
            );
            loadedFrom = dir;
            result = 'CPU';
          case 'translate':
            final current = model ?? (throw StateError('Model is not loaded'));
            final options = request['value'] as Map;
            final text = options['text'] as String;
            final target = options['target'] as String;
            if (chat == null || chatTarget != target) {
              chat = current.chat(
                system: translationSystemPrompt(target),
                maxNewTokens: 160,
              );
              chatTarget = target;
            }
            // Each sentence alone: the system prompt's state is reused and
            // the context never fills up.
            final session = chat!..clear();
            if (version != generation) {
              throw StateError('Translation was cancelled');
            }
            var cancelled = false;
            try {
              final turn = await session.send(
                {'role': 'user', 'content': translationRequest(text, target)},
                onText: (partial) async {
                  out.send({
                    'id': id,
                    'textUpdate': true,
                    'text': cleanTranslation(partial),
                  });
                  // Lets a cancel message arrive between tokens.
                  await Future<void>.delayed(Duration.zero);
                  cancelled = version != generation;
                  return !cancelled;
                },
              );
              if (cancelled) {
                throw StateError('Translation was cancelled');
              }
              result = cleanTranslation(turn.text);
            } on ContextLengthExceeded {
              throw StateError('The sentence is too long to translate');
            }
          default:
            throw ArgumentError('Unknown translation command $op');
        }
        out.send({'id': id, 'value': result});
      } catch (e) {
        out.send({'id': id, 'error': '$e'});
      }
      if (op == 'close') {
        commands.close();
        Isolate.exit();
      }
    });
  });
}
