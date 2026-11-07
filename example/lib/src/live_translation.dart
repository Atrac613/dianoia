import 'dart:async';
import 'dart:collection';

import 'package:flutter/foundation.dart';

/// Translates text with a language model. The UI never runs the model; the
/// implementation owns a worker (see translation_worker.dart).
abstract interface class TranslationBackend {
  /// Loads the model in [languageDirectory] once; later calls return the
  /// same result. Returns where it runs, such as `CPU`.
  Future<String> load(String languageDirectory);

  /// Translates English [text] into [target], streaming the text so far to
  /// [onText].
  Future<String> translate(
    String text,
    String target, {
    void Function(String)? onText,
  });

  /// Stops the translation in progress, which then fails.
  Future<void> cancel();
  Future<void> close();
}

/// Languages the live translation offers, as named in the prompt.
const translationTargets = [
  'Japanese',
  'Chinese',
  'Korean',
  'French',
  'Spanish',
  'German',
];

/// One finished sentence of the transcript and its translation, which is
/// null until it arrives and partial while it streams.
final class TranslatedSegment {
  TranslatedSegment(this.source);
  final String source;
  String translation = '';
  bool done = false, failed = false;
}

/// Cuts a growing English transcript into sentences that will not change.
///
/// LiveASR appends to its transcript and only the end of the last word can
/// still grow, so a sentence is final once its end mark is followed by
/// another word, or by the line break LiveASR adds when it restarts a
/// language context. A long stretch without punctuation is cut at a word
/// boundary so the translation does not stall.
final class SentenceSplitter {
  SentenceSplitter({this.maxLength = 200});

  /// Characters after which an unpunctuated stretch is cut at a space.
  final int maxLength;

  static final _end = RegExp(r'''[.!?…。！？]["')\]]*(?=\s+\S)|\n''');

  /// The sentences of [text] after offset [from] that are final, and the
  /// offset after them. With [flush] the rest of the text is final too.
  (List<String>, int) take(String text, int from, {bool flush = false}) {
    final sentences = <String>[];
    var start = from.clamp(0, text.length);
    void add(int end) {
      final sentence = text.substring(start, end).trim();
      if (sentence.isNotEmpty) sentences.add(sentence);
      start = end;
    }

    for (final match in _end.allMatches(text, start)) {
      add(match.end);
    }
    while (text.length - start > maxLength) {
      // Only a space followed by more text marks a whole word.
      final cut = text.lastIndexOf(' ', start + maxLength);
      if (cut <= start) break;
      add(cut);
    }
    if (flush) add(text.length);
    return (sentences, start);
  }
}

/// Follows a live transcript and translates each final sentence in order,
/// one at a time, beside the speech recognition. Translations lag behind
/// the speech; finished sentences are never translated twice.
final class LiveTranslator extends ChangeNotifier {
  LiveTranslator(this._backend, {SentenceSplitter? splitter})
    : _splitter = splitter ?? SentenceSplitter();
  final TranslationBackend _backend;
  final SentenceSplitter _splitter;

  String target = translationTargets.first;

  /// Where the translation model runs once [prepare] finished.
  String? backendName;
  String? error;
  final _segments = <TranslatedSegment>[];
  List<TranslatedSegment> get segments => List.unmodifiable(_segments);
  final _queue = ListQueue<TranslatedSegment>();
  Future<void>? _translating, _loading;
  String? _loadedFrom;
  int _consumed = 0, _epoch = 0;
  bool _disposed = false;

  bool get busy => _translating != null;

  /// The part of [transcript] that is not yet a finished sentence.
  String unfinished(String transcript) => _consumed >= transcript.length
      ? ''
      : transcript.substring(_consumed).trim();

  /// Sentences of the transcript still waiting for a translation.
  int get pending =>
      _segments.where((segment) => !segment.done && !segment.failed).length;

  /// Loads the translation model from the speech model's language weights.
  Future<void> prepare(String languageDirectory) {
    if (_loadedFrom == languageDirectory && _loading != null) return _loading!;
    _loadedFrom = languageDirectory;
    return _loading = _load(languageDirectory);
  }

  Future<void> _load(String directory) async {
    try {
      backendName = await _backend.load(directory);
      error = null;
    } catch (e) {
      // A later attempt may succeed, for example after a cancelled load.
      _loading = null;
      _loadedFrom = null;
      error = 'The translator could not be loaded: $e';
      rethrow;
    } finally {
      _notify();
    }
  }

  void setTarget(String value) {
    if (busy || pending > 0 || target == value) return;
    target = value;
    _notify();
  }

  /// Forgets the earlier transcript and its translations.
  Future<void> reset() async {
    _epoch++;
    _segments.clear();
    _queue.clear();
    _consumed = 0;
    error = null;
    final running = _translating;
    // The stale loop sees the new epoch and stops; a new one may start now.
    _translating = null;
    _notify();
    if (running != null) await _backend.cancel();
  }

  /// Takes the new final sentences of [transcript]; with [flush] (after the
  /// recording stopped) its last sentence as well.
  void update(String transcript, {bool flush = false}) {
    final (sentences, consumed) = _splitter.take(
      transcript,
      _consumed,
      flush: flush,
    );
    _consumed = consumed;
    if (sentences.isEmpty) return;
    for (final sentence in sentences) {
      final segment = TranslatedSegment(sentence);
      _segments.add(segment);
      _queue.add(segment);
    }
    _translating ??= _drain(_epoch);
    _notify();
  }

  Future<void> _drain(int epoch) async {
    try {
      while (_queue.isNotEmpty && epoch == _epoch && !_disposed) {
        final segment = _queue.removeFirst();
        try {
          final text = await _backend.translate(
            segment.source,
            target,
            onText: (text) {
              if (epoch != _epoch) return;
              segment.translation = text;
              _notify();
            },
          );
          if (epoch != _epoch) return;
          segment
            ..translation = text.trim()
            ..done = true;
        } catch (e) {
          if (epoch != _epoch) return;
          segment.failed = true;
          error = 'Translation failed: $e';
        }
        _notify();
      }
    } finally {
      if (epoch == _epoch) _translating = null;
      _notify();
    }
  }

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  /// Stops translating and closes the model's worker.
  Future<void> close() async {
    _epoch++;
    _queue.clear();
    await _backend.close();
  }

  @override
  void dispose() {
    _disposed = true;
    unawaited(close());
    super.dispose();
  }
}
