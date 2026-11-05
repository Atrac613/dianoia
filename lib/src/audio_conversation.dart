/// Native audio-to-interleaved-text/speech conversation, not an ASR/TTS cascade.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'audio_generation.dart';
import 'audio_gguf.dart';
import 'audio_synthesis.dart';
import 'model.dart';
import 'speech_reply.dart';
import 'tokenizer.dart';

enum AudioReplyMode { mixed, audio, text }

/// The pinned upstream generation schedule: six text slots, twelve audio slots.
/// EOS speech frames return to text mode and still supply feedback embeddings.
final class AudioInterleavingState {
  AudioInterleavingState({
    required this.textEnd,
    required this.turnEnd,
    required this.mixedStart,
    required this.audioStart,
  });
  final int textEnd, turnEnd, mixedStart, audioStart;
  AudioReplyMode mode = AudioReplyMode.mixed;
  int slotsLeft = 6;
  bool speaking = false, textDone = false, finished = false;
  bool get nextIsAudio =>
      mode == AudioReplyMode.audio ||
      (mode == AudioReplyMode.mixed && speaking);
  void _advance(bool wasMixed) {
    if (wasMixed && !textDone && --slotsLeft == 0) {
      speaking = !speaking;
      slotsLeft = speaking ? 12 : 6;
    }
  }

  void acceptText(int token) {
    if (finished || nextIsAudio) throw StateError('Not a text slot');
    final wasMixed = mode == AudioReplyMode.mixed;
    if (token == textEnd) {
      if (wasMixed) {
        textDone = true;
        speaking = true;
      } else {
        finished = true;
      }
    } else if (token == turnEnd) {
      finished = true;
    } else if (!wasMixed) {
      if (token == audioStart) {
        mode = AudioReplyMode.audio;
      } else if (token == mixedStart) {
        mode = AudioReplyMode.mixed;
        slotsLeft = 6;
        speaking = false;
        textDone = false;
      }
    }
    _advance(wasMixed);
  }

  void acceptAudio({required bool eos}) {
    if (finished || !nextIsAudio) throw StateError('Not an audio slot');
    final wasMixed = mode == AudioReplyMode.mixed;
    if (eos) mode = AudioReplyMode.text;
    _advance(wasMixed);
  }
}

const audioConversationSystem = 'Respond with interleaved text and audio.';
const audioConversationHead =
    '<|startoftext|><|im_start|>system\n$audioConversationSystem<|im_end|>\n';
const audioConversationUser = '<|im_start|>user\n';
const audioConversationAssistant =
    '<|im_end|>\n<|im_start|>assistant\n<|mixed_start|>';

/// The user-turn opening for [variant], ending where audio rows begin.
String audioConversationUserOpen(Lfm3SpeechVariant variant) =>
    '$audioConversationUser${variant.audioOpen}';

/// Closes the user's audio and opens the interleaved assistant reply.
String audioConversationAssistantOpen(Lfm3SpeechVariant variant) =>
    '${variant.audioClose}<|im_end|>\n<|im_start|>assistant\n'
    '${variant.assistantOpen}<|mixed_start|>';

final class AudioConversationReply {
  AudioConversationReply(
    this.text,
    this.samples,
    this.sampleRate,
    this.frames,
    this.generatedSlots,
    this.historyTokens,
  );
  final String text;
  final Float32List samples;
  final int sampleRate, frames, generatedSlots, historyTokens;
  double get seconds => samples.length / sampleRate;
}

final class _Slot {
  _Slot(this.id, [this.embedding]);
  final int id;
  final Float32List? embedding;
}

/// A bounded, transactional conversation on caller-owned language/generation
/// models. Input is projected encoder rows. Completed turns retain exact audio
/// input/output embeddings; no waveform or microphone recording is retained.
/// Begin replays committed slots, so other operations may use the backbone
/// between turns. Use the backbone exclusively throughout begin/add/finish.
final class AudioConversationSession {
  AudioConversationSession(this.model, this.tokenizer, this.generator) {
    if (model.config.hiddenSize != generator.hiddenSize) {
      throw ArgumentError('Audio and language dimensions differ');
    }
    for (final marker in [
      '<|startoftext|>',
      '<|im_start|>',
      '<|im_end|>',
      '<|audio|>',
      '<|audio_out|>',
      '<|text_end|>',
      '<|mixed_start|>',
      '<|audio_start|>',
      if (generator.variant == Lfm3SpeechVariant.omni) ...[
        '<|vision_start|>',
        '<|vision_end|>',
        '<|image|>',
        '<|video|>',
      ],
    ]) {
      _id(marker);
    }
  }
  final Lfm3Model model;
  final Lfm3Tokenizer tokenizer;
  final Lfm3AudioGenerator generator;
  final _history = <_Slot>[];
  List<_Slot>? _pending;
  List<_Slot>? _candidate;
  bool _busy = false, _closed = false, _cancelled = false;
  int _rows = 0, _maxSlots = 768, _maxFrames = 375;
  int get turns => _turns;
  int _turns = 0;
  int get historyTokens => _history.length;
  int get retainedEmbeddingBytes =>
      _history.fold(0, (n, s) => n + (s.embedding?.lengthInBytes ?? 0));
  int _id(String marker) {
    final id = tokenizer.addedTokenId(marker);
    if (id == null || id < 0 || id >= model.config.vocabSize) {
      throw FormatException('Conversation tokenizer lacks $marker');
    }
    return id;
  }

  void _open() {
    if (_closed || model.isClosed) {
      throw StateError('Audio conversation is closed');
    }
    if (_busy) throw StateError('Audio conversation is busy');
    if (_pending != null && model.length != _pending!.length) {
      _pending = null;
      throw StateError('Language state changed outside the conversation');
    }
  }

  void _check() {
    if (_closed || _cancelled) throw const AudioSynthesisCancelled();
  }

  void cancel() {
    _cancelled = true;
    _candidate = null;
    if (!_busy) _pending = null;
  }

  void clear() {
    _open();
    if (_pending != null) throw StateError('Cancel the current turn first');
    _history.clear();
    _candidate = null;
    _turns = 0;
  }

  void close() {
    _closed = true;
    cancel();
    _history.clear();
    _turns = 0;
  }

  /// A deferred reply enters history only after its receiver accepts it.
  /// Cancelling an unaccepted reply preserves the previously completed turns.
  void acceptReply() {
    _open();
    final candidate = _candidate;
    if (candidate == null) throw StateError('No reply awaits acceptance');
    _history
      ..clear()
      ..addAll(candidate);
    _candidate = null;
    _turns++;
  }

  List<_Slot> get _assistant =>
      _tokens(audioConversationAssistantOpen(generator.variant));

  List<_Slot> _tokens(String text) {
    final ids = tokenizer.encode(text);
    if (ids.any((id) => id < 0 || id >= model.config.vocabSize)) {
      throw const FormatException('Conversation tokens exceed vocabulary');
    }
    return ids.map(_Slot.new).toList();
  }

  void _capacity(int count) {
    if (count + _maxSlots + 1 > model.maxSeq) {
      throw StateError(
        'Conversation context is full. Start a new conversation.',
      );
    }
  }

  Future<void> _forward(List<_Slot> slots, {bool logits = true}) async {
    for (var at = 0; at < slots.length; at += model.maxChunk) {
      _check();
      final end = math.min(at + model.maxChunk, slots.length);
      await model.forward(
        [for (var i = at; i < end; i++) slots[i].id],
        computeLogits: logits && end == slots.length,
        embeddingOverrides: {
          for (var i = at; i < end; i++)
            if (slots[i].embedding != null) i - at: slots[i].embedding!,
        },
      );
      await Future<void>.delayed(Duration.zero);
      _check();
    }
  }

  Future<void> begin({int maxNewSlots = 768, int maxFrames = 375}) async {
    _open();
    if (_pending != null) {
      throw StateError('A conversation turn is already active');
    }
    if (_candidate != null) {
      throw StateError('Accept or cancel the previous reply first');
    }
    if (maxNewSlots < 1 ||
        maxNewSlots > 1024 ||
        maxFrames < 1 ||
        maxFrames > 375) {
      throw ArgumentError('Invalid conversation output limit');
    }
    _maxSlots = maxNewSlots;
    _maxFrames = maxFrames;
    final head = _tokens(
      '${_history.isEmpty ? audioConversationHead : '\n'}'
      '${audioConversationUserOpen(generator.variant)}',
    );
    _capacity(_history.length + head.length + _assistant.length + 1);
    _busy = true;
    _cancelled = false;
    _rows = 0;
    final pending = _pending = [..._history, ...head];
    try {
      await model.reset();
      await _forward(pending, logits: false);
    } catch (_) {
      _pending = null;
      rethrow;
    } finally {
      _busy = false;
    }
  }

  Future<void> addProjected(Float32List rows) async {
    _open();
    final pending = _pending;
    if (pending == null) throw StateError('No conversation turn is active');
    final h = model.config.hiddenSize;
    if (rows.length % h != 0 || rows.any((v) => !v.isFinite)) {
      throw ArgumentError('Expected finite projected audio rows');
    }
    final count = rows.length ~/ h;
    _capacity(pending.length + count + _assistant.length);
    if (count == 0) return;
    final added = [
      for (var i = 0; i < count; i++)
        _Slot(
          _id('<|audio|>'),
          Float32List.fromList(rows.sublist(i * h, (i + 1) * h)),
        ),
    ];
    _busy = true;
    try {
      await _forward(added, logits: false);
      pending.addAll(added);
      _rows += count;
    } catch (_) {
      _pending = null;
      rethrow;
    } finally {
      _busy = false;
    }
  }

  /// Native interleaved decoding, matching the upstream six/twelve schedule.
  /// Only a complete, nonempty text+speech reply commits history. Cancellation,
  /// callback failure, context limits or output limits discard the current turn.
  Future<AudioConversationReply> finish({
    int seed = 0,
    bool deferCommit = false,
    void Function(String text)? onText,
    void Function(Float32List pcm)? onAudio,
    void Function(int token, List<int>? codes)? onSlot,
  }) async {
    _open();
    final pending = _pending;
    if (pending == null || _rows == 0) throw StateError('No audio received');
    _busy = true;
    try {
      final suffix = _assistant;
      await _forward(suffix);
      pending.addAll(suffix);
      final reply = await decodeSpeechReply(
        model: model,
        tokenizer: tokenizer,
        generator: generator,
        mode: AudioReplyMode.mixed,
        maxSlots: _maxSlots,
        maxFrames: _maxFrames,
        seed: seed,
        check: _check,
        onText: onText,
        onAudio: onAudio,
        onSlot: onSlot,
        feed: (id, embedding, {required logits}) async {
          final slot = _Slot(id, embedding);
          await _forward([slot], logits: logits);
          pending.add(slot);
        },
      );
      if (reply.text.isEmpty || reply.frames == 0) {
        throw StateError(
          'Model returned no complete spoken reply. Please try again.',
        );
      }
      if (pending.last.id != _id('<|im_end|>')) {
        pending.add(_Slot(_id('<|im_end|>')));
      }
      _candidate = pending;
      if (!deferCommit) {
        _history
          ..clear()
          ..addAll(pending);
        _candidate = null;
        _turns++;
      }
      return AudioConversationReply(
        reply.text,
        reply.samples,
        reply.sampleRate,
        reply.frames,
        reply.generatedSlots,
        pending.length,
      );
    } finally {
      _pending = null;
      _busy = false;
    }
  }
}
