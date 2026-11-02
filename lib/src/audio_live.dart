/// English LiveASR using the pinned model's chunked assistant-turn protocol.
library;

import 'dart:typed_data';

import 'audio_encoder.dart';
import 'audio_features.dart';
import 'model.dart';
import 'tokenizer.dart';

const audioLivePrompt =
    '<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n<|im_start|>assistant\n';

final class AudioLiveBlock {
  AudioLiveBlock(List<int> ids, this.audioTokens, this.closed, this.isFinal)
    : ids = List.unmodifiable(ids);
  final List<int> ids;
  final int audioTokens;
  final bool closed, isFinal;
}

/// Token suppression differs on final/nonfinal chunks; placeholders are always
/// forbidden. This policy follows the pinned transcribe_live implementation.
final class AudioLiveTokenPolicy {
  AudioLiveTokenPolicy(Map<String, int> markers, this.vocabSize)
    : markers = Map.unmodifiable(markers) {
    for (final name in names) {
      final id = markers[name];
      if (id == null || id < 0 || id >= vocabSize) {
        throw FormatException('LiveASR tokenizer lacks $name');
      }
    }
    if (names.map((name) => markers[name]).toSet().length != names.length) {
      throw const FormatException('LiveASR markers must have distinct IDs');
    }
  }
  factory AudioLiveTokenPolicy.fromTokenizer(
    Lfm3Tokenizer tokenizer,
    int vocabSize,
  ) => AudioLiveTokenPolicy({
    for (final n in names)
      if (tokenizer.addedTokenId(n) != null) n: tokenizer.addedTokenId(n)!,
  }, vocabSize);
  static const names = [
    '<|audio|>',
    '<|audio_out|>',
    '<|audio_start|>',
    '<|mixed_start|>',
    '<|text_end|>',
    '<|mixed_end|>',
    '<|im_end|>',
    '<|endoftext|>',
    '<|im_start|>',
  ];
  final Map<String, int> markers;
  final int vocabSize;
  int get audio => markers['<|audio|>']!;
  int get textEnd => markers['<|text_end|>']!;
  bool closesBlock(int token, {required bool isFinal}) =>
      token == textEnd ||
      (isFinal &&
          [
            '<|im_end|>',
            '<|endoftext|>',
            '<|im_start|>',
          ].any((n) => token == markers[n]));
  int pick(Float32List logits, {required bool isFinal}) {
    if (logits.length != vocabSize) {
      throw ArgumentError('LiveASR logits differ from vocabulary');
    }
    final suppressed = {
      audio,
      markers['<|audio_out|>']!,
      if (!isFinal)
        ...[
          '<|audio_start|>',
          '<|mixed_start|>',
          '<|im_end|>',
          '<|mixed_end|>',
          '<|endoftext|>',
        ].map((n) => markers[n]!),
    };
    var token = -1, best = double.negativeInfinity;
    for (var i = 0; i < logits.length; i++) {
      if (!suppressed.contains(i) && logits[i].isFinite && logits[i] > best) {
        token = i;
        best = logits[i];
      }
    }
    if (token < 0) throw StateError('No finite LiveASR text token');
    return token;
  }
}

/// One caller-owned backbone context. The prompt/previous text-end is deferred
/// until a chunk completes audio rows. A sampled closing marker is not fed;
/// a nonfinal block queues text-end for the next chunk, even at the token cap.
final class AudioLiveSession {
  AudioLiveSession._(this.model, this.tokenizer, this._pending, this._policy);
  static Future<AudioLiveSession> start(
    Lfm3Model model,
    Lfm3Tokenizer tokenizer,
  ) async {
    final policy = AudioLiveTokenPolicy.fromTokenizer(
      tokenizer,
      model.config.vocabSize,
    );
    final prompt = tokenizer.encode(audioLivePrompt);
    if (prompt.first != tokenizer.addedTokenId('<|startoftext|>') ||
        prompt.where((id) => id == policy.markers['<|im_start|>']).length !=
            2 ||
        prompt.where((id) => id == policy.markers['<|im_end|>']).length != 1) {
      throw const FormatException('Incompatible LiveASR prompt markers');
    }
    if (prompt.any((id) => id < 0 || id >= model.config.vocabSize) ||
        prompt.length + 48 > model.maxSeq) {
      throw ArgumentError('LiveASR prompt exceeds model context');
    }
    await model.reset();
    return AudioLiveSession._(model, tokenizer, prompt, policy);
  }

  final Lfm3Model model;
  final Lfm3Tokenizer tokenizer;
  final AudioLiveTokenPolicy _policy;
  List<int> _pending;
  final _ids = <int>[];
  bool _busy = false, _closed = false, _cancelled = false, _finished = false;
  int _expectedLength = 0;
  bool get isFinished => _finished;
  int get generatedTokens => _ids.length;
  String get text => tokenizer
      .decode(
        _ids.where((id) {
          final piece = tokenizer.decode([id]);
          return tokenizer.addedTokenId(piece) != id;
        }).toList(),
      )
      .trim();
  void close() {
    _closed = true;
    _cancelled = true;
    _pending = [];
  }

  void _check() {
    if (_closed || _cancelled) {
      throw StateError('Live transcription was cancelled');
    }
  }

  Future<AudioLiveBlock> addProjected(
    Float32List rows, {
    bool isFinal = false,
    int? maxNewTokens,
  }) async {
    _check();
    if (_busy || _finished || model.isClosed) {
      throw StateError('Live transcription is unavailable');
    }
    if (model.length != _expectedLength) {
      close();
      throw StateError('Language state changed outside LiveASR');
    }
    final h = model.config.hiddenSize,
        limit = maxNewTokens ?? (isFinal ? 96 : 48);
    if (rows.length % h != 0 ||
        rows.any((v) => !v.isFinite) ||
        limit < 1 ||
        limit > 96) {
      throw ArgumentError('Invalid LiveASR rows or token budget');
    }
    final count = rows.length ~/ h;
    if (count == 0 && !isFinal) return AudioLiveBlock([], 0, false, false);
    if (model.length + _pending.length + count + limit > model.maxSeq) {
      throw StateError(
        'Live transcription context is full. Start a new recording.',
      );
    }
    final prefix = List<int>.from(_pending), audio = _policy.audio;
    final owned = Float32List.fromList(rows);
    final ids = <int>[];
    var closed = false;
    _busy = true;
    try {
      await model.forward(
        [...prefix, ...List.filled(count, audio)],
        embeddingOverrides: {
          for (var i = 0; i < count; i++)
            prefix.length + i: Float32List.sublistView(
              owned,
              i * h,
              (i + 1) * h,
            ),
        },
      );
      await Future<void>.delayed(Duration.zero);
      _check();
      for (var i = 0; i < limit; i++) {
        final token = _policy.pick(model.logits, isFinal: isFinal);
        if (_policy.closesBlock(token, isFinal: isFinal)) {
          closed = true;
          break;
        }
        ids.add(token);
        await model.step(token);
        await Future<void>.delayed(Duration.zero);
        _check();
      }
      _ids.addAll(ids);
      _expectedLength = model.length;
      _pending = isFinal ? [] : [_policy.textEnd];
      _finished = isFinal;
      return AudioLiveBlock(ids, count, closed, isFinal);
    } catch (_) {
      close();
      rethrow;
    } finally {
      _busy = false;
    }
  }
}

/// Mono 16 kHz PCM to incremental English text. The encoder follows one stream;
/// language decoding happens after every 96 mel frames (about 0.96 seconds).
/// Owns only per-recording states; the caller owns the model and encoder.
final class Lfm3LiveTranscriber {
  Lfm3LiveTranscriber._(
    this._session,
    this._encoder,
    this.onText,
    this.onBlock,
  );
  static Future<Lfm3LiveTranscriber> start(
    Lfm3Model model,
    Lfm3Tokenizer tokenizer,
    Lfm3AudioEncoder encoder, {
    void Function(String)? onText,
    void Function(AudioLiveBlock)? onBlock,
  }) async {
    if (model.config.hiddenSize != encoder.config.projectionSize) {
      throw ArgumentError('Audio projection differs');
    }
    final session = await AudioLiveSession.start(model, tokenizer);
    try {
      return Lfm3LiveTranscriber._(
        session,
        encoder.startStream(),
        onText,
        onBlock,
      );
    } catch (_) {
      session.close();
      rethrow;
    }
  }

  final AudioLiveSession _session;
  final Lfm3AudioEncoderStream _encoder;
  final _frontend = Lfm3AudioFrontend().startStream();
  final void Function(String)? onText;
  final void Function(AudioLiveBlock)? onBlock;
  Float32List _mel = Float32List(0);
  int _frames = 0, _samples = 0;
  bool _closed = false, _busy = false;
  int get pendingFeatureFrames => _frames;
  String get text => _session.text;
  void close() {
    _closed = true;
    _session.close();
    _encoder.close();
    _mel = Float32List(0);
    _frames = 0;
  }

  void _open() {
    if (_closed || _busy) throw StateError('Live transcription is unavailable');
  }

  void _append(AudioFeatures features) {
    final count = _frames + features.frames, joined = Float32List(128 * count);
    for (var m = 0; m < 128; m++) {
      joined.setRange(m * count, m * count + _frames, _mel, m * _frames);
      joined.setRange(
        m * count + _frames,
        (m + 1) * count,
        features.values,
        m * features.frames,
      );
    }
    _frames = count;
    _mel = joined;
  }

  AudioFeatures _take(int count) {
    final values = Float32List(128 * count),
        remaining = Float32List(128 * (_frames - count));
    for (var m = 0; m < 128; m++) {
      values.setRange(m * count, (m + 1) * count, _mel, m * _frames);
      remaining.setRange(
        m * (_frames - count),
        (m + 1) * (_frames - count),
        _mel,
        m * _frames + count,
      );
    }
    _frames -= count;
    _mel = remaining;
    return AudioFeatures.fromLogMel(values, frames: count);
  }

  Future<AudioLiveBlock> _decode(
    Float32List rows, {
    bool isFinal = false,
  }) async {
    final block = await _session.addProjected(rows, isFinal: isFinal);
    if (_closed) throw StateError('Live transcription was cancelled');
    onBlock?.call(block);
    onText?.call(text);
    if (_closed) throw StateError('Live transcription was cancelled');
    return block;
  }

  Future<void> addPcm(Float32List pcm) async {
    _open();
    if (_samples + pcm.length > 30 * 16000 || pcm.any((v) => !v.isFinite)) {
      throw ArgumentError(
        'Expected at most 30 seconds of finite mono 16 kHz PCM',
      );
    }
    _busy = true;
    try {
      _append(_frontend.add(pcm, samplingRate: 16000));
      _samples += pcm.length;
      while (_frames >= 96) {
        await _decode(_encoder.add(_take(96)));
      }
    } catch (_) {
      close();
      rethrow;
    } finally {
      _busy = false;
    }
  }

  Future<String> finish() async {
    _open();
    if (_samples == 0) throw StateError('No audio received');
    _busy = true;
    try {
      _append(_frontend.finish());
      while (_frames > 96) {
        await _decode(_encoder.add(_take(96)));
      }
      final first = _encoder.add(_take(_frames)), tail = _encoder.finish();
      final rows = Float32List(first.length + tail.length)
        ..setAll(0, first)
        ..setAll(first.length, tail);
      final last = await _decode(rows, isFinal: true);
      if (!last.closed) {
        throw StateError(
          'Live transcript reached the final decoding limit. Please try a shorter recording.',
        );
      }
      return text;
    } finally {
      close();
      _busy = false;
    }
  }
}
