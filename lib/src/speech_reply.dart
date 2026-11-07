/// One reply in text, speech or interleaved text and speech, decoded after
/// any LFM3-Audio or LFM3-Omni prompt. Shared by voice conversation and the
/// Omni multimodal prompts.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'audio_conversation.dart';
import 'audio_generation.dart';
import 'audio_gguf.dart';
import 'model.dart';
import 'tokenizer.dart';

final class SpeechReply {
  SpeechReply(
    this.text,
    this.samples,
    this.sampleRate,
    this.frames,
    this.generatedSlots,
  );

  /// Visible text, without special markers. Empty for a speech-only reply.
  final String text;

  /// Mono speech; empty for a text reply.
  final Float32List samples;
  final int sampleRate, frames, generatedSlots;
  double get seconds => samples.length / sampleRate;
}

/// Decodes one reply. The caller has already fed the prompt, ending with the
/// mode's opening marker (`<|mixed_start|>` or `<|audio_start|>`; none for
/// text), so `model.logits` and `model.lastHiddenState` describe the first
/// slot. [feed] forwards one generated slot — a token, or `<|audio_out|>`
/// with its feedback embedding — and computes logits when asked; callers use
/// it to record history. [check] may throw to cancel between slots.
///
/// Follows the upstream schedule: greedy text slots, sampled speech frames,
/// six/twelve alternation in interleaved mode, and a frame with any EOS code
/// normalized to EOS, fed back but never decoded to audio. Text mode
/// suppresses the speech and mixed markers, as upstream text mode does.
/// [variant] defaults to the generator's, or LFM3-Audio without one.
Future<SpeechReply> decodeSpeechReply({
  required Lfm3Model model,
  required Lfm3Tokenizer tokenizer,
  required AudioReplyMode mode,
  required int maxSlots,
  required Future<void> Function(
    int id,
    Float32List? embedding, {
    required bool logits,
  })
  feed,
  Lfm3AudioGenerator? generator,
  Lfm3SpeechVariant? variant,
  int maxFrames = 375,
  int seed = 0,
  void Function()? check,
  void Function(String text)? onText,
  void Function(Float32List pcm)? onAudio,
  void Function(int token, List<int>? codes)? onSlot,
}) async {
  int id(String marker) {
    final value = tokenizer.addedTokenId(marker);
    if (value == null || value < 0 || value >= model.config.vocabSize) {
      throw FormatException('Reply tokenizer lacks $marker');
    }
    return value;
  }

  if (mode != AudioReplyMode.text && generator == null) {
    throw ArgumentError('Speech replies need an audio generator');
  }
  if (generator != null && generator.hiddenSize != model.config.hiddenSize) {
    throw ArgumentError('Audio and language dimensions differ');
  }
  if (maxSlots < 1 || maxFrames < 1) {
    throw ArgumentError('Invalid reply limits');
  }
  // Omni also suppresses its image and video placeholders.
  final omni =
      (variant ?? generator?.variant ?? Lfm3SpeechVariant.audio) ==
      Lfm3SpeechVariant.omni;
  final state = AudioInterleavingState(
    textEnd: id('<|text_end|>'),
    turnEnd: id('<|im_end|>'),
    mixedStart: id('<|mixed_start|>'),
    audioStart: id('<|audio_start|>'),
  )..mode = mode;
  final suppressed = {
    id('<|audio|>'),
    id('<|audio_out|>'),
    if (omni) ...[id('<|image|>'), id('<|video|>')],
    if (mode == AudioReplyMode.text) ...[
      id('<|audio_start|>'),
      id('<|mixed_start|>'),
    ],
  };
  final audioOut = id('<|audio_out|>');
  final codec = generator?.startCodecStream();
  final audio = <Float32List>[], visible = <int>[], history = <List<int>>[];
  final random = math.Random(seed);
  var frames = 0, generated = 0;
  void emit(Float32List pcm) {
    if (pcm.isEmpty) return;
    audio.add(pcm);
    onAudio?.call(Float32List.fromList(pcm));
  }

  try {
    while (!state.finished && generated < maxSlots) {
      check?.call();
      late final int token;
      Float32List? embedding;
      List<int>? codes;
      if (state.nextIsAudio) {
        final g = generator!;
        codes = g.sampleFrame(model.lastHiddenState, random, history: history);
        final eos = codes.contains(g.eosCode);
        if (eos) {
          codes = List.filled(g.codebooks, g.eosCode);
        } else {
          if (frames == maxFrames) {
            throw StateError(
              'Speech reply exceeded 30 seconds. Try a shorter question.',
            );
          }
          frames++;
          history.add(codes);
          emit(codec!.add(codes));
        }
        state.acceptAudio(eos: eos);
        token = audioOut;
        embedding = g.feedback(codes);
      } else {
        final scores = model.logits;
        var best = double.negativeInfinity, pick = -1;
        for (var i = 0; i < scores.length; i++) {
          if (!suppressed.contains(i) &&
              scores[i].isFinite &&
              scores[i] > best) {
            pick = i;
            best = scores[i];
          }
        }
        if (pick < 0) throw StateError('No finite reply token');
        state.acceptText(pick);
        token = pick;
        if (tokenizer.addedTokenId(tokenizer.decode([pick])) != pick) {
          visible.add(pick);
          onText?.call(tokenizer.decode(visible));
        }
      }
      check?.call();
      onSlot?.call(token, codes == null ? null : List.unmodifiable(codes));
      await feed(token, embedding, logits: !state.finished);
      generated++;
    }
    if (!state.finished) {
      throw StateError(
        'Speech reply reached the generation limit. Try a shorter question.',
      );
    }
    if (codec != null) emit(codec.finish());
    check?.call();
    final samples = Float32List(audio.fold(0, (n, chunk) => n + chunk.length));
    var at = 0;
    for (final chunk in audio) {
      samples.setAll(at, chunk);
      at += chunk.length;
    }
    if (generator != null &&
        samples.length != frames * generator.frameSamples) {
      throw StateError('Reply codec length differs');
    }
    return SpeechReply(
      tokenizer.decode(visible).trim(),
      samples,
      generator?.sampleRate ?? 24000,
      frames,
      generated,
    );
  } finally {
    codec?.close();
  }
}
