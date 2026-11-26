/// A conversation on one [TextModel] that keeps the model's state between
/// requests, so each turn feeds only the tokens the model has not seen.
///
/// Every prompt starts with the same block: the system prompt and the tool
/// definitions (`<|im_start|>system ... <|im_end|>\n`). [warmUp] feeds it
/// once, or loads it from a state file saved by an earlier run, and
/// checkpoints the state after it, so a new conversation ([clear])
/// restarts there instead of at token 0. Within a conversation the next
/// prompt extends everything the model has seen, its own reply included:
/// the reply is stored as generated, and the template renders it back to
/// the same text. Only the new message costs prompt time.
library;

import 'dart:async';
import 'dart:typed_data';

import 'chat_format.dart';
import 'model.dart';
import 'model_use.dart';
import 'repetition.dart';
import 'sampler.dart';
import 'tokenizer.dart';
import 'tool_calls.dart';

/// One reply and what it cost.
final class ChatTurn {
  ChatTurn({
    required this.text,
    required this.reply,
    required this.promptTokens,
    required this.reusedTokens,
    required this.prefillSeconds,
    required this.generatedTokens,
    required this.decodeSeconds,
    this.thinkingCut = false,
    this.stoppedOnRepetition = false,
    this.rejectedLines = 0,
    this.stoppedOnCheck = false,
  });

  /// The generated text, without the end-of-turn token.
  final String text;
  final AssistantReply reply;
  final int promptTokens;

  /// Prompt tokens whose state was reused rather than computed.
  final int reusedTokens;

  /// Time to the first generated token: feeding the new prompt tokens.
  final double prefillSeconds;
  final int generatedTokens;
  final double decodeSeconds;

  /// Whether the session closed the reasoning itself, with
  /// [ChatSession.thinkingCutoff], because it reached
  /// [ChatSession.thinkingBudget] or started to repeat itself. The
  /// answer then rests on reasoning that did not finish.
  final bool thinkingCut;

  /// Whether the reply ended because [ChatSession.repetitionGuard] saw it
  /// repeat itself, outside the reasoning.
  final bool stoppedOnRepetition;

  /// Lines that `checkLine` rejected and the session sampled again.
  final int rejectedLines;

  /// Whether the reply ended because a line failed `checkLine` on every
  /// try; the reply then ends at the last line that passed.
  final bool stoppedOnCheck;

  int get newTokens => promptTokens - reusedTokens;
}

/// A conversation that no longer fits the model's context: start a new one
/// ([ChatSession.clear]) or load the model with a longer context.
final class ContextLengthExceeded implements Exception {
  const ContextLengthExceeded(this.promptTokens, this.contextLength);

  final int promptTokens;
  final int contextLength;

  @override
  String toString() =>
      'ContextLengthExceeded: the prompt has $promptTokens tokens and the '
      'context holds $contextLength';
}

final class ChatSession {
  ChatSession(
    this.model,
    this.tokenizer, {
    String? system,
    this.tools,
    this.maxNew = 256,
    ChatFormat? format,
    this.sampling = SamplingParams.greedy,
    this.enableThinking = false,
    this.thinkingBudget,
    this.thinkingCutoff = defaultThinkingCutoff,
    this.repetitionGuard,
  }) : _system = system,
       format = format ?? chatFormatFor(model.config),
       _sampler = Sampler(sampling) {
    final budget = thinkingBudget;
    if (budget != null && (budget < 1 || budget >= maxNew)) {
      throw ArgumentError.value(
        budget,
        'thinkingBudget',
        'must be at least 1 and below maxNew ($maxNew)',
      );
    }
  }

  final TextModel model;
  final BpeTokenizer tokenizer;
  final List<Map<String, Object?>>? tools;
  final int maxNew;
  final String? _system;

  /// How prompts are rendered and replies parsed; the model family's own
  /// unless the constructor was given another.
  final ChatFormat format;

  /// How replies pick their tokens: greedy unless set. Draws continue one
  /// random sequence across the conversation's replies.
  final SamplingParams sampling;
  final Sampler _sampler;

  /// The samplers for retrying a rejected line, by retry (from 1).
  final _retrySamplers = <int, Sampler>{};

  /// The sampler for retry [retry] of a line `checkLine` rejected. A retry
  /// under the same parameters mostly draws the same line again: the
  /// mistake is what the model is sure of. Each retry samples hotter
  /// (temperature at least 0.7, plus 0.3 a retry) from more of the
  /// vocabulary (no top-k or top-p; min-p 0.02 keeps out the tail).
  Sampler _retrySampler(int retry) => _retrySamplers.putIfAbsent(
    retry,
    () => Sampler(
      SamplingParams(
        temperature:
            (sampling.temperature < 0.7 ? 0.7 : sampling.temperature) +
            0.3 * retry,
        minP: 0.02,
        presencePenalty: sampling.presencePenalty,
        seed: sampling.seed == null ? null : sampling.seed! + 7919 * retry,
      ),
    ),
  );

  /// Whether the model may reason before it answers. Off, the prompt asks
  /// for a direct answer (LFM3 and Qwen3 both close an empty think block);
  /// on, the reply starts with its reasoning, which [ChatTurn.reply]
  /// separates.
  final bool enableThinking;

  /// With [enableThinking], the most tokens the reply may generate before
  /// its reasoning closes: counted from the start of the reply, so they
  /// include an opening `<think>` only where the family generates it
  /// (Qwen3, not LFM3, whose prompt ends in it). A reply that reaches it
  /// without closing its reasoning gets [thinkingCutoff] fed as if
  /// generated, and answers from there. Null lets the reasoning run up to
  /// [maxNew]. Small models rarely reason better for reasoning longer, and
  /// a cut keeps room in the context for the answer. It must be below
  /// [maxNew]; the cutoff is fed whole, even past [maxNew], so [maxNew]
  /// should leave room past the budget for it and the answer.
  final int? thinkingBudget;

  /// The text that closes cut reasoning, ending in `</think>`.
  final String thinkingCutoff;

  /// Qwen's suggested wording for closing reasoning early.
  static const defaultThinkingCutoff =
      '\n\nConsidering the limited time, I have to give the answer based '
      'on the thinking so far.\n</think>\n\n';

  /// Watches each reply for a loop; null watches nothing. A loop inside
  /// the reasoning (with [enableThinking]) closes it as [thinkingBudget]
  /// does; a loop in the answer ends the reply there.
  final RepetitionGuard? repetitionGuard;

  /// The conversation so far, after the system prompt.
  final messages = <Map<String, Object?>>[];

  /// Whether [send] keeps a copy of the logits its prompt ends with, in
  /// [promptLogits] (for checking that reuse changes nothing).
  bool keepPromptLogits = false;
  Float32List? promptLogits;

  List<Map<String, Object?>> get _systemMessages => [
    if (_system != null) {'role': 'system', 'content': _system},
  ];

  /// The tokens of the system prompt and tools block.
  late final List<int> prefix = tokenizer.encode(
    format.render(_systemMessages, tools: tools, addGenerationPrompt: false),
  );

  /// Feeds the system prompt and tools block and checkpoints after it.
  /// With [statePath], first tries to load exactly that block from the file
  /// (TextModel.loadState), and after computing it saves it there. Returns
  /// whether the state came from the file.
  Future<bool> warmUp({String? statePath}) async {
    beginModelUse(model);
    try {
      return await _warmUp(statePath);
    } finally {
      endModelUse(model);
    }
  }

  Future<bool> _warmUp(String? statePath) async {
    if (statePath != null &&
        await model.loadState(statePath) &&
        _startsWith(prefix, model.history) &&
        model.history.length == prefix.length) {
      return true;
    }
    await model.prefill(prefix);
    if (statePath != null) await model.saveState(statePath);
    return false;
  }

  /// Starts a new conversation; the model's state is kept for reuse.
  void clear() => messages.clear();

  /// The prompt tokens for the conversation [conversation] (after the
  /// system prompt), ending in the assistant's generation prompt.
  List<int> promptIds(List<Map<String, Object?>> conversation) =>
      tokenizer.encode(
        format.render(
          [..._systemMessages, ...conversation],
          tools: tools,
          enableThinking: enableThinking,
        ),
      );

  /// Adds [message] (a user or tool message) and generates the reply
  /// under [sampling], up to [maxNew] tokens or the end of the model's context
  /// (`maxSeq`), with [thinkingBudget] and [repetitionGuard] closing
  /// reasoning that runs long or loops ([ChatTurn.thinkingCut]) and
  /// [repetitionGuard] ending an answer that loops
  /// ([ChatTurn.stoppedOnRepetition]). [SamplingParams.presencePenalty]
  /// counts the tokens this reply has generated.
  /// [onText] streams the reply: after each generated token it gets the
  /// text so far, without an incomplete UTF-8 sequence at the end.
  /// Returning false ends the reply there, which is then stored as it
  /// stands. An asynchronous [onText] lets the caller's event loop run
  /// between tokens, so that a request to stop can arrive.
  /// [suppressedTokens] are never generated in this reply, for example
  /// `<tool_call>` when the caller offers no tools and the model would
  /// otherwise still start a call it cannot run. Decoding picks only
  /// tokens outside the set; end-of-reply tokens are never suppressed. An empty set leaves decoding unchanged.
  /// [checkLine] verifies the reply line by line: whenever a generated
  /// token ends a line, it gets the reply up to that line's newline, and
  /// returning false rejects the line. The session then rewinds the model
  /// to the start of the line and samples it again, up to [lineRetries]
  /// times, after which the reply ends at the last line that passed
  /// ([ChatTurn.stoppedOnCheck]). Rewinding costs a step on attention-only
  /// models (Qwen3) and a line's worth of tokens on others (LFM3, from the
  /// checkpoint at the previous line). Each retry samples the line hotter
  /// than [sampling], even after greedy decoding, since the same
  /// parameters mostly draw the same mistake again. A rejected line is taken back
  /// from [onText] too, which then sees shorter text.
  /// [onLogits] observes the prompt and each decode step synchronously.
  /// Do not modify the view; copy it before retaining it across model calls.
  /// Observer work, [onText]'s included, counts in the generation timings.
  ///
  /// Throws [ContextLengthExceeded] when the prompt leaves no room in the
  /// context for a reply. Whatever [send] throws, from the model, [onLogits]
  /// or [onText], it leaves [messages] as they were before the call.
  Future<ChatTurn> send(
    Map<String, Object?> message, {
    void Function(Float32List logits)? onLogits,
    FutureOr<bool> Function(String text)? onText,
    Set<int> suppressedTokens = const {},
    bool Function(String reply)? checkLine,
    int lineRetries = 4,
  }) async {
    if (lineRetries < 0) {
      throw ArgumentError.value(lineRetries, 'lineRetries', 'is negative');
    }
    beginModelUse(model);
    try {
      return await _send(
        message,
        onLogits,
        onText,
        suppressedTokens,
        checkLine,
        lineRetries,
      );
    } finally {
      endModelUse(model);
    }
  }

  Future<ChatTurn> _send(
    Map<String, Object?> message,
    void Function(Float32List logits)? onLogits,
    FutureOr<bool> Function(String text)? onText,
    Set<int> suppressedTokens,
    bool Function(String reply)? checkLine,
    int lineRetries,
  ) async {
    final suppressed = {...suppressedTokens}
      ..removeAll(model.config.eosTokenIds);
    final ids = promptIds([...messages, message]);
    if (ids.length >= model.maxSeq) {
      throw ContextLengthExceeded(ids.length, model.maxSeq);
    }
    messages.add(message);
    final sw = Stopwatch()..start();
    final int reused;
    try {
      reused = await model.prefill(
        ids,
        checkpoints: _startsWith(ids, prefix) ? [prefix.length] : const [],
      );
    } catch (_) {
      messages.removeLast();
      rethrow;
    }
    final prefillSeconds = sw.elapsedMicroseconds / 1e6;
    if (keepPromptLogits) promptLogits = Float32List.fromList(model.logits);
    onLogits?.call(model.logits);
    final gen = <int>[];
    // Whether each token of [gen] was sampled rather than fed by the
    // session. Only sampled tokens go into [seen]: the presence penalty
    // counts what the model chose, not a cutoff the session fed.
    final sampled = <bool>[];
    final seen = <int>{};
    Future<bool> report() async =>
        onText == null || await onText(_complete(tokenizer.decode(gen)));
    // Feeds [token] as part of the reply; false when the caller stops it.
    Future<bool> feed(int token, {bool wasSampled = false}) async {
      gen.add(token);
      sampled.add(wasSampled);
      if (wasSampled) seen.add(token);
      await model.step(token);
      onLogits?.call(model.logits);
      return report();
    }

    var thinking = enableThinking;
    var thinkingCut = false;
    var stoppedOnRepetition = false;
    // The line being written starts at gen[lineStart]; it is retried
    // [retries] times so far.
    var lineStart = 0;
    var thinkingAtLineStart = thinking;
    var retries = 0;
    var rejectedLines = 0;
    var stoppedOnCheck = false;
    // Takes the reply back to the start of the current line.
    Future<void> rewindLine() async {
      gen.length = lineStart;
      sampled.length = lineStart;
      seen
        ..clear()
        ..addAll([
          for (var i = 0; i < gen.length; i++)
            if (sampled[i]) gen[i],
        ]);
      thinking = thinkingAtLineStart;
      await model.prefill(model.history.sublist(0, ids.length + lineStart));
      onLogits?.call(model.logits);
    }

    sw
      ..reset()
      ..start();
    try {
      decode:
      while (gen.length < maxNew && model.history.length < model.maxSeq) {
        final next = (retries == 0 ? _sampler : _retrySampler(retries)).next(
          model.logits,
          suppressed: suppressed,
          seen: seen,
        );
        if (model.config.isEos(next)) break;
        if (!await feed(next, wasSampled: true)) break;
        if (checkLine != null && tokenizer.decode([next]).contains('\n')) {
          final text = tokenizer.decode(gen);
          if (checkLine(text.substring(0, text.lastIndexOf('\n') + 1))) {
            lineStart = gen.length;
            retries = 0;
            if (thinking && _closesThinking(gen)) thinking = false;
            thinkingAtLineStart = thinking;
            await model.checkpoint();
          } else {
            rejectedLines++;
            await rewindLine();
            if (retries++ >= lineRetries) {
              stoppedOnCheck = true;
              await report();
              break;
            }
            if (!await report()) break;
            continue;
          }
        }
        if (thinking && _closesThinking(gen)) thinking = false;
        final loops = repetitionGuard?.loops(gen) ?? false;
        final budget = thinkingBudget;
        if (thinking && (loops || (budget != null && gen.length >= budget))) {
          thinking = false;
          final cutoff = tokenizer.encode(thinkingCutoff);
          // Without room for the cutoff and an answer, the reasoning ends
          // where the context does, unclosed.
          if (model.history.length + cutoff.length >= model.maxSeq) break;
          for (final token in cutoff) {
            if (!await feed(token)) break decode;
          }
          thinkingCut = true;
          // The cutoff is the session's own: lines start after it.
          lineStart = gen.length;
          thinkingAtLineStart = false;
          retries = 0;
        } else if (loops) {
          stoppedOnRepetition = true;
          break;
        }
      }
    } catch (_) {
      // The model keeps what it has fed; the next prompt, which no longer
      // has this reply, rewinds to the checkpoint at its end.
      messages.removeLast();
      rethrow;
    }
    final decodeSeconds = sw.elapsedMicroseconds / 1e6;
    final text = tokenizer.decode(gen);
    messages.add({'role': 'assistant', 'content': text});
    return ChatTurn(
      text: text,
      reply: format.parseReply(text),
      promptTokens: ids.length,
      reusedTokens: reused,
      prefillSeconds: prefillSeconds,
      generatedTokens: gen.length,
      decodeSeconds: decodeSeconds,
      thinkingCut: thinkingCut,
      stoppedOnRepetition: stoppedOnRepetition,
      rejectedLines: rejectedLines,
      stoppedOnCheck: stoppedOnCheck,
    );
  }

  /// [text] without the replacement character that a multi-byte UTF-8
  /// sequence cut short by the last token decodes to.
  static String _complete(String text) =>
      text.endsWith('\uFFFD') ? text.substring(0, text.length - 1) : text;

  /// Whether the end of [gen] spells `</think>`, however the model split
  /// it into tokens. Checked after every token while the reasoning is
  /// open, so the first match is the token that closed it. Decodes only
  /// the tail: the eight bytes of `</think>` span at most eight tokens.
  bool _closesThinking(List<int> gen) {
    const end = '</think>';
    final from = gen.length > end.length ? gen.length - end.length : 0;
    return tokenizer.decode(gen.sublist(from)).contains(end);
  }

  static bool _startsWith(List<int> ids, List<int> prefix) {
    if (ids.length < prefix.length) return false;
    for (var i = 0; i < prefix.length; i++) {
      if (ids[i] != prefix[i]) return false;
    }
    return true;
  }

  /// The index of the largest of [logits] outside [suppressed]; the first
  /// one on a tie.
  static int greedyToken(
    Float32List logits, {
    Set<int> suppressed = const {},
  }) => argmaxToken(logits, suppressed: suppressed);
}
