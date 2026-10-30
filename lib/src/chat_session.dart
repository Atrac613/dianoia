/// A conversation on one [Lfm3Model] that keeps the model's state between
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

import 'chat_template.dart';
import 'model.dart';
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
  }) : _system = system;

  final Lfm3Model model;
  final Lfm3Tokenizer tokenizer;
  final List<Map<String, Object?>>? tools;
  final int maxNew;
  final String? _system;

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
    renderChat(_systemMessages, tools: tools, addGenerationPrompt: false),
  );

  /// Feeds the system prompt and tools block and checkpoints after it.
  /// With [statePath], first tries to load exactly that block from the file
  /// (Lfm3Model.loadState), and after computing it saves it there. Returns
  /// whether the state came from the file.
  Future<bool> warmUp({String? statePath}) async {
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
        renderChat(
          [..._systemMessages, ...conversation],
          tools: tools,
          enableThinking: false,
        ),
      );

  /// Adds [message] (a user or tool message) and generates the reply
  /// greedily, up to [maxNew] tokens or the end of the model's context
  /// (`maxSeq`).
  /// [onText] streams the reply: after each generated token it gets the
  /// text so far, without an incomplete UTF-8 sequence at the end.
  /// Returning false ends the reply there, which is then stored as it
  /// stands. An asynchronous [onText] lets the caller's event loop run
  /// between tokens, so that a request to stop can arrive.
  /// [onLogits] observes the prompt and each decode step synchronously.
  /// Do not modify the view; copy it before retaining it across model calls.
  /// Observer work, [onText]'s included, counts in the generation timings.
  ///
  /// Throws [ContextLengthExceeded], and leaves the conversation as it was,
  /// when the prompt leaves no room in the context for a reply.
  Future<ChatTurn> send(
    Map<String, Object?> message, {
    void Function(Float32List logits)? onLogits,
    FutureOr<bool> Function(String text)? onText,
  }) async {
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
    sw
      ..reset()
      ..start();
    while (gen.length < maxNew && model.history.length < model.maxSeq) {
      final next = _argmax(model.logits);
      if (next == model.config.eosTokenId) break;
      gen.add(next);
      await model.step(next);
      onLogits?.call(model.logits);
      if (onText != null && !await onText(_complete(tokenizer.decode(gen)))) {
        break;
      }
    }
    final decodeSeconds = sw.elapsedMicroseconds / 1e6;
    final text = tokenizer.decode(gen);
    messages.add({'role': 'assistant', 'content': text});
    return ChatTurn(
      text: text,
      reply: parseAssistantReply(text),
      promptTokens: ids.length,
      reusedTokens: reused,
      prefillSeconds: prefillSeconds,
      generatedTokens: gen.length,
      decodeSeconds: decodeSeconds,
    );
  }

  /// [text] without the replacement character that a multi-byte UTF-8
  /// sequence cut short by the last token decodes to.
  static String _complete(String text) =>
      text.endsWith('\uFFFD') ? text.substring(0, text.length - 1) : text;

  static bool _startsWith(List<int> ids, List<int> prefix) {
    if (ids.length < prefix.length) return false;
    for (var i = 0; i < prefix.length; i++) {
      if (ids[i] != prefix[i]) return false;
    }
    return true;
  }

  static int _argmax(Float32List v) {
    var best = 0;
    for (var i = 1; i < v.length; i++) {
      if (v[i] > v[best]) best = i;
    }
    return best;
  }
}
