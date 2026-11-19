/// How a model family spells a conversation and its replies.
///
/// Each architecture's chat template renders messages (Hugging Face JSON
/// shapes, see chat_template.dart) to the text its tokenizer encodes, and
/// its replies carry reasoning and tool calls in its own syntax. A
/// [ChatFormat] holds both directions for one family.
library;

import 'chat_template.dart';
import 'config.dart';
import 'tool_calls.dart';

abstract interface class ChatFormat {
  /// Renders [messages] exactly as the family's
  /// `tokenizer.apply_chat_template` does.
  String render(
    List<Map<String, Object?>> messages, {
    List<Map<String, Object?>>? tools,
    bool addGenerationPrompt = true,
    bool enableThinking = true,
  });

  /// Splits generated text into reasoning, content and tool calls.
  AssistantReply parseReply(String text);
}

/// LFM3's chat_template.jinja and response format.
final class Lfm3ChatFormat implements ChatFormat {
  const Lfm3ChatFormat();

  @override
  String render(
    List<Map<String, Object?>> messages, {
    List<Map<String, Object?>>? tools,
    bool addGenerationPrompt = true,
    bool enableThinking = true,
  }) => renderChat(
    messages,
    tools: tools,
    addGenerationPrompt: addGenerationPrompt,
    enableThinking: enableThinking,
  );

  @override
  AssistantReply parseReply(String text) => parseAssistantReply(text);
}

/// The chat format of the family [config] describes.
ChatFormat chatFormatFor(TextModelConfig config) =>
    switch (config.architecture) {
      ModelArchitecture.lfm3 => const Lfm3ChatFormat(),
      ModelArchitecture.qwen3 => throw UnsupportedError(
        'no chat format for Qwen3 yet',
      ),
    };
