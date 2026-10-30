/// Parses an LFM3 assistant reply into reasoning, text and tool calls.
///
/// The patterns follow `response_template` in tokenizer_config.json:
/// reasoning in `<think>…</think>`, and each call as
///
///     <tool_call>
///     <function=NAME>
///     <parameter=KEY>
///     VALUE
///     </parameter>
///     </function>
///     </tool_call>
///
/// Values stay strings, as in the reference parser. Converting "3" or "true"
/// to typed values is up to the caller, which knows the tool's schema.
library;

final class ToolCall {
  const ToolCall(this.name, this.arguments);

  final String name;
  final Map<String, String> arguments;

  Map<String, Object?> toJson() => {'name': name, 'arguments': arguments};
}

final class AssistantReply {
  const AssistantReply(this.reasoning, this.content, this.toolCalls);

  final String? reasoning;
  final String content;
  final List<ToolCall> toolCalls;
}

final RegExp _toolCall = RegExp(
  r'<tool_call>\s*<function=([^\n>]+)>(.*?)</function>\s*</tool_call>\s*',
  dotAll: true,
);
final RegExp _parameter = RegExp(
  r'<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>(?=\s*(?:<parameter=|$))',
  dotAll: true,
);

/// Parses generated text. The prompt usually ends in `<think>` or
/// `<think></think>`, so the reply may start with the reasoning body itself
/// and contain only the closing tag.
AssistantReply parseAssistantReply(String text) {
  var rest = text;
  final end = rest.indexOf('<|im_end|>');
  if (end >= 0) rest = rest.substring(0, end);

  String? reasoning;
  final close = rest.indexOf('</think>');
  if (close >= 0) {
    var body = rest.substring(0, close);
    if (body.startsWith('<think>')) body = body.substring('<think>'.length);
    reasoning = body.replaceFirst(RegExp(r'^\n'), '');
    rest = rest.substring(close + '</think>'.length).trimLeft();
  }

  final calls = <ToolCall>[];
  final content = rest.replaceAllMapped(_toolCall, (m) {
    final body = m[2]!.trim();
    calls.add(
      ToolCall(m[1]!, {
        for (final p in _parameter.allMatches(body)) p[1]!: p[2]!,
      }),
    );
    return '';
  });
  return AssistantReply(reasoning, content.trim(), calls);
}
