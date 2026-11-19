/// Qwen3's chat_template.jinja and reply format, translated to Dart.
///
/// Messages and tools use the Hugging Face JSON shapes (see
/// chat_template.dart). The template differs from LFM3's in ways a client
/// sees:
///
/// * no BOS token; tools go in a `# Tools` system block, one JSON object
///   per line inside `<tools></tools>`;
/// * calls are `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`
///   and results come back as user turns of `<tool_response>` blocks;
/// * reasoning (`reasoning_content`, or `<think>` inside the content) is
///   kept only for assistant turns after the last real user query;
/// * `enableThinking: false` ends the generation prompt with an empty
///   `<think>\n\n</think>\n\n`, so the model answers directly.
library;

import 'dart:convert';

import 'chat_format.dart';
import 'chat_template.dart' show pyJsonDumps;
import 'tool_calls.dart';

final class Qwen3ChatFormat implements ChatFormat {
  const Qwen3ChatFormat();

  @override
  String render(
    List<Map<String, Object?>> messages, {
    List<Map<String, Object?>>? tools,
    bool addGenerationPrompt = true,
    bool enableThinking = true,
  }) => renderQwen3Chat(
    messages,
    tools: tools,
    addGenerationPrompt: addGenerationPrompt,
    enableThinking: enableThinking,
  );

  @override
  AssistantReply parseReply(String text) => parseQwen3Reply(text);
}

/// Renders a conversation exactly as Qwen3's `apply_chat_template` does.
/// [enableThinking] true matches the template's default (left undefined).
String renderQwen3Chat(
  List<Map<String, Object?>> messages, {
  List<Map<String, Object?>>? tools,
  bool addGenerationPrompt = true,
  bool enableThinking = true,
}) {
  final out = StringBuffer();
  final firstIsSystem =
      messages.isNotEmpty && messages.first['role'] == 'system';
  if (tools != null && tools.isNotEmpty) {
    out.write('<|im_start|>system\n');
    if (firstIsSystem) out.write('${messages.first['content']}\n\n');
    out.write(
      '# Tools\n\nYou may call one or more functions to assist with the user '
      'query.\n\nYou are provided with function signatures within '
      '<tools></tools> XML tags:\n<tools>',
    );
    for (final tool in tools) {
      out.write('\n${pyJsonDumps(tool)}');
    }
    out.write(
      '\n</tools>\n\nFor each function call, return a json object with '
      'function name and arguments within <tool_call></tool_call> XML '
      'tags:\n<tool_call>\n{"name": <function-name>, "arguments": '
      '<args-json-object>}\n</tool_call><|im_end|>\n',
    );
  } else if (firstIsSystem) {
    out.write('<|im_start|>system\n${messages.first['content']}<|im_end|>\n');
  }

  // The last user message that is a query rather than relayed tool output.
  var lastQuery = messages.length - 1;
  for (var i = messages.length - 1; i >= 0; i--) {
    final m = messages[i];
    final content = m['content'];
    if (m['role'] == 'user' &&
        content is String &&
        !(content.startsWith('<tool_response>') &&
            content.endsWith('</tool_response>'))) {
      lastQuery = i;
      break;
    }
  }

  for (var i = 0; i < messages.length; i++) {
    final m = messages[i];
    final role = m['role'];
    var content = m['content'] is String ? m['content'] as String : '';
    if (role == 'user' || (role == 'system' && i > 0)) {
      out.write('<|im_start|>$role\n$content<|im_end|>\n');
    } else if (role == 'assistant') {
      var reasoning = '';
      if (m['reasoning_content'] case final String r) {
        reasoning = r;
      } else if (content.contains('</think>')) {
        reasoning = _lstripNewlines(
          _rstripNewlines(
            content.split('</think>').first,
          ).split('<think>').last,
        );
        content = _lstripNewlines(content.split('</think>').last);
      }
      final last = i == messages.length - 1;
      if (i > lastQuery && (last || reasoning.isNotEmpty)) {
        out.write(
          '<|im_start|>assistant\n<think>\n'
          '${_rstripNewlines(_lstripNewlines(reasoning))}\n</think>\n\n'
          '${_lstripNewlines(content)}',
        );
      } else {
        out.write('<|im_start|>assistant\n$content');
      }
      final calls = m['tool_calls'];
      if (calls is List && calls.isNotEmpty) {
        for (var c = 0; c < calls.length; c++) {
          if (c > 0 || content.isNotEmpty) out.write('\n');
          var call = (calls[c] as Map).cast<String, Object?>();
          if (call['function'] case final Map function
              when function.isNotEmpty) {
            call = function.cast<String, Object?>();
          }
          final arguments = call['arguments'];
          out.write(
            '<tool_call>\n{"name": "${call['name']}", "arguments": '
            '${arguments is String ? arguments : pyJsonDumps(arguments)}}'
            '\n</tool_call>',
          );
        }
      }
      out.write('<|im_end|>\n');
    } else if (role == 'tool') {
      if (i == 0 || messages[i - 1]['role'] != 'tool') {
        out.write('<|im_start|>user');
      }
      out.write('\n<tool_response>\n$content\n</tool_response>');
      if (i == messages.length - 1 || messages[i + 1]['role'] != 'tool') {
        out.write('<|im_end|>\n');
      }
    }
  }
  if (addGenerationPrompt) {
    out.write('<|im_start|>assistant\n');
    if (!enableThinking) out.write('<think>\n\n</think>\n\n');
  }
  return out.toString();
}

String _lstripNewlines(String s) => s.replaceFirst(RegExp(r'^\n+'), '');
String _rstripNewlines(String s) => s.replaceFirst(RegExp(r'\n+$'), '');

final RegExp _toolCall = RegExp(
  r'<tool_call>\s*(.*?)\s*</tool_call>\s*',
  dotAll: true,
);

/// Parses text Qwen3 generated after its generation prompt. With thinking
/// enabled the reply starts with `<think>`; with it disabled the prompt
/// already closed the block, so there is no reasoning. A `<tool_call>`
/// whose body is not a JSON call stays in the content. Argument values
/// that are not strings are kept as their JSON text, as [ToolCall]
/// documents.
AssistantReply parseQwen3Reply(String text) {
  var rest = text;
  for (final end in const ['<|im_end|>', '<|endoftext|>']) {
    final at = rest.indexOf(end);
    if (at >= 0) rest = rest.substring(0, at);
  }

  String? reasoning;
  final close = rest.indexOf('</think>');
  if (close >= 0) {
    var body = rest.substring(0, close);
    final open = body.indexOf('<think>');
    if (open >= 0) body = body.substring(open + '<think>'.length);
    reasoning = _rstripNewlines(_lstripNewlines(body));
    rest = rest.substring(close + '</think>'.length);
  }

  final calls = <ToolCall>[];
  final content = rest.replaceAllMapped(_toolCall, (m) {
    final call = _jsonCall(m[1]!);
    if (call == null) return m[0]!;
    calls.add(call);
    return '';
  });
  return AssistantReply(reasoning, content.trim(), calls);
}

ToolCall? _jsonCall(String body) {
  Object? decoded;
  try {
    decoded = jsonDecode(body);
  } on FormatException {
    return null;
  }
  if (decoded case {'name': final String name}) {
    var arguments = decoded['arguments'] ?? const <String, Object?>{};
    if (arguments is String) {
      try {
        arguments = jsonDecode(arguments);
      } on FormatException {
        return null;
      }
    }
    if (arguments is! Map) return null;
    return ToolCall(name, {
      for (final MapEntry(:key, :value) in arguments.entries)
        '$key': value is String ? value : pyJsonDumps(value),
    });
  }
  return null;
}
