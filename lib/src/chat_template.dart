/// LFM3's chat_template.jinja, translated to Dart for the features a chat
/// client needs.
///
/// Messages and tools use the Hugging Face JSON shapes, so the same values
/// can go to `apply_chat_template` and here:
///
///     {"role": "system" | "user" | "assistant" | "tool",
///      "content": String or a list of {"type": "text", "text": ...},
///      "tool_calls": [{"function": {"name": ..., "arguments": {...}}}],
///      "reasoning": optional, also read as "thinking" or "reasoning_content"}
///
/// Not supported, and rejected instead of rendered differently: images and
/// video, deferred tools (`defer_loading`), `additional_tools`, and merging
/// consecutive tool messages by `tool_call_id`.
library;

/// Renders a conversation exactly as `tokenizer.apply_chat_template` does.
String renderChat(
  List<Map<String, Object?>> messages, {
  List<Map<String, Object?>>? tools,
  bool addGenerationPrompt = true,
  bool enableThinking = true,
  bool preserveThinking = false,
}) {
  final out = StringBuffer('<|startoftext|>');
  var rest = messages;
  var system = '';
  if (rest.isNotEmpty && rest.first['role'] == 'system') {
    if (_truthy(rest.first['content'])) {
      system = _content(rest.first['content']);
    }
    rest = rest.sublist(1);
  }
  if (tools != null && tools.isNotEmpty) {
    final lines = <String>[];
    for (final tool in tools) {
      final fn = _function(tool);
      if (_truthy(fn['defer_loading']) || _truthy(tool['defer_loading'])) {
        throw UnsupportedError('deferred tools are not supported');
      }
      lines.add(_toolToJson(fn));
    }
    system =
        '$system${system.isEmpty ? '' : '\n\n'}<tools>\n${lines.join('\n')}\n</tools>';
  }
  if (system.isNotEmpty) out.write('<|im_start|>system\n$system<|im_end|>\n');

  final lastUser = rest.lastIndexWhere((m) => m['role'] == 'user');
  for (var i = 0; i < rest.length; i++) {
    final message = rest[i];
    final role = message['role'] as String;
    out.write('<|im_start|>$role\n');
    switch (role) {
      case 'assistant':
        out.write(_assistant(message, preserveThinking || i > lastUser));
      case 'tool':
        if (_truthy(message['additional_tools'])) {
          throw UnsupportedError('additional_tools is not supported');
        }
        final next = i + 1 < rest.length ? rest[i + 1] : null;
        if (next != null &&
            next['role'] == 'tool' &&
            message['tool_call_id'] != null &&
            next['tool_call_id'] == message['tool_call_id']) {
          throw UnsupportedError('merging tool messages is not supported');
        }
        final content = message['content'] ?? '';
        if (content is! String) {
          throw UnsupportedError('tool content must be a string');
        }
        out.write(
          content.contains('<tool_response>')
              ? content
              : '<tool_response>\n$content\n</tool_response>',
        );
      default:
        if (_truthy(message['content'])) {
          out.write(_content(message['content']));
        }
    }
    out.write('<|im_end|>\n');
  }
  if (addGenerationPrompt) {
    out.write('<|im_start|>assistant\n');
    out.write(enableThinking ? '<think>' : '<think></think>');
  }
  return out.toString();
}

const _continueFinalMessageTag = 'CONTINUE_FINAL_MESSAGE_TAG ';

String _assistant(Map<String, Object?> message, bool keepThinking) {
  final out = StringBuffer();
  // Jinja's "is defined" is true for a present key even if its value is null.
  final thinkingKey = const [
    'thinking',
    'reasoning',
    'reasoning_content',
  ].where(message.containsKey).firstOrNull;
  final rawContent = message['content'];
  if (thinkingKey != null && keepThinking) {
    out.write('<think>${message[thinkingKey] ?? ''}</think>');
  } else if (keepThinking &&
      !(rawContent is String ? rawContent : '').contains('</think>')) {
    out.write('<think></think>');
  }
  var content = _truthy(rawContent) ? _content(rawContent) : '';
  if (!keepThinking) {
    if (content.contains('</think>')) {
      content = content.split('</think>').last.trim();
    }
    content = '<think></think>$content';
  }
  final continueFinal = content.endsWith(_continueFinalMessageTag);
  if (continueFinal) {
    content = content.substring(
      0,
      content.length - _continueFinalMessageTag.length,
    );
  }
  out.write(content);
  final toolCalls = message['tool_calls'];
  if (_truthy(toolCalls)) out.write(_toolCalls(toolCalls as List));
  if (continueFinal) out.write(_continueFinalMessageTag);
  return out.toString();
}

String _toolCalls(List calls) {
  final out = StringBuffer();
  for (var i = 0; i < calls.length; i++) {
    if (i > 0) out.write('\n\n');
    final call = calls[i] as Map<String, Object?>;
    final fn = _function(call);
    final args = fn['arguments'];
    if (args is String && !const ['', '{}', 'null'].contains(args.trim())) {
      throw ArgumentError(
        'Tool call arguments must be a mapping, got a JSON-encoded string',
      );
    }
    out.write('<tool_call>\n<function=${fn['name']}>\n');
    if (args is Map) {
      for (final MapEntry(:key, :value) in args.entries) {
        out.write('<parameter=$key>\n${_renderValue(value)}\n</parameter>\n');
      }
    }
    out.write('</function>\n</tool_call>');
  }
  return out.toString();
}

Map<String, Object?> _function(Map<String, Object?> tool) =>
    tool.containsKey('function')
    ? tool['function'] as Map<String, Object?>
    : tool;

String _toolToJson(Map<String, Object?> fn) =>
    '{${[for (final MapEntry(:key, :value) in fn.entries)
      if (key != 'defer_loading') '${pyJsonDumps(key)}: ${pyJsonDumps(value)}'].join(', ')}}';

/// The template's render_value: strings raw, JSON literals for booleans and
/// null, JSON for containers, Python's str() for numbers.
String _renderValue(Object? value) => switch (value) {
  String() => value,
  bool() => value ? 'true' : 'false',
  null => 'null',
  Map() || List() => pyJsonDumps(value),
  num() => _pyNumber(value),
  _ => value.toString(),
};

String _content(Object? content) {
  if (content is String) return content;
  if (content is List) {
    return [
      for (final item in content)
        if (item is String)
          item
        else if (item is Map && item['type'] == 'text')
          '${item['text'] ?? ''}'
        else
          throw UnsupportedError('only text content is supported'),
    ].join();
  }
  throw UnsupportedError('only text content is supported');
}

/// Jinja truthiness for JSON values.
bool _truthy(Object? value) => switch (value) {
  null => false,
  bool() => value,
  String() => value.isNotEmpty,
  List() => value.isNotEmpty,
  Map() => value.isNotEmpty,
  num() => value != 0,
  _ => true,
};

/// `json.dumps(value, ensure_ascii=False)`: ", " and ": " separators, keys
/// in insertion order, non-ASCII kept. Dart's jsonEncode omits the spaces,
/// which would change the prompt and therefore the tokens.
String pyJsonDumps(Object? value) => switch (value) {
  null => 'null',
  bool() => value ? 'true' : 'false',
  num() => _pyNumber(value),
  String() => _pyJsonString(value),
  List() => '[${value.map(pyJsonDumps).join(', ')}]',
  Map() =>
    '{${[for (final MapEntry(:key, :value) in value.entries) '${_pyJsonString('$key')}: ${pyJsonDumps(value)}'].join(', ')}}',
  _ => throw ArgumentError('cannot encode ${value.runtimeType} as JSON'),
};

String _pyJsonString(String s) {
  final out = StringBuffer('"');
  for (final unit in s.codeUnits) {
    switch (unit) {
      case 0x22:
        out.write(r'\"');
      case 0x5c:
        out.write(r'\\');
      case 0x0a:
        out.write(r'\n');
      case 0x0d:
        out.write(r'\r');
      case 0x09:
        out.write(r'\t');
      case 0x08:
        out.write(r'\b');
      case 0x0c:
        out.write(r'\f');
      default:
        if (unit < 0x20) {
          out.write('\\u${unit.toRadixString(16).padLeft(4, '0')}');
        } else {
          out.writeCharCode(unit);
        }
    }
  }
  out.write('"');
  return out.toString();
}

/// Python's repr for numbers: integers as is; floats in the shortest
/// round-trip form, switching to exponent notation below 1e-4 or from 1e16,
/// with at least two exponent digits ("1e-07").
String _pyNumber(num value) {
  if (value is int) return value.toString();
  final d = value as double;
  if (d.isNaN) return 'NaN';
  if (d.isInfinite) return d > 0 ? 'Infinity' : '-Infinity';
  if (d == 0) return d.isNegative ? '-0.0' : '0.0';
  final exponential = d.toStringAsExponential();
  final e = exponential.indexOf('e');
  final exponent = int.parse(exponential.substring(e + 1));
  if (exponent < -4 || exponent >= 16) {
    final mantissa = exponential.substring(0, e);
    final sign = exponent < 0 ? '-' : '+';
    return '$mantissa'
        'e$sign${exponent.abs().toString().padLeft(2, '0')}';
  }
  final positional = d.toString();
  return positional.contains('.') || positional.contains('e')
      ? positional
      : '$positional.0';
}
