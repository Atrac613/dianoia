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
/// Tool search follows the template too. A tool with a truthy
/// `defer_loading` (on the tool or its function) stays out of the `<tools>`
/// block; a tool message brings it in with `tool_reference` items (which
/// name a tool that must still be in `tools`), `{"result": {"tools": ...}}`
/// items or the `additional_tools` field, and consecutive tool messages with
/// the same `tool_call_id` merge into one response.
///
/// Not supported, and rejected instead of rendered differently: images and
/// video.
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
      system = _parseContent(rest.first['content']);
    }
    rest = rest.sublist(1);
  }
  final lines = [
    for (final tool in tools ?? const <Map<String, Object?>>[])
      if (!_truthy(_function(tool)['defer_loading']) &&
          !_truthy(tool['defer_loading']))
        _toolToJson(tool),
  ];
  if (lines.isNotEmpty) {
    system =
        '$system${system.isEmpty ? '' : '\n\n'}<tools>\n${lines.join('\n')}\n</tools>';
  }
  if (system.isNotEmpty) out.write('<|im_start|>system\n$system<|im_end|>\n');

  final lastUser = rest.lastIndexWhere((m) => m['role'] == 'user');
  // Items of tool messages waiting to merge into the next one.
  var pending = <Object?>[];
  for (var i = 0; i < rest.length; i++) {
    final message = rest[i];
    final role = message['role'] as String;
    final next = i + 1 < rest.length ? rest[i + 1] : null;
    final plainTool = role == 'tool' && !_truthy(message['additional_tools']);
    final items = plainTool ? _toolItems(message['content']) : const [];
    if (plainTool &&
        next != null &&
        next['role'] == 'tool' &&
        !_truthy(next['additional_tools']) &&
        _truthy(message['tool_call_id']) &&
        next['tool_call_id'] == message['tool_call_id']) {
      pending = [...pending, ...items];
      continue;
    }
    out.write('<|im_start|>$role\n');
    switch (role) {
      case 'assistant':
        out.write(_assistant(message, preserveThinking || i > lastUser));
      case 'tool' when !plainTool:
        out.write(_additionalTools(message));
      case 'tool' when pending.isNotEmpty:
        out.write(_toolResponse([...pending, ...items], tools));
        pending = [];
      case 'tool':
        out.write(_toolResponse(message['content'] ?? '', tools));
      default:
        if (_truthy(message['content'])) {
          out.write(_parseContent(message['content']));
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
  var content = _truthy(rawContent) ? _parseContent(rawContent) : '';
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

/// The template's tool_to_json: the tool's function (or the tool itself)
/// as JSON, without `defer_loading`.
String _toolToJson(Object? tool) {
  final fn = tool is Map<String, Object?> ? _function(tool) : tool;
  if (fn is! Map<String, Object?>) {
    throw ArgumentError('a tool must be a mapping, got $fn');
  }
  return '{${[for (final MapEntry(:key, :value) in fn.entries)
    if (key != 'defer_loading') '${pyJsonDumps(key)}: ${pyJsonDumps(value)}'].join(', ')}}';
}

/// Tools as the lines of an `<additional_tools>` block: a string as is, a
/// tool or a list of tools as JSON.
String _toolLines(Object? tools) => switch (tools) {
  String() => tools,
  Map() => _toolToJson(tools),
  List() => tools.map(_toolToJson).join('\n'),
  _ => throw ArgumentError(
    'additional tools must be a string, a tool or a '
    'list of tools, got $tools',
  ),
};

/// A tool message's content as the list of items it merges.
List<Object?> _toolItems(Object? content) => switch (content) {
  null => const [],
  String() when content.isEmpty => const [],
  String() => [
    {'type': 'text', 'text': content},
  ],
  List() => content,
  _ => [content],
};

/// A tool message with `additional_tools`: the field's tools, or with a
/// field that is not a tool or a list, the content.
String _additionalTools(Map<String, Object?> message) {
  final field = message['additional_tools'];
  final source = field is Map || field is List ? field : message['content'];
  if (source is String && source.contains('<tool_response>')) return source;
  if (source is String && source.contains('<additional_tools>')) {
    return '<tool_response>\n$source\n</tool_response>';
  }
  if (!_truthy(source)) return '';
  return '<tool_response>\n<additional_tools>\n${_toolLines(source)}'
      '\n</additional_tools>\n</tool_response>';
}

/// The template's render_tool_response. In a list of items, adjacent
/// `tool_reference` items share one `<additional_tools>` block.
String _toolResponse(Object? content, List<Map<String, Object?>>? tools) {
  if (content is String && content.contains('<tool_response>')) {
    return content;
  }
  final body = switch (content) {
    String() || Map() => _parseContent(content),
    List() => [
      for (var i = 0; i < content.length; i++)
        if (_isToolReference(content[i]))
          [
            if (i == 0 || !_isToolReference(content[i - 1]))
              '<additional_tools>\n',
            _referencedTool((content[i] as Map)['name'], tools),
            if (i == content.length - 1 || !_isToolReference(content[i + 1]))
              '\n</additional_tools>',
          ].join()
        else if (content[i] case {'result': {'tools': final found}})
          '<additional_tools>\n${_toolLines(found)}\n</additional_tools>'
        else
          _parseContent(content[i]),
    ].join('\n'),
    _ => pyJsonDumps(content),
  };
  return '<tool_response>\n$body\n</tool_response>';
}

bool _isToolReference(Object? item) =>
    item is Map && item['type'] == 'tool_reference';

/// The JSON of the tool named [name], found among all [tools], deferred or
/// not; the last one wins.
String _referencedTool(Object? name, List<Map<String, Object?>>? tools) {
  final found = (tools ?? const <Map<String, Object?>>[])
      .map(_function)
      .lastWhere((fn) => fn['name'] == name, orElse: () => const {});
  if (found.isEmpty) {
    throw ArgumentError(
      "Tool reference '$name' not found in available tools. Every tool "
      'referenced by a tool_reference must be in `tools` on every request.',
    );
  }
  return _toolToJson(found);
}

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

/// The template's parse_content: strings as is, text items as their text,
/// other values as JSON.
String _parseContent(Object? content) => switch (content) {
  String() => content,
  Map() => _contentItem(content),
  List() =>
    content.map((item) => item is String ? item : _contentItem(item)).join(),
  null => '',
  _ => pyJsonDumps(content),
};

String _contentItem(Object? item) => switch (item) {
  {'type': 'image' || 'video'} => throw UnsupportedError(
    'only text content is supported',
  ),
  {'type': 'text'} => _pyStr((item as Map)['text']),
  _ => pyJsonDumps(item),
};

/// Python's str() for a JSON scalar, as Jinja's `(x or "") | string` does.
String _pyStr(Object? value) => switch (value) {
  _ when !_truthy(value) => '',
  String() => value,
  bool() => value ? 'True' : 'False',
  num() => _pyNumber(value),
  _ => throw UnsupportedError('text must be a string, got $value'),
};

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
