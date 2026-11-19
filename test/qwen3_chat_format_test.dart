// renderQwen3Chat against prompts rendered by Hugging Face's
// apply_chat_template (test/fixtures/qwen3_text_oracle.json, from
// tool/qwen3_oracle.py), and parseQwen3Reply on Qwen3's reply format.
import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

List<Map<String, Object?>> _maps(Object? list) => [
  for (final m in list as List) (m as Map).cast<String, Object?>(),
];

void main() {
  group('template', () {
    final oracle =
        jsonDecode(
              File('test/fixtures/qwen3_text_oracle.json').readAsStringSync(),
            )
            as Map<String, Object?>;
    for (final c
        in (oracle['templates'] as List).cast<Map<String, Object?>>()) {
      test('${c['name']}', () {
        final rendered = const Qwen3ChatFormat().render(
          _maps(c['messages']),
          tools: c['tools'] == null ? null : _maps(c['tools']),
          addGenerationPrompt: c['add_generation_prompt'] as bool,
          enableThinking: (c['enable_thinking'] as bool?) ?? true,
        );
        expect(rendered, c['prompt']);
      });
    }
  });

  group('reply', () {
    test('reasoning, content and a JSON call', () {
      final reply = parseQwen3Reply(
        '<think>\nThe user wants weather.\n</think>\n\nChecking.\n'
        '<tool_call>\n{"name": "get_weather", "arguments": '
        '{"city": "Kobe", "days": 3, "units": {"temp": "C"}}}\n'
        '</tool_call><|im_end|>',
      );
      expect(reply.reasoning, 'The user wants weather.');
      expect(reply.content, 'Checking.');
      expect(reply.toolCalls.single.toJson(), {
        'name': 'get_weather',
        'arguments': {'city': 'Kobe', 'days': '3', 'units': '{"temp": "C"}'},
      });
    });

    test('without thinking the reply is the answer', () {
      final reply = parseQwen3Reply('Paris is the capital.');
      expect(reply.reasoning, isNull);
      expect(reply.content, 'Paris is the capital.');
      expect(reply.toolCalls, isEmpty);
    });

    test('parallel calls, with arguments given as a JSON string', () {
      final reply = parseQwen3Reply(
        '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"name": "b", "arguments": "{\\"x\\": true}"}\n'
        '</tool_call>',
      );
      expect(
        [for (final c in reply.toolCalls) c.toJson()],
        [
          {'name': 'a', 'arguments': <String, String>{}},
          {
            'name': 'b',
            'arguments': {'x': 'true'},
          },
        ],
      );
      expect(reply.content, isEmpty);
    });

    test('a malformed call stays text', () {
      const text = '<tool_call>\n{"name": broken}\n</tool_call>';
      final reply = parseQwen3Reply(text);
      expect(reply.toolCalls, isEmpty);
      expect(reply.content, text);
    });
  });
}
