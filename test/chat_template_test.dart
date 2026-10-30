// renderChat against prompts rendered by Hugging Face's apply_chat_template:
// dart-llm-lab's text oracle and tool/template_oracle.py's tool search cases.
import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

List<Map<String, Object?>> _maps(Object? list) => [
  for (final m in list as List) (m as Map).cast<String, Object?>(),
];

void main() {
  for (final fixture in const ['text_oracle.json', 'template_oracle.json']) {
    group(fixture, () {
      final cases =
          (jsonDecode(File('test/fixtures/$fixture').readAsStringSync())
                  as Map)['templates']
              as List;
      for (final c in cases.cast<Map<String, Object?>>()) {
        test('${c['name']}', () {
          final rendered = renderChat(
            _maps(c['messages']),
            tools: c['tools'] == null ? null : _maps(c['tools']),
            enableThinking: c['enable_thinking'] as bool,
          );
          expect(rendered, c['prompt']);
        });
      }
    });
  }

  test('a tool_reference to a tool not in tools throws', () {
    expect(
      () => renderChat(
        [
          {'role': 'user', 'content': 'Hi'},
          {
            'role': 'tool',
            'content': [
              {'type': 'tool_reference', 'name': 'missing'},
            ],
          },
        ],
        tools: [
          {
            'type': 'function',
            'function': {'name': 'present', 'defer_loading': true},
          },
        ],
      ),
      throwsA(
        isA<ArgumentError>().having(
          (e) => e.message,
          'message',
          contains("'missing'"),
        ),
      ),
    );
  });

  test('images are rejected', () {
    expect(
      () => renderChat([
        {
          'role': 'user',
          'content': [
            {'type': 'image'},
          ],
        },
      ]),
      throwsUnsupportedError,
    );
  });
}
