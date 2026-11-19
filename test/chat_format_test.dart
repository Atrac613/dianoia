import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

void main() {
  final textConfig = TextModelConfig.fromJson(
    ((jsonDecode(File('test/fixtures/vision/config.json').readAsStringSync())
                as Map)['text_config']
            as Map)
        .cast<String, Object?>(),
  );

  test('an LFM3 config gets the LFM3 format', () {
    expect(textConfig.architecture, ModelArchitecture.lfm3);
    expect(chatFormatFor(textConfig), isA<Lfm3ChatFormat>());
  });

  test('the LFM3 format renders and parses as the template does', () {
    const format = Lfm3ChatFormat();
    final messages = [
      {'role': 'system', 'content': 'Be brief.'},
      {'role': 'user', 'content': 'Weather in Paris?'},
    ];
    for (final thinking in [true, false]) {
      expect(
        format.render(messages, enableThinking: thinking),
        renderChat(messages, enableThinking: thinking),
      );
    }
    const reply =
        '<think>\nCheck it.</think>\n\nSure.<tool_call>\n'
        '<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n'
        '</function>\n</tool_call><|im_end|>';
    final parsed = format.parseReply(reply);
    expect(parsed.reasoning, 'Check it.');
    expect(parsed.content, 'Sure.');
    expect(parsed.toolCalls.single.toJson(), {
      'name': 'get_weather',
      'arguments': {'city': 'Paris'},
    });
  });
}
