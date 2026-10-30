// Streaming and stopping a reply. Needs a Q4 model bundle (config.json,
// tokenizer.json, model.safetensors, lfm3-q4.cache) in DIANOIA_LFM3_DIR, and
// the native kernels; skipped otherwise.
import 'dart:io';

import 'package:dianoia/dianoia.dart';
import 'package:test/test.dart';

final _modelDir = Platform.environment['DIANOIA_LFM3_DIR'];

void main() {
  group('ChatSession.send', () {
    late ChatSession session;

    setUpAll(() async {
      final model = await LocalModel.load(_modelDir!, contextLength: 512);
      session = model.chat(system: 'Answer briefly.', maxNewTokens: 48);
      await session.warmUp();
    });

    setUp(() => session.clear());

    test('onText sees the reply grow to the final text', () async {
      final seen = <String>[];
      final turn = await session.send(
        {'role': 'user', 'content': 'Name three colors.'},
        onText: (text) {
          seen.add(text);
          return true;
        },
      );
      expect(seen, hasLength(turn.generatedTokens));
      for (var i = 1; i < seen.length; i++) {
        expect(seen[i].startsWith(seen[i - 1]), isTrue);
      }
      expect(seen.last, turn.text);
    });

    test('returning false stops the reply and keeps it', () async {
      final turn = await session.send({
        'role': 'user',
        'content': '日本の首都について説明して。',
      }, onText: (text) async => text.length < 4);
      expect(turn.generatedTokens, lessThan(session.maxNew));
      expect(turn.text.length, greaterThanOrEqualTo(4));
      expect(session.messages.last, {
        'role': 'assistant',
        'content': turn.text,
      });

      // The conversation goes on from the stopped reply.
      final next = await session.send({'role': 'user', 'content': 'Thanks!'});
      expect(next.reusedTokens, greaterThan(session.prefix.length));
    });

    test('a prompt past the context throws and changes nothing', () async {
      await session.send({'role': 'user', 'content': 'Hello.'});
      final before = [...session.messages];
      await expectLater(
        session.send({'role': 'user', 'content': 'word ' * 600}),
        throwsA(isA<ContextLengthExceeded>()),
      );
      expect(session.messages, before);
      final next = await session.send({'role': 'user', 'content': 'Hi.'});
      expect(next.reusedTokens, greaterThan(session.prefix.length));
    });
  }, skip: _modelDir == null ? 'set DIANOIA_LFM3_DIR to a Q4 bundle' : false);
}
