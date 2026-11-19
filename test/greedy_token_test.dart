import 'dart:typed_data';

import 'package:dianoia/dianoia.dart';
import 'package:test/test.dart';

void main() {
  group('ChatSession.greedyToken', () {
    final logits = Float32List.fromList([0.5, 2.0, 2.0, 1.5]);

    test('picks the first largest logit', () {
      expect(ChatSession.greedyToken(logits), 1);
    });

    test('skips suppressed tokens', () {
      expect(ChatSession.greedyToken(logits, suppressed: {1}), 2);
      expect(ChatSession.greedyToken(logits, suppressed: {1, 2}), 3);
    });

    test('throws when every token is suppressed', () {
      expect(
        () => ChatSession.greedyToken(logits, suppressed: {0, 1, 2, 3}),
        throwsStateError,
      );
    });
  });
}
