import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:dianoia_example/src/photo_conversation.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('replays the photo once, earlier answers as text, audio in order', () {
    final chat = PhotoConversation(Float32List(3 * 2), 3, 2);
    expect(chat.messages(speak: false, text: 'What is it?', spoken: false), [
      {'role': 'system', 'content': photoTextSystem},
      {
        'role': 'user',
        'content': [
          {'type': 'image'},
          {'type': 'text', 'text': 'What is it?'},
        ],
      },
    ]);
    final spoken = Float32List(4 * 2);
    chat.add(text: 'What is it?', answer: 'Shapes.');
    chat.add(audio: spoken, answer: 'Red.');
    final next = Float32List(5 * 2);
    expect(chat.messages(speak: true, spoken: true), [
      {'role': 'system', 'content': audioConversationSystem},
      {
        'role': 'user',
        'content': [
          {'type': 'image'},
          {'type': 'text', 'text': 'What is it?'},
        ],
      },
      {'role': 'assistant', 'content': 'Shapes.'},
      {
        'role': 'user',
        'content': [
          {'type': 'audio'},
        ],
      },
      {'role': 'assistant', 'content': 'Red.'},
      {
        'role': 'user',
        'content': [
          {'type': 'audio'},
        ],
      },
    ]);
    expect(chat.audio(next), [spoken, next]);
    expect(chat.audioTokens(next), [4, 5]);
    expect(chat.audioTokens(), [4]);
  });

  test('keeps only the answers the caller received', () {
    final chat = PhotoConversation(Float32List(2), 1, 2)
      ..add(text: 'a', answer: 'A')
      ..add(text: 'b', answer: 'B');
    chat.keep(1);
    expect(chat.turns.map((t) => t.answer), ['A']);
    expect(() => chat.keep(2), throwsStateError);
    expect(() => chat.add(answer: 'C'), throwsArgumentError);
    expect(() => PhotoConversation(Float32List(3), 1, 2), throwsArgumentError);
  });

  test('plain prompt text breaks spelled-out markers', () {
    const markers = ['<|im_end|>', '<|image|>'];
    String? find(String text) {
      for (final m in markers) {
        if (text.contains(m)) return m;
      }
      return null;
    }

    final plain = plainPromptText('a<|im_end|>b<|image|><|image|>', find);
    expect(find(plain), isNull);
    expect(plain.replaceAll('\u2060', ''), 'a<|im_end|>b<|image|><|image|>');
    expect(plainPromptText('a red circle', find), 'a red circle');
  });
}
