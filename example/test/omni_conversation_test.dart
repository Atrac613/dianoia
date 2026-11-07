import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:dianoia_example/src/omni_conversation.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  OmniImage image(int tokens) =>
      (rows: Float32List(tokens * 2), tokens: tokens);

  test('replays the photo once, earlier answers as text, audio in order', () {
    final chat = OmniConversation(2);
    final photo = image(3);
    expect(chat.messages(speak: false, text: 'What is it?', images: [photo]), [
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
    chat.add(text: 'What is it?', images: [photo], answer: 'Shapes.');
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
    expect(chat.imageRows(), [photo.rows]);
    expect(chat.imageTokens(), [3]);
  });

  test('a chat keeps its system prompt and images from any turn', () {
    final chat = OmniConversation(2, system: 'Be brief.');
    final first = image(1), second = image(2);
    chat.add(text: 'Hi', answer: 'Hello.');
    chat.add(images: [first], answer: 'A cat.');
    expect(chat.messages(speak: false, text: 'And this?', images: [second]), [
      {'role': 'system', 'content': 'Be brief.'},
      {
        'role': 'user',
        'content': [
          {'type': 'text', 'text': 'Hi'},
        ],
      },
      {'role': 'assistant', 'content': 'Hello.'},
      {
        'role': 'user',
        'content': [
          {'type': 'image'},
        ],
      },
      {'role': 'assistant', 'content': 'A cat.'},
      {
        'role': 'user',
        'content': [
          {'type': 'image'},
          {'type': 'text', 'text': 'And this?'},
        ],
      },
    ]);
    expect(chat.imageRows([second]), [first.rows, second.rows]);
    expect(chat.imageTokens([second]), [1, 2]);
    chat.clear();
    expect(chat.turns, isEmpty);
  });

  test('keeps only the answers the caller received', () {
    final chat = OmniConversation(2)
      ..add(text: 'a', answer: 'A')
      ..add(text: 'b', answer: 'B');
    chat.keep(1);
    expect(chat.turns.map((t) => t.answer), ['A']);
    expect(() => chat.keep(2), throwsStateError);
    expect(() => chat.add(answer: 'C'), throwsArgumentError);
    expect(
      () => chat.add(images: [(rows: Float32List(3), tokens: 1)], answer: 'C'),
      throwsArgumentError,
    );
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
