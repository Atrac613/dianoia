import 'package:dianoia_example/src/chat_controller.dart';
import 'package:dianoia_example/src/message_bubble.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  Widget host(ChatMessage message) => MaterialApp(
    home: Scaffold(body: MessageBubble(message: message)),
  );

  testWidgets('a reply shows its text and speed', (tester) async {
    await tester.pumpWidget(
      host(
        ChatMessage(ChatRole.assistant, 'Tokyo.')
          ..stats = '3 tokens · 29.4 tok/s · first token 120 ms',
      ),
    );
    expect(find.text('Tokyo.'), findsOneWidget);
    expect(find.textContaining('29.4 tok/s'), findsOneWidget);
  });

  testWidgets('a reply without text yet shows progress', (tester) async {
    await tester.pumpWidget(host(ChatMessage(ChatRole.assistant, '')));
    expect(find.byType(CircularProgressIndicator), findsOneWidget);
  });
}
