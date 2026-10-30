import 'package:dianoia_example/src/chat_controller.dart';
import 'package:dianoia/dianoia.dart';
import 'package:dianoia_example/src/message_bubble.dart';
import 'package:dianoia_example/src/model_setup.dart';
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

  testWidgets('without a model it offers the download and its license', (
    tester,
  ) async {
    final chat = ChatController()
      ..setup = Setup.needsModel
      ..spec = ModelSpec.lfm3Q4;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: ModelSetup(chat: chat)),
      ),
    );
    expect(find.textContaining('1.07 GB to download'), findsOneWidget);
    expect(find.text('License: LFM Open License v1.0'), findsOneWidget);
    expect(find.text('Download'), findsOneWidget);
  });

  testWidgets('a download in progress shows how far it got', (tester) async {
    final chat = ChatController()
      ..setup = Setup.installing
      ..spec = ModelSpec.lfm3Q4
      ..installBytes = 534000000
      ..installTotal = ModelSpec.lfm3Q4.downloadBytes;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: ModelSetup(chat: chat)),
      ),
    );
    expect(find.text('0.53 of 1.07 GB'), findsOneWidget);
    expect(find.text('Cancel'), findsOneWidget);
  });
}
