import 'dart:async';

import 'package:dianoia_example/main.dart';
import 'package:dianoia_example/src/example_model.dart';
import 'package:dianoia_example/src/chat_controller.dart';
import 'package:dianoia_example/src/chat_page.dart';
import 'package:dianoia_example/src/model_selection_page.dart';
import 'package:dianoia/dianoia.dart';
import 'package:dianoia_example/src/message_bubble.dart';
import 'package:dianoia_example/src/model_setup.dart';
import 'package:flutter/material.dart';
import 'package:dianoia_example/src/model_size.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('model sizes format numeric estimates consistently', () {
    expect(gigabytes(1540284520), '1.54');
    expect(gigabytes(5000000000, compact: true), '5');
    expect(gigabytes(10500000000, compact: true), '10.5');
    expect(gigabytes(0, compact: true), '0');
  });

  Widget host(ChatMessage message) => MaterialApp(
    home: Scaffold(body: MessageBubble(message: message)),
  );

  testWidgets('launch shows model choices before starting a chat', (
    tester,
  ) async {
    await tester.pumpWidget(const DianoiaExampleApp());
    expect(find.text('Choose a model'), findsOneWidget);
    expect(find.text(ModelSpec.lfm3Q4.name), findsOneWidget);
    expect(find.text(ExampleModel.lfm3Vl.name), findsOneWidget);
    expect(find.textContaining('1.45 GB download'), findsOneWidget);
    expect(find.textContaining('1.54 GB after setup'), findsOneWidget);
    final vl = ExampleModel.lfm3Vl.downloadSpec as VisionGgufSpec;
    expect(vl.language.name, 'LFM3-VL-2B-QAD-Q4_0.gguf');
    expect(vl.vision.name, 'mmproj-LFM3-VL-2B-BF16.gguf');
    expect(vl.id, isNot(VisionCheckpointSpec.lfm3VlQ8.id));
    expect(ExampleModel.lfm3Vl.directoryName, 'LFM3-VL-2B-QAD-Q4');
    expect(find.textContaining('1.07 GB download'), findsOneWidget);
    expect(find.textContaining(ModelSpec.lfm3Q4.license), findsNWidgets(3));
    expect(find.text(AudioGgufSpec.lfm3AudioQ4.name), findsOneWidget);
    expect(find.textContaining('1.23 GB download'), findsOneWidget);
    expect(find.byType(ChatPage), findsNothing);
    expect(find.byType(CircularProgressIndicator), findsNothing);
  });

  for (final model in ExampleModel.values) {
    testWidgets('select ${model.name} and return while startup is pending', (
      tester,
    ) async {
      final directory = Completer<String>();
      const channel = MethodChannel('plugins.flutter.io/path_provider');
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (_) => directory.future);
      addTearDown(() {
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockMethodCallHandler(channel, null);
      });

      await tester.pumpWidget(const DianoiaExampleApp());
      await tester.tap(find.text(model.name));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 350));
      expect(
        tester.widget<ModelChatPage>(find.byType(ModelChatPage)).model,
        model,
      );
      expect(find.byType(ChatPage), findsOneWidget);

      await tester.pageBack();
      await tester.pumpAndSettle();
      expect(find.text('Choose a model'), findsOneWidget);
      expect(find.byType(ChatPage), findsNothing);

      // Resolving the platform call after leaving must not start a worker
      // or notify a controller that has already been disposed.
      directory.complete('/unused-model-selection-test');
      await tester.pump();
      expect(tester.takeException(), isNull);
    });
  }

  testWidgets('a missing VL model offers a download with conversion storage', (
    tester,
  ) async {
    final chat = ChatController(model: ExampleModel.lfm3Vl)
      ..setup = Setup.needsModel
      ..spec = ExampleModel.lfm3Vl.downloadSpec;
    addTearDown(chat.dispose);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: ModelSetup(chat: chat)),
      ),
    );
    expect(find.text(ExampleModel.lfm3Vl.name), findsOneWidget);
    expect(find.textContaining('1.45 GB to download'), findsOneWidget);
    expect(find.textContaining('at least 5 GB'), findsOneWidget);
    expect(find.text('Download'), findsOneWidget);
  });

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
    await tester.pumpWidget(
      host(ChatMessage(ChatRole.assistant, '')..pending = true),
    );
    expect(find.byType(CircularProgressIndicator), findsOneWidget);
  });

  testWidgets(
    'image analysis shows stage progress and completion clears the spinner',
    (tester) async {
      final message = ChatMessage(ChatRole.assistant, '')
        ..pending = true
        ..progress = 'Analyzing image… 7/16';
      await tester.pumpWidget(host(message));
      expect(find.text('Analyzing image… 7/16'), findsOneWidget);
      expect(find.byType(CircularProgressIndicator), findsOneWidget);
      message.pending = false;
      await tester.pumpWidget(host(message));
      expect(find.byType(CircularProgressIndicator), findsNothing);
      expect(find.text('No reply text.'), findsOneWidget);
    },
  );

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
