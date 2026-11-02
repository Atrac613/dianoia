import 'package:dianoia_example/src/audio_controller.dart';
import 'package:dianoia_example/src/audio_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'dart:async';

import 'audio_controller_test.dart'
    show TestAudioBackend, TestMicrophone, TestSpeechPlayer;

void main() {
  late AudioController audio;
  setUp(() {
    audio = AudioController(
      backend: TestAudioBackend(),
      microphone: TestMicrophone(),
      player: TestSpeechPlayer(),
    );
  });
  tearDown(() async {
    await audio.close();
    audio.dispose();
  });
  Future<void> show(WidgetTester tester) => tester.pumpWidget(
    MaterialApp(
      home: Scaffold(
        body: AudioTranscriptionView(
          controller: audio,
          directory: '/models/audio',
          onChoose: () {},
        ),
      ),
    ),
  );

  testWidgets('speech composer generates only on tap and stops playback', (
    tester,
  ) async {
    final backend = TestAudioBackend(), player = TestSpeechPlayer();
    player.completing = Completer<void>();
    final controller = AudioController(
      backend: backend,
      microphone: TestMicrophone(),
      player: player,
    );
    addTearDown(controller.dispose);
    await controller.load('/prepared');
    await tester.pumpWidget(
      MaterialApp(home: AudioTranscriptionPage(controller: controller)),
    );
    expect(backend.utterances, isEmpty);
    await tester.ensureVisible(find.byType(TextField));
    await tester.enterText(find.byType(TextField), 'こんにちは。');
    await tester.ensureVisible(find.text('Generate and play'));
    await tester.tap(find.text('Generate and play'));
    await tester.pump();
    expect(backend.utterances, [('こんにちは。', 'ja', 'female')]);
    expect(controller.phase, AudioPhase.playing);
    await tester.ensureVisible(find.text('Stop speech'));
    await tester.tap(find.text('Stop speech'));
    await tester.pumpAndSettle();
    expect(controller.phase, AudioPhase.ready);
    expect(player.stops, 1);
  });

  testWidgets('background stops speech and resume does not restart it', (
    tester,
  ) async {
    final backend = TestAudioBackend(), player = TestSpeechPlayer();
    player.completing = Completer<void>();
    final controller = AudioController(
      backend: backend,
      microphone: TestMicrophone(),
      player: player,
    );
    addTearDown(controller.dispose);
    await controller.load('/prepared');
    await tester.pumpWidget(
      MaterialApp(home: AudioTranscriptionPage(controller: controller)),
    );
    final speaking = controller.speak('Hello');
    await tester.pump();
    expect(controller.phase, AudioPhase.playing);
    tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.inactive);
    await tester.pumpAndSettle();
    await speaking;
    expect(controller.phase, AudioPhase.ready);
    tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.resumed);
    await tester.pump();
    expect(player.played, hasLength(1));
    await tester.pumpWidget(const MaterialApp(home: Text('Left')));
    await tester.pump();
    await controller.close();
    expect(player.disposals, 1);
  });

  testWidgets('local model setup never requests microphone permission', (
    tester,
  ) async {
    await show(tester);
    expect(find.text('Choose model folder'), findsOneWidget);
    expect(
      tester
          .widget<FilledButton>(find.widgetWithText(FilledButton, 'Record'))
          .onPressed,
      isNull,
    );
    expect(find.textContaining('not saved'), findsOneWidget);
  });

  testWidgets('record stop and cancel controls reflect capture state', (
    tester,
  ) async {
    await audio.load('/prepared');
    await show(tester);
    await tester.tap(find.text('Record'));
    await tester.pump();
    expect(find.text('Stop and transcribe'), findsOneWidget);
    expect(find.text('Cancel recording'), findsOneWidget);
    await tester.tap(find.text('Cancel recording'));
    await tester.pump();
    // Subscription cancellation may complete outside Flutter's fake clock.
    await tester.runAsync(() => Future<void>.delayed(Duration.zero));
    await tester.pumpAndSettle();
    expect(find.text('Record'), findsOneWidget);
    expect(audio.phase, AudioPhase.ready);
  });

  testWidgets('recording has a 30 second wall-clock stop even without input', (
    tester,
  ) async {
    await audio.load('/prepared');
    await show(tester);
    await tester.tap(find.text('Record'));
    await tester.pump();
    await tester.pump(const Duration(seconds: 30));
    await tester.pump();
    // Subscription cancellation may complete outside Flutter's fake clock.
    await tester.runAsync(() => Future<void>.delayed(Duration.zero));
    await tester.pumpAndSettle();
    expect(audio.active, isFalse);
    expect(audio.error, contains('No audio'));
  });

  testWidgets(
    'background and route exit discard capture without automatic resume',
    (tester) async {
      final mic = TestMicrophone(), backend = TestAudioBackend();
      final controller = AudioController(backend: backend, microphone: mic);
      addTearDown(controller.dispose);
      await controller.load('/prepared');
      await tester.pumpWidget(
        MaterialApp(home: AudioTranscriptionPage(controller: controller)),
      );
      await tester.tap(find.text('Record'));
      await tester.pump();
      expect(mic.starts, 1);
      tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.inactive);
      tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.hidden);
      tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.paused);
      await tester.runAsync(() => Future<void>.delayed(Duration.zero));
      await tester.pumpAndSettle();
      expect(controller.active, isFalse);
      expect(controller.transcript, isEmpty);
      tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.hidden);
      tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.inactive);
      tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.resumed);
      await tester.pump();
      expect(mic.starts, 1);
      await tester.tap(find.text('Record'));
      await tester.pump();
      expect(mic.starts, 2);
      await tester.pumpWidget(const MaterialApp(home: Text('Left')));
      await tester.runAsync(() => Future<void>.delayed(Duration.zero));
      await tester.pump();
      await controller.close();
      expect(backend.closes, 1);
      expect(mic.stops, greaterThanOrEqualTo(2));
    },
  );
}
