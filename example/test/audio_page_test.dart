import 'package:dianoia_example/src/audio_controller.dart';
import 'package:dianoia_example/src/audio_backend.dart';
import 'package:dianoia_example/src/audio_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:dianoia/dianoia.dart';

import 'dart:async';
import 'dart:typed_data';

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
  testWidgets('live recording has no 30-second wall-clock timer', (
    tester,
  ) async {
    await audio.load('/prepared');
    audio.setTask(AudioTask.liveTranscription);
    await show(tester);
    await tester.ensureVisible(find.text('Start live transcription'));
    await tester.tap(find.text('Start live transcription'));
    await tester.pump();
    await tester.pump(const Duration(seconds: 31));
    expect(audio.phase, AudioPhase.recording);
    final cancelled = audio.cancel();
    await tester.pump();
    await tester.runAsync(() => Future<void>.delayed(Duration.zero));
    await tester.pumpAndSettle();
    await tester.runAsync(() => cancelled);
    expect(audio.phase, AudioPhase.ready);
  });

  testWidgets('live mode displays text while recording and finalizes on stop', (
    tester,
  ) async {
    final backend = TestAudioBackend(), mic = TestMicrophone();
    final controller = AudioController(
      backend: backend,
      microphone: mic,
      player: TestSpeechPlayer(),
    );
    addTearDown(controller.dispose);
    await controller.load('/prepared');
    await tester.pumpWidget(
      MaterialApp(home: AudioTranscriptionPage(controller: controller)),
    );
    await tester.tap(find.text('Transcription and TTS'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Live transcription (English)').last);
    await tester.pumpAndSettle();
    expect(
      find.textContaining('Japanese is available after recording'),
      findsOneWidget,
    );
    await tester.ensureVisible(find.text('Start live transcription'));
    await tester.tap(find.text('Start live transcription'));
    await tester.pump();
    backend.liveText!('The quick');
    await tester.pump();
    expect(controller.phase, AudioPhase.recording);
    await tester.scrollUntilVisible(
      find.text('The quick'),
      120,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('The quick'), findsOneWidget);
    mic.source!.add(Uint8List(32));
    await tester.pump();
    await tester.ensureVisible(find.text('Stop live transcription'));
    await tester.tap(find.text('Stop live transcription'));
    await tester.pump();
    await tester.runAsync(() => Future<void>.delayed(Duration.zero));
    await tester.pumpAndSettle();
    expect(controller.phase, AudioPhase.ready);
    expect(controller.transcript, 'Recognized speech.');
  });

  testWidgets(
    'background discards a generating voice reply without automatic resume',
    (tester) async {
      final backend = TestAudioBackend()
        ..answering = Completer<AudioVoiceReply>();
      final mic = TestMicrophone(), player = TestSpeechPlayer();
      final controller = AudioController(
        backend: backend,
        microphone: mic,
        player: player,
      );
      addTearDown(controller.dispose);
      await controller.load('/prepared');
      controller.setTask(AudioTask.conversation);
      await tester.pumpWidget(
        MaterialApp(home: AudioTranscriptionPage(controller: controller)),
      );
      await tester.runAsync(controller.record);
      mic.source!.add(Uint8List(32));
      await tester.pump();
      late Future<void> stopping;
      await tester.runAsync(() async {
        stopping = controller.stop();
        await Future<void>.delayed(Duration.zero);
      });
      await tester.pump();
      expect(controller.phase, AudioPhase.replying);
      backend.onReplyText!('Partial reply');
      tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.inactive);
      await tester.runAsync(() => Future<void>.delayed(Duration.zero));
      await tester.pumpAndSettle();
      backend.answering!.complete(
        AudioVoiceReply('Late reply', Uint8List(44), 1),
      );
      await tester.pump();
      await tester.runAsync(() => stopping);
      expect(controller.partialReply, isEmpty);
      expect(controller.voiceTurns, isEmpty);
      expect(backend.acceptedReplies, 0);
      expect(player.played, isEmpty);
      tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.resumed);
      await tester.pump();
      expect(mic.starts, 1);
      expect(controller.phase, AudioPhase.ready);
    },
  );

  testWidgets(
    'English voice mode records a question, displays the reply and clears context',
    (tester) async {
      final backend = TestAudioBackend(),
          mic = TestMicrophone(),
          player = TestSpeechPlayer();
      player.completing = Completer<void>();
      final controller = AudioController(
        backend: backend,
        microphone: mic,
        player: player,
      );
      addTearDown(controller.dispose);
      await controller.load('/prepared');
      await tester.pumpWidget(
        MaterialApp(home: AudioTranscriptionPage(controller: controller)),
      );
      await tester.tap(find.text('Transcription and TTS'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Voice conversation (English)').last);
      await tester.pumpAndSettle();
      expect(find.textContaining('supports English only'), findsOneWidget);
      await tester.ensureVisible(find.text('Record question'));
      await tester.tap(find.text('Record question'));
      await tester.pump();
      expect(backend.conversationInput, isTrue);
      mic.source!.add(Uint8List(32));
      await tester.pump();
      await tester.ensureVisible(find.text('Stop and reply'));
      await tester.tap(find.text('Stop and reply'));
      await tester.pump();
      await tester.runAsync(() => Future<void>.delayed(Duration.zero));
      await tester.pumpAndSettle();
      expect(controller.voiceTurns, hasLength(1));
      expect(controller.phase, AudioPhase.playing, reason: controller.error);
      await tester.scrollUntilVisible(
        find.text('Four.'),
        160,
        scrollable: find.byType(Scrollable).first,
      );
      expect(find.text('Four.'), findsOneWidget);
      await tester.ensureVisible(find.text('Stop reply'));
      await tester.tap(find.text('Stop reply'));
      await tester.pumpAndSettle();
      await tester.ensureVisible(find.text('New conversation'));
      await tester.tap(find.text('New conversation'));
      await tester.pumpAndSettle();
      expect(controller.voiceTurns, isEmpty);
      expect(backend.resets, 1);
    },
  );

  testWidgets(
    'download offer exposes progress and cancels without loading a late result',
    (tester) async {
      final backend = TestAudioBackend()..installation = Completer<String>();
      final controller = AudioController(
        backend: backend,
        microphone: TestMicrophone(),
        player: TestSpeechPlayer(),
      );
      addTearDown(controller.dispose);
      await controller.discover('/store');
      await tester.pumpWidget(
        MaterialApp(home: AudioTranscriptionPage(controller: controller)),
      );
      expect(find.textContaining('1.23 GB to download'), findsOneWidget);
      expect(find.text('License: LFM Open License v1.0'), findsOneWidget);
      expect(backend.installs, 0);
      await tester.tap(find.text('Download'));
      await tester.pump();
      backend.progress!(
        const InstallProgress(
          InstallStage.downloading,
          bytes: 10000000,
          totalBytes: 20000000,
        ),
      );
      await tester.pump();
      expect(find.text('0.01 of 0.02 GB'), findsOneWidget);
      await tester.tap(find.text('Cancel download'));
      await tester.pumpAndSettle();
      backend.installation!.complete('/late');
      await tester.pump();
      expect(backend.loads, 0);
      expect(controller.phase, AudioPhase.needsModel);
    },
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
