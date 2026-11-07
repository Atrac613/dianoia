import 'dart:io';
import 'dart:async';

import 'package:dianoia_example/src/live_translation.dart';
import 'package:dianoia_example/src/translation_worker.dart';
import 'package:flutter_test/flutter_test.dart';

import 'fake_translation.dart';

Future<void> settle() => Future<void>.delayed(Duration.zero);

void main() {
  group('SentenceSplitter', () {
    final splitter = SentenceSplitter(maxLength: 40);

    test('waits for the next word before ending a sentence', () {
      final (none, start) = splitter.take('Hello there.', 0);
      expect(none, isEmpty);
      expect(start, 0);
      final (sentences, consumed) = splitter.take('Hello there. How', 0);
      expect(sentences, ['Hello there.']);
      expect('Hello there. How'.substring(consumed).trim(), 'How');
    });

    test('ends at question marks, quotes and context breaks', () {
      const text = 'Is it "done?" Yes!\nno punctuation here\nNext';
      final (sentences, _) = splitter.take(text, 0);
      expect(sentences, ['Is it "done?"', 'Yes!', 'no punctuation here']);
    });

    test('continues from the offset it returned', () {
      const first = 'One. Two';
      final (_, consumed) = splitter.take(first, 0);
      final (sentences, _) = splitter.take('One. Two. Three', consumed);
      expect(sentences, ['Two.']);
    });

    test('cuts a long unpunctuated stretch at a finished word', () {
      const text =
          'so we kept talking about the release and the build and the tests';
      final (sentences, consumed) = splitter.take(text, 0);
      expect(sentences, isNotEmpty);
      expect(sentences.every((s) => s.length <= 40), isTrue);
      expect(text.substring(0, consumed).trim(), sentences.join(' '));
      expect(text[consumed], ' ');
    });

    test('flush takes the rest', () {
      final (sentences, consumed) = splitter.take(
        'Done. And the end',
        0,
        flush: true,
      );
      expect(sentences, ['Done.', 'And the end']);
      expect(consumed, 'Done. And the end'.length);
    });
  });

  group('LiveTranslator', () {
    late FakeTranslation backend;
    late LiveTranslator translator;
    setUp(() {
      backend = FakeTranslation();
      translator = LiveTranslator(backend);
    });
    tearDown(() => translator.dispose());

    test('translates each finished sentence once, in order', () async {
      await translator.prepare('/model/language');
      translator.update('Good morning. We');
      translator.update('Good morning. We shipped it. And');
      await settle();
      translator.update('Good morning. We shipped it. And', flush: true);
      while (translator.busy) {
        await settle();
      }
      expect(backend.requests.map((r) => r.$1), [
        'Good morning.',
        'We shipped it.',
        'And',
      ]);
      expect(translator.segments.map((s) => s.translation), [
        '[Japanese] Good morning.',
        '[Japanese] We shipped it.',
        '[Japanese] And',
      ]);
      expect(translator.segments.every((s) => s.done), isTrue);
      expect(translator.pending, 0);
      expect(translator.unfinished('Good morning. We shipped it. And'), '');
    });

    test('loads once per directory and retries after a failure', () async {
      backend.failure = StateError('no model');
      await expectLater(translator.prepare('/m'), throwsStateError);
      expect(translator.error, contains('no model'));
      backend.failure = null;
      await translator.prepare('/m');
      await translator.prepare('/m');
      expect(backend.loads, ['/m', '/m']);
      expect(translator.backendName, 'CPU');
      expect(translator.error, isNull);
    });

    test('reset drops queued sentences and a late translation', () async {
      backend.hold = Completer<void>();
      translator.update('First one. Second one. Third');
      await settle();
      expect(translator.busy, isTrue);
      await translator.reset();
      expect(backend.cancels, 1);
      expect(translator.segments, isEmpty);
      backend.hold = null;
      translator.update('New start. Next');
      while (translator.busy) {
        await settle();
      }
      expect(translator.segments.single.translation, '[Japanese] New start.');
      expect(backend.requests.map((r) => r.$1), ['First one.', 'New start.']);
    });

    test('the target cannot change while sentences wait', () async {
      backend.hold = Completer<void>();
      translator.update('One. Two');
      translator.setTarget('French');
      expect(translator.target, 'Japanese');
      backend.hold!.complete();
      while (translator.busy) {
        await settle();
      }
      translator.setTarget('French');
      translator.update('One. Two. Three', flush: true);
      while (translator.busy) {
        await settle();
      }
      expect(backend.requests, [
        ('One.', 'Japanese'),
        ('Two.', 'French'),
        ('Three', 'French'),
      ]);
    });
  });

  test('cleanTranslation keeps the answer after a stray reasoning block', () {
    expect(cleanTranslation('決めましょう。</think>決めましょう。'), '決めましょう。');
    expect(cleanTranslation(' こんにちは '), 'こんにちは');
  });

  final directory = Platform.environment['DIANOIA_AUDIO_EXAMPLE_DIR'];
  test(
    'the audio bundle language backbone translates on the CPU',
    () async {
      final backend = TranslationIsolateBackend();
      try {
        expect(await backend.load('$directory/language'), 'CPU');
        final updates = <String>[];
        final text = await backend.translate(
          'Could you send me the slides after the call?',
          'Japanese',
          onText: updates.add,
        );
        expect(text, contains('スライド'));
        expect(updates, isNotEmpty);
      } finally {
        await backend.close();
      }
    },
    skip: directory == null ? 'Set DIANOIA_AUDIO_EXAMPLE_DIR' : false,
    timeout: const Timeout(Duration(minutes: 2)),
  );
}
