// Thinking budgets, loop guards and the presence penalty in a ChatSession.
// Runs on the synthetic backbone of the vision fixture, on the native CPU
// kernels.
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/vision_model_fixture.dart';

void main() {
  group('long reasoning', () {
    late Directory root;
    late TextModel model;
    late BpeTokenizer tokenizer;

    setUpAll(() async {
      root = Directory.systemTemp.createTempSync('dianoia-reasoning-');
      await prepareVisionFixture(root);
      final backbone = '${root.path}/backbone';
      model = await TextModel.load(
        backbone,
        maxSeq: 256,
        maxChunk: 4,
        weights: WeightFormat.q8,
        q8CachePath: '$backbone/lfm3-q8.cache',
      );
      tokenizer = BpeTokenizer.load(backbone);
    });

    tearDownAll(() {
      model.close();
      root.deleteSync(recursive: true);
    });

    setUp(() => model.reset());

    const cutoff = '\n</think>\n';
    // Three of the same token in a row.
    const triple = RepetitionGuard(maxPeriod: 1, minRepeats: 3, minTokens: 3);
    final hi = {'role': 'user', 'content': 'hi'};
    // Every token but [token] suppressed: a reply that repeats [token]
    // until something stops it (end tokens are never suppressed).
    Set<int> onlyToken(int token) => {
      for (var i = 0; i < model.config.vocabSize; i++)
        if (i != token) i,
    };
    // 'A': with only it allowed, the synthetic model keeps choosing it over
    // its end token, with or without a cutoff before it. Many tokens do;
    // pick another if the fixture changes and this one stops.
    const repeated = 65;

    test('a thinking budget closes the reasoning with the cutoff', () async {
      final session = ChatSession(
        model,
        tokenizer,
        maxNew: 24,
        enableThinking: true,
        thinkingBudget: 5,
        thinkingCutoff: cutoff,
      );
      final turn = await session.send(hi);
      final ids = tokenizer.encode(turn.text);
      final cut = tokenizer.encode(cutoff);
      expect(turn.thinkingCut, isTrue);
      expect(turn.stoppedOnRepetition, isFalse);
      expect(ids.sublist(5, 5 + cut.length), cut);
      expect(turn.reply.reasoning, isNotNull);
      expect(turn.generatedTokens, 24);
    });

    test('a loop in the reasoning closes it like the budget', () async {
      final session = ChatSession(
        model,
        tokenizer,
        maxNew: 24,
        enableThinking: true,
        thinkingCutoff: cutoff,
        repetitionGuard: triple,
      );
      final turn = await session.send(
        hi,
        suppressedTokens: onlyToken(repeated),
      );
      final cut = tokenizer.encode(cutoff);
      expect(turn.thinkingCut, isTrue);
      // The answer repeats the same token, which ends the reply.
      expect(turn.stoppedOnRepetition, isTrue);
      expect(tokenizer.encode(turn.text), [
        ...List.filled(3, repeated),
        ...cut,
        ...List.filled(3, repeated),
      ]);
    });

    test('a loop in the answer ends the reply', () async {
      final session = ChatSession(
        model,
        tokenizer,
        maxNew: 120,
        repetitionGuard: const RepetitionGuard(),
      );
      final turn = await session.send(
        hi,
        suppressedTokens: onlyToken(repeated),
      );
      expect(turn.stoppedOnRepetition, isTrue);
      expect(turn.thinkingCut, isFalse);
      // The default guard waits for 32 repeated tokens.
      expect(tokenizer.encode(turn.text), List.filled(32, repeated));
    });

    test('a cut with no room left in the context is not reported', () async {
      final session = ChatSession(
        model,
        tokenizer,
        maxNew: 24,
        enableThinking: true,
        thinkingBudget: 1,
        thinkingCutoff: cutoff,
      );
      final empty = session.promptIds([
        {'role': 'user', 'content': ''},
      ]).length;
      // A prompt that leaves the cutoff no room after one reasoning token.
      final long = {
        'role': 'user',
        'content': 'a' * (model.maxSeq - 3 - empty),
      };
      expect(session.promptIds([long]).length, model.maxSeq - 3);
      expect(tokenizer.encode(cutoff).length, greaterThanOrEqualTo(2));
      final turn = await session.send(
        long,
        suppressedTokens: onlyToken(repeated),
      );
      expect(turn.thinkingCut, isFalse);
      expect(turn.generatedTokens, 1);
      expect(turn.text, isNot(contains('</think>')));
    });

    test('a budget must leave room below maxNew', () {
      for (final budget in [0, 24, 30]) {
        expect(
          () => ChatSession(
            model,
            tokenizer,
            maxNew: 24,
            enableThinking: true,
            thinkingBudget: budget,
          ),
          throwsArgumentError,
        );
      }
    });

    test('without a guard or budget the reply runs on', () async {
      final session = ChatSession(model, tokenizer, maxNew: 24);
      final turn = await session.send(hi);
      expect(turn.generatedTokens, 24);
      expect(turn.thinkingCut, isFalse);
      expect(turn.stoppedOnRepetition, isFalse);
    });

    test('a presence penalty keeps the reply from repeating tokens', () async {
      final session = ChatSession(
        model,
        tokenizer,
        maxNew: 40,
        sampling: const SamplingParams(presencePenalty: 1000),
        repetitionGuard: triple,
      );
      final turn = await session.send(hi);
      final ids = tokenizer.encode(turn.text);
      expect(turn.generatedTokens, 40);
      expect(turn.stoppedOnRepetition, isFalse);
      expect(ids.toSet(), hasLength(ids.length));
    });
    group('checkLine', () {
      // Only 'B' and newline allowed: greedy replies of several short lines.
      Set<int> lines() => {
        for (var i = 0; i < model.config.vocabSize; i++)
          if (i != 66 && i != 10) i,
      };

      test('rewinding a line restores the state at its start', () async {
        final logits = <Float32List>[];
        var calls = 0;
        int? lineStart;
        int? rewound;
        final session = ChatSession(model, tokenizer, maxNew: 30);
        final turn = await session.send(
          hi,
          suppressedTokens: lines(),
          onLogits: (l) => logits.add(Float32List.fromList(l)),
          // Accepts the first line and rejects the second once.
          checkLine: (reply) {
            calls++;
            if (calls == 1) lineStart = logits.length - 1;
            if (calls == 2) rewound = logits.length;
            return calls != 2;
          },
        );
        expect(turn.rejectedLines, 1);
        expect(turn.stoppedOnCheck, isFalse);
        // The logits right after the rewind are those the line began with.
        expect(logits[rewound!], orderedEquals(logits[lineStart!]));
        expect(model.history.length, turn.promptTokens + turn.generatedTokens);
      });

      test('a line that fails with no retries left ends the reply', () async {
        final session = ChatSession(model, tokenizer, maxNew: 30);
        final seen = <String>[];
        final turn = await session.send(
          hi,
          suppressedTokens: lines(),
          checkLine: (_) => false,
          lineRetries: 0,
          onText: (text) {
            seen.add(text);
            return true;
          },
        );
        expect(turn.stoppedOnCheck, isTrue);
        expect(turn.rejectedLines, 1);
        expect(turn.text, isEmpty);
        expect(seen.last, isEmpty);
        expect(model.history.length, turn.promptTokens);
      });

      test('sampled retries keep only lines that pass', () async {
        final session = ChatSession(
          model,
          tokenizer,
          maxNew: 40,
          sampling: const SamplingParams(temperature: 1, seed: 3),
        );
        bool short(String reply) =>
            reply.trimRight().split('\n').every((l) => l.length <= 1);
        final turn = await session.send(
          hi,
          suppressedTokens: lines(),
          checkLine: short,
          lineRetries: 20,
        );
        expect(turn.rejectedLines, greaterThan(0));
        final complete = turn.text.substring(
          0,
          turn.text.lastIndexOf('\n') + 1,
        );
        expect(short(complete), isTrue);
      });

      test('a negative retry count is refused', () {
        expect(
          () => ChatSession(model, tokenizer).send(hi, lineRetries: -1),
          throwsArgumentError,
        );
      });
    });
  }, skip: NativeKernels.isAvailable ? false : 'requires native kernels');
}
