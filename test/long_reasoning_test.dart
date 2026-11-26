// Thinking budgets, loop guards and the presence penalty in a ChatSession.
// Runs on the synthetic backbone of the vision fixture, on the native CPU
// kernels.
import 'dart:io';

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
  }, skip: NativeKernels.isAvailable ? false : 'requires native kernels');
}
