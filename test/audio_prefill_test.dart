import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/synthetic_tokenizer.dart';
import 'support/vision_model_fixture.dart';

const _specials = [
  '<|audio|>',
  '<|audio_out|>',
  '<|audio_start|>',
  '<|mixed_start|>',
  '<|text_end|>',
];

Float32List _rows(int n) => Float32List.fromList([
  for (var i = 0; i < n * 32; i++) ((i * 17 + i ~/ 3) % 97 - 48) / 128,
]);

void main() {
  final supported = NativeKernels.isAvailable;
  late Directory temp;
  late Lfm3Model model;
  late Lfm3Tokenizer tokenizer;
  setUpAll(() async {
    if (!supported) return;
    temp = Directory.systemTemp.createTempSync('dianoia-audio-prefill-');
    // Reuse the existing tiny synthetic language bundle (no vision is loaded).
    await prepareVisionFixture(temp);
    final dir = '${temp.path}/backbone';
    writeSyntheticTokenizer(dir, additionalSpecials: _specials);
    tokenizer = Lfm3Tokenizer.load(dir);
    model = await Lfm3Model.load(
      dir,
      maxSeq: 256,
      maxChunk: 2,
      weights: WeightFormat.q8,
      q8CachePath: '$dir/lfm3-q8.cache',
    );
  });
  tearDownAll(() async {
    if (!supported) return;
    await model.close();
    temp.deleteSync(recursive: true);
  });
  group('incremental audio prefill', () {
    for (final chunk in [1, 2, 7, 13]) {
      test('$chunk-row chunks match a complete audio prompt', () async {
        final projected = _rows(13);
        final prompt = prepareAudioTranscriptionPrompt(
          tokenizer: tokenizer,
          audioTokens: 13,
          hiddenSize: 32,
          language: 'English',
        );
        await prompt.prefill(model, projected);
        final expected = Float32List.fromList(model.logits);
        final expectedHistory = model.history.toList();
        final input = await AudioTranscriptionPrefill.start(
          model: model,
          tokenizer: tokenizer,
          hiddenSize: 32,
          language: 'English',
          reservedTokens: 16,
        );
        expect(model.length, prompt.audioStart);
        for (var at = 0; at < 13; at += chunk) {
          final end = (at + chunk).clamp(0, 13);
          await input.addProjected(projected.sublist(at * 32, end * 32));
          expect(input.audioTokens, end);
          expect(model.length, prompt.audioStart + end);
        }
        await input.finish();
        expect(input.isFinished, isTrue);
        expect(input.promptTokens, expectedHistory.length);
        expect(model.history, expectedHistory);
        for (var i = 0; i < expected.length; i++) {
          expect(
            model.logits[i],
            closeTo(expected[i], 2e-5 * (1 + expected[i].abs())),
          );
        }
        expect(input.greedyToken(model.logits), prompt.greedyToken(expected));
        await expectLater(input.finish(), throwsStateError);
        await expectLater(input.addProjected(_rows(1)), throwsStateError);
        input.close();
      });
    }

    test(
      'copies rows before awaiting and rejects overlapping operations',
      () async {
        final prompt = prepareAudioTranscriptionPrompt(
          tokenizer: tokenizer,
          audioTokens: 3,
          hiddenSize: 32,
        );
        await prompt.prefill(model, _rows(3));
        final expected = Float32List.fromList(model.logits);
        final input = await AudioTranscriptionPrefill.start(
          model: model,
          tokenizer: tokenizer,
          hiddenSize: 32,
        );
        final rows = _rows(3);
        final pending = input.addProjected(rows);
        rows.fillRange(0, rows.length, 0);
        await expectLater(input.addProjected(_rows(1)), throwsStateError);
        await expectLater(input.finish(), throwsStateError);
        await pending;
        await input.finish();
        for (var i = 0; i < expected.length; i++) {
          expect(
            model.logits[i],
            closeTo(expected[i], 2e-5 * (1 + expected[i].abs())),
          );
        }
      },
    );

    test(
      'invalid rows and overflow preserve state and permit a valid retry',
      () async {
        final input = await AudioTranscriptionPrefill.start(
          model: model,
          tokenizer: tokenizer,
          hiddenSize: 32,
        );
        final header = model.history.toList();
        for (final rows in [
          Float32List(1),
          _rows(1)..[0] = double.nan,
          _rows(300),
        ]) {
          await expectLater(input.addProjected(rows), throwsArgumentError);
          expect(input.audioTokens, 0);
          expect(model.history, header);
        }
        await input.addProjected(Float32List(0));
        await expectLater(input.finish(), throwsStateError);
        await input.addProjected(_rows(1));
        await input.finish();
        expect(input.audioTokens, 1);
        expect(input.isFinished, isTrue);
      },
    );

    test('validates start before resetting the caller-owned model', () async {
      await model.reset();
      await model.forward([10]);
      for (final config in [(64, 16), (32, -1), (32, 256)]) {
        await expectLater(
          AudioTranscriptionPrefill.start(
            model: model,
            tokenizer: tokenizer,
            hiddenSize: config.$1,
            reservedTokens: config.$2,
          ),
          throwsArgumentError,
        );
        expect(model.history, [10]);
      }
    });

    test(
      'close during a pending call forbids continued input without closing the model',
      () async {
        final input = await AudioTranscriptionPrefill.start(
          model: model,
          tokenizer: tokenizer,
          hiddenSize: 32,
        );
        final pending = input.addProjected(_rows(1));
        input.close();
        await expectLater(pending, throwsStateError);
        await expectLater(input.finish(), throwsStateError);
        expect(model.isClosed, isFalse);
        final next = await AudioTranscriptionPrefill.start(
          model: model,
          tokenizer: tokenizer,
          hiddenSize: 32,
        );
        await next.addProjected(_rows(1));
        await next.finish();
      },
    );

    test('detects externally changed language state', () async {
      final input = await AudioTranscriptionPrefill.start(
        model: model,
        tokenizer: tokenizer,
        hiddenSize: 32,
      );
      await model.reset();
      await expectLater(input.addProjected(_rows(1)), throwsStateError);
      await expectLater(input.finish(), throwsStateError);
    });
  }, skip: supported ? false : 'native kernels unavailable');
}
