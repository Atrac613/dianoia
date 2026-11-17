// The pure parts of the control-vector evaluation, and its teacher-forced
// likelihood against stepping token by token on the synthetic backbone of
// the vision fixture (native CPU kernels).
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/vision_model_fixture.dart';

void main() {
  group('readEvalRows', () {
    late Directory dir;
    setUp(() => dir = Directory.systemTemp.createTempSync('dianoia-eval-'));
    tearDown(() => dir.deleteSync(recursive: true));

    test('reads text files a prompt per line', () {
      final path = '${dir.path}/p.txt';
      File(path).writeAsStringSync('  first  \n\nsecond\n   \n');
      final rows = readEvalRows(path);
      expect([for (final r in rows) r.prompt], ['first', 'second']);
      expect(rows.every((r) => r.reference == null), isTrue);
    });

    test('reads JSON lines with multi-line prompts and references', () {
      final path = '${dir.path}/p.jsonl';
      File(path).writeAsStringSync(
        '{"prompt": "a\\nb", "reference": "c"}\n\n{"prompt": "d", "x": 1}\n',
      );
      final rows = readEvalRows(path);
      expect([for (final r in rows) r.prompt], ['a\nb', 'd']);
      expect([for (final r in rows) r.reference], ['c', null]);
    });

    test('rejects malformed JSON lines with their line number', () {
      final path = '${dir.path}/p.jsonl';
      for (final bad in [
        '{"prompt": 1}',
        '["prompt"]',
        '{"prompt": "a", "reference": 2}',
        '{"prompt": ',
      ]) {
        File(path).writeAsStringSync('{"prompt": "ok"}\n$bad\n');
        expect(
          () => readEvalRows(path),
          throwsA(
            isA<FormatException>().having(
              (e) => e.message,
              'message',
              contains(':2'),
            ),
          ),
        );
      }
    });
  });

  test('wilsonInterval matches known values', () {
    final (l0, h0) = wilsonInterval(0, 10);
    expect(l0, 0);
    expect(h0, closeTo(0.2775, 1e-4));
    final (l5, h5) = wilsonInterval(5, 10);
    expect(l5, closeTo(0.2366, 1e-4));
    expect(h5, closeTo(0.7634, 1e-4));
    final (l9, h9) = wilsonInterval(90, 100);
    expect(l9, closeTo(0.8256, 1e-4));
    expect(h9, closeTo(0.9448, 1e-4));
    expect(() => wilsonInterval(3, 2), throwsArgumentError);
    expect(() => wilsonInterval(0, 0), throwsArgumentError);
  });

  test('opensWithRefusal takes both apostrophes and only openings', () {
    for (final yes in [
      "I can't help with that.",
      'I can’t help with that.',
      'I cannot do that',
      "  I'm sorry, but",
      'I’m sorry, but',
      'I am sorry',
      "I won't",
      'I won’t',
    ]) {
      expect(opensWithRefusal(yes), isTrue, reason: yes);
    }
    for (final no in [
      'Sure, here is how.',
      "Here's why I can't stop smiling",
      'I can help with that.',
      'i cannot',
      '',
    ]) {
      expect(opensWithRefusal(no), isFalse, reason: no);
    }
  });

  test('bootstrapMean is deterministic and brackets the mean', () {
    final values = [for (var i = 0; i < 50; i++) (i % 7) - 3.0];
    final (m, low, high) = bootstrapMean(values);
    expect(m, closeTo(values.reduce((a, b) => a + b) / 50, 1e-12));
    expect(low, lessThanOrEqualTo(m));
    expect(high, greaterThanOrEqualTo(m));
    expect(bootstrapMean(values), (m, low, high));
    expect(bootstrapMean([2, 2, 2]), (2.0, 2.0, 2.0));
  });

  test('logProbability and klDivergence', () {
    final p = Float32List.fromList([0, math.log(3)]);
    expect(logProbability(p, 1), closeTo(math.log(0.75), 1e-6));
    expect(klDivergence(p, p), closeTo(0, 1e-12));
    final q = Float32List.fromList([0, 0]);
    expect(
      klDivergence(p, q),
      closeTo(0.25 * math.log(0.5) + 0.75 * math.log(1.5), 1e-6),
    );
  });

  group('referenceNll', () {
    late Directory root;
    late Lfm3Model model;
    late ControlVectorEval eval;

    setUpAll(() async {
      root = Directory.systemTemp.createTempSync('dianoia-eval-model-');
      await prepareVisionFixture(root);
      final backbone = '${root.path}/backbone';
      model = await Lfm3Model.load(
        backbone,
        maxSeq: 128,
        maxChunk: 4,
        weights: WeightFormat.q8,
        q8CachePath: '$backbone/lfm3-q8.cache',
      );
      final c = model.config;
      final random = math.Random(3);
      eval = ControlVectorEval(
        model,
        Lfm3Tokenizer.load(backbone),
        ControlVector(
          {
            1: Float32List.fromList([
              for (var j = 0; j < c.hiddenSize; j++) random.nextDouble() - .5,
            ]),
          },
          hiddenSize: c.hiddenSize,
          mode: ControlMode.project,
        ),
      );
    });

    tearDownAll(() async {
      await model.close();
      root.deleteSync(recursive: true);
    });

    test('equals stepping through the reference', () async {
      const reference = [40, 50, 60, 70, 80, 90, 100];
      for (final v in [null, eval.vector]) {
        final nll = await eval.referenceNll('hi', reference, v);
        // The same tokens fed one at a time, each scored by the logits
        // before it.
        await model.reset();
        model.setControlVector(v);
        final prompt = Lfm3Tokenizer.load('${root.path}/backbone').encode(
          renderChat([
            {'role': 'user', 'content': 'hi'},
          ], enableThinking: false),
        );
        await model.forward(prompt);
        var total = 0.0;
        for (final token in reference) {
          total -= logProbability(model.logits, token);
          await model.step(token);
        }
        expect(nll, closeTo(total / reference.length, 1e-9));
      }
    });
  }, skip: NativeKernels.isAvailable ? false : 'requires native kernels');
}
