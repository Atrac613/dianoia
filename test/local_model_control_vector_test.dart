// A control vector on a LocalModel (lab-only): conversations re-prefill
// under the new vector and reply as a model steered from the start would.
// Runs on the synthetic backbone of the vision fixture, on the CPU kernels.
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/vision_model_fixture.dart';

void main() {
  group('LocalModel control vector', () {
    late Directory root;
    late String backbone;
    late ControlVector vector;
    final opened = <LocalModel>[];

    setUpAll(() async {
      root = Directory.systemTemp.createTempSync('dianoia-local-cv-');
      await prepareVisionFixture(root);
      backbone = '${root.path}/backbone';
      final c = TextModelConfig.load(backbone);
      final random = math.Random(7);
      vector = ControlVector(
        {
          for (var l = 0; l < c.numLayers; l++)
            l: Float32List.fromList([
              for (var j = 0; j < c.hiddenSize; j++)
                random.nextDouble() * 2 - 1,
            ]),
        },
        hiddenSize: c.hiddenSize,
        scale: 3,
      );
    });

    tearDown(() async {
      for (final m in opened) {
        await m.close();
      }
      opened.clear();
    });

    tearDownAll(() => root.deleteSync(recursive: true));

    Future<LocalModel> load() async {
      final m = await LocalModel.load(
        backbone,
        backend: InferenceBackend.cpu,
        contextLength: 256,
        threads: 2,
      );
      opened.add(m);
      return m;
    }

    ChatSession chat(LocalModel m) =>
        m.chat(system: 'S', maxNewTokens: 8)..keepPromptLogits = true;

    final hi = {'role': 'user', 'content': 'hi'};

    /// The reply of a fresh model steered by [v] to [history] then [message].
    Future<(String, Float32List)> fresh(
      ControlVector? v,
      List<Map<String, Object?>> history,
      Map<String, Object?> message,
    ) async {
      final m = await load();
      await m.setControlVector(v);
      final s = chat(m)..messages.addAll(history);
      final turn = await s.send(message);
      return (turn.text, s.promptLogits!);
    }

    test('a vector set between replies applies to the next one', () async {
      final m = await load();
      final s = chat(m);
      final plain = await s.send(hi);
      final plainLogits = s.promptLogits!;

      await m.setControlVector(vector);
      expect(m.controlVector, same(vector));
      s.clear();
      final steered = await s.send(hi);
      final (text, logits) = await fresh(vector, const [], hi);
      expect(steered.text, text);
      expect(s.promptLogits, orderedEquals(logits));
      expect(logits, isNot(orderedEquals(plainLogits)));

      await m.setControlVector(null);
      expect(m.controlVector, isNull);
      s.clear();
      final again = await s.send(hi);
      expect(again.text, plain.text);
      expect(s.promptLogits, orderedEquals(plainLogits));
    });

    test('a vector set mid-conversation replays the history', () async {
      final m = await load();
      final s = chat(m);
      await s.send(hi);
      await s.send({'role': 'user', 'content': 'and then?'});
      final history = [...s.messages];
      await m.setControlVector(vector);
      final third = {'role': 'user', 'content': 'why?'};
      final turn = await s.send(third);
      expect(turn.reusedTokens, 0);
      final (text, logits) = await fresh(vector, history, third);
      expect(turn.text, text);
      expect(s.promptLogits, orderedEquals(logits));
    });

    test('a saved state is not reused under another vector', () async {
      final path = '${root.path}/warm.state';
      final m = await load();
      expect(await chat(m).warmUp(statePath: path), isFalse);
      expect(await chat(m).warmUp(statePath: path), isTrue);
      await m.setControlVector(vector);
      final s = chat(m);
      expect(await s.warmUp(statePath: path), isFalse);
      final turn = await s.send(hi);
      final (text, logits) = await fresh(vector, const [], hi);
      expect(turn.text, text);
      expect(s.promptLogits, orderedEquals(logits));
      // The file now holds the steered state, which unsteered refuses.
      await m.setControlVector(null);
      expect(await chat(m).warmUp(statePath: path), isFalse);
    });

    test('changing the vector during a reply fails cleanly', () async {
      final m = await load();
      final s = chat(m);
      Object? error;
      final turn = await s.send(
        hi,
        onText: (_) async {
          try {
            await m.setControlVector(vector);
          } on StateError catch (e) {
            error ??= e;
          }
          return true;
        },
      );
      expect(error, isA<StateError>());
      expect(m.controlVector, isNull);
      final (text, _) = await fresh(null, const [], hi);
      expect(turn.text, text);
      expect(s.messages, hasLength(2));
    });

    test('an equal vector changes nothing', () async {
      final m = await load();
      await m.setControlVector(vector);
      final s = chat(m);
      await s.send(hi);
      await m.setControlVector(vector.withSettings());
      final next = await s.send({'role': 'user', 'content': 'more'});
      expect(next.reusedTokens, greaterThan(0));
    });

    group('LocalVisionModel', () {
      Future<LocalVisionModel> loadVision() => LocalVisionModel.load(
        '${root.path}/model',
        backboneDir: backbone,
        backend: InferenceBackend.cpu,
        contextLength: 256,
        threads: 2,
      );

      /// A turn's text and prompt logits.
      Future<(String, Float32List)> turn(VisionChatSession s) async {
        Float32List? first;
        final t = await s.send(
          'Hello.',
          onLogits: (l) => first ??= Float32List.fromList(l),
        );
        return (t.text, first!);
      }

      test('a vector set between turns applies to the next one', () async {
        final m = await loadVision();
        final f = await loadVision();
        try {
          final s = m.chat(maxNewTokens: 4);
          final (_, plainLogits) = await turn(s);
          await m.setControlVector(vector);
          expect(m.controlVector, same(vector));
          s.clear();
          final (text, logits) = await turn(s);
          await f.setControlVector(vector);
          final (freshText, freshLogits) = await turn(f.chat(maxNewTokens: 4));
          expect(text, freshText);
          expect(logits, orderedEquals(freshLogits));
          expect(logits, isNot(orderedEquals(plainLogits)));
          await m.setControlVector(null);
          s.clear();
          expect((await turn(s)).$2, orderedEquals(plainLogits));
        } finally {
          await m.close();
          await f.close();
        }
      });

      test('changing the vector during a turn fails cleanly', () async {
        final m = await loadVision();
        try {
          Object? error;
          await m
              .chat(maxNewTokens: 4)
              .send(
                'Hello.',
                onText: (_) async {
                  try {
                    await m.setControlVector(vector);
                  } on StateError catch (e) {
                    error ??= e;
                  }
                  return true;
                },
              );
          expect(error, isA<StateError>());
          expect(m.controlVector, isNull);
        } finally {
          await m.close();
        }
      });
    });
  }, skip: NativeKernels.isAvailable ? false : 'requires native kernels');
}
