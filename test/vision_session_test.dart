import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/dianoia.dart';
import 'package:test/test.dart';

import 'support/vision_model_fixture.dart';

ChatImage _image([int value = 200]) => ChatImage.rgb(
  Uint8List(8 * 8 * 3)..fillRange(0, 8 * 8 * 3, value),
  width: 8,
  height: 8,
);

void main() {
  test('RGB construction validates dimensions and byte count', () {
    expect(
      () => ChatImage.rgb(Uint8List(3), width: 0, height: 1),
      throwsArgumentError,
    );
    expect(
      () => ChatImage.rgb(Uint8List(3), width: 2, height: 1),
      throwsArgumentError,
    );
  });

  group('public image chat', () {
    late Directory root;
    late LocalVisionModel model;
    setUpAll(() async {
      root = Directory.systemTemp.createTempSync('dianoia_vision_session');
      await prepareVisionFixture(root);
    });
    tearDownAll(() => root.deleteSync(recursive: true));
    setUp(() async {
      model = await LocalVisionModel.load(
        '${root.path}/model',
        backboneDir: '${root.path}/backbone',
        backend: InferenceBackend.cpu,
        contextLength: 512,
        threads: 2,
      );
    });
    tearDown(() => model.close());

    test(
      'streams a visual turn, follows up, clears and protects history',
      () async {
        expect(model.backend, InferenceBackend.cpu);
        expect(model.contextLength, 512);
        final session = model.chat(maxNewTokens: 3);
        final seen = <String>[];
        final turn = await session.send(
          'Describe.',
          images: [_image()],
          onText: (text) {
            seen.add(text);
            return true;
          },
        );
        expect(turn.generatedTokens, 3);
        expect(turn.reusedTokens, 0);
        expect(seen, hasLength(3));
        expect(session.messages, hasLength(2));
        expect(session.messages.last['content'], turn.text);
        expect(() => session.messages.clear(), throwsUnsupportedError);
        expect(
          () => session.messages.first['content'] = 'changed',
          throwsUnsupportedError,
        );
        final followup = await session.send('Explain.');
        expect(followup.generatedTokens, 3);
        expect(followup.promptTokens, greaterThan(turn.promptTokens));
        expect(session.messages, hasLength(4));
        session.clear();
        expect(session.messages, isEmpty);
        expect((await session.send('Hello.')).generatedTokens, 3);
      },
    );

    test(
      'visual progress covers new images and skips retained features',
      () async {
        final session = model.chat(maxNewTokens: 1);
        final seen = <VisionProgress>[];
        await session.send(
          'Describe.',
          images: [_image(), _image(80)],
          onProgress: seen.add,
        );
        final encoding = seen
            .where((p) => p.phase == VisionPhase.encodingImage)
            .toList();
        expect(encoding.first.completed, 0);
        expect(encoding.last.completed, 10);
        expect(encoding.map((p) => p.total), everyElement(10));
        expect(seen[seen.length - 2].phase, VisionPhase.prefilling);
        expect(seen.last.phase, VisionPhase.generating);
        seen.clear();
        await session.send('Explain.', onProgress: seen.add);
        expect(seen.map((p) => p.phase), [
          VisionPhase.prefilling,
          VisionPhase.generating,
        ]);
      },
    );

    test('cancellation from progress leaves history intact', () async {
      final session = model.chat(maxNewTokens: 1);
      await expectLater(
        session.send(
          'Describe.',
          images: [_image()],
          onProgress: (p) {
            if (p.phase == VisionPhase.encodingImage && p.completed == 1) {
              session.cancel();
            }
          },
        ),
        throwsA(isA<VisionRequestCancelled>()),
      );
      expect(session.messages, isEmpty);
      expect((await session.send('Hello.')).generatedTokens, 1);
    });

    test(
      'RGB, attachment lists and message maps are owned before yielding',
      () async {
        final pixels = Uint8List(8 * 8 * 3)..fillRange(0, 8 * 8 * 3, 200);
        final owned = ChatImage.rgb(pixels, width: 8, height: 8);
        pixels.fillRange(0, pixels.length, 0);
        final a = model.chat(maxNewTokens: 2);
        final b = model.chat(maxNewTokens: 2);
        Float32List? expected;
        await a.send(
          'Describe.',
          images: [_image()],
          onLogits: (v) {
            expected ??= Float32List.fromList(v);
          },
        );
        Float32List? actual;
        final images = [owned];
        final message = <String, Object?>{
          'role': 'user',
          'content': [
            {'type': 'image'},
            {'type': 'text', 'text': 'Describe.'},
          ],
        };
        final pending = b.sendMessage(
          message,
          images: images,
          onLogits: (v) {
            actual ??= Float32List.fromList(v);
          },
        );
        images.clear();
        (message['content'] as List).clear();
        await pending;
        expect(actual, orderedEquals(expected!));
      },
    );

    test(
      'encoded and file PNG inputs preserve RGB logits and own their pixels',
      () async {
        final pixels = Uint8List.fromList([
          for (var y = 0; y < 8; y++)
            for (var x = 0; x < 8; x++) ...[x * 31, y * 29, (x + y) * 17],
        ]);
        final bytes = File(
          'test/fixtures/image_input/gradient.png',
        ).readAsBytesSync();
        final encoded = ChatImage.encoded(bytes);
        bytes.fillRange(0, bytes.length, 0);
        final fromFile = await ChatImage.fromFile(
          'test/fixtures/image_input/gradient.png',
        );
        List<Float32List>? expected;
        for (final image in [
          ChatImage.rgb(pixels, width: 8, height: 8),
          encoded,
          fromFile,
        ]) {
          final actual = <Float32List>[];
          await model
              .chat(maxNewTokens: 3)
              .send(
                'Describe.',
                images: [image],
                onLogits: (v) => actual.add(Float32List.fromList(v)),
              );
          expect(actual, hasLength(4));
          if (expected == null) {
            expected = actual;
          } else {
            for (var i = 0; i < actual.length; i++) {
              expect(actual[i], orderedEquals(expected[i]));
            }
          }
        }
      },
    );

    test(
      'interleaved images and tool responses remain bound across turns',
      () async {
        final session = model.chat(maxNewTokens: 1);
        await session.sendMessage(
          {
            'role': 'user',
            'content': [
              {'type': 'text', 'text': 'A'},
              {'type': 'image'},
              {'type': 'text', 'text': 'B'},
              {'type': 'image'},
            ],
          },
          images: [_image(80), _image(150)],
        );
        await session.sendMessage(
          {
            'role': 'tool',
            'content': [
              {'type': 'image'},
              {'type': 'text', 'text': 'Screenshot'},
            ],
          },
          images: [_image(220)],
        );
        expect(session.messages, hasLength(4));
        expect((await session.send('Compare.')).generatedTokens, 1);
      },
    );

    test(
      'invalid prompts and context overflow preserve existing history',
      () async {
        final session = model.chat(maxNewTokens: 1);
        await session.send('Describe.', images: [_image()]);
        final before = session.messages;
        await expectLater(
          session.send('word ' * 1000, images: [_image()]),
          throwsA(isA<ContextLengthExceeded>()),
        );
        expect(session.messages, before);
        await expectLater(
          session.sendMessage({
            'role': 'user',
            'content': [
              {'type': 'image'},
            ],
          }),
          throwsArgumentError,
        );
        await expectLater(
          session.sendMessage({'role': 'assistant', 'content': 'invalid'}),
          throwsArgumentError,
        );
        expect(session.messages, before);
        expect((await session.send('Continue.')).generatedTokens, 1);
      },
    );

    test(
      'the image token budget downscales images that overflow the context',
      () async {
        await model.close();
        final large = ChatImage.rgb(
          Uint8List(64 * 64 * 3)..fillRange(0, 64 * 64 * 3, 120),
          width: 64,
          height: 64,
        );
        Future<LocalVisionModel> load({
          int maxImageTokens = LocalVisionModel.defaultMaxImageTokens,
        }) => LocalVisionModel.load(
          '${root.path}/model',
          backboneDir: '${root.path}/backbone',
          backend: InferenceBackend.cpu,
          contextLength: 128,
          threads: 2,
          maxImageTokens: maxImageTokens,
        );
        // The default exceeds the fixture checkpoint's own 256-token budget.
        final uncapped = await load();
        try {
          expect(uncapped.maxImageTokens, 256);
          await expectLater(
            uncapped.chat(maxNewTokens: 1).send('Describe.', images: [large]),
            throwsA(isA<ContextLengthExceeded>()),
          );
        } finally {
          await uncapped.close();
        }
        final capped = await load(maxImageTokens: 4);
        try {
          expect(capped.maxImageTokens, 4);
          final turn = await capped
              .chat(maxNewTokens: 1)
              .send('Describe.', images: [large]);
          // 64 x 64 shrinks to 8 x 8, the size of an image within budget.
          final small = await capped
              .chat(maxNewTokens: 1)
              .send('Describe.', images: [_image()]);
          expect(turn.promptTokens, small.promptTokens);
        } finally {
          await capped.close();
        }
        await expectLater(load(maxImageTokens: 0), throwsArgumentError);
        expect(fixtureBackboneIsMapped(root), isFalse);
      },
    );

    test(
      'explicit cancellation rolls back draft and session can be reused',
      () async {
        final session = model.chat(maxNewTokens: 3);
        final pending = session.send('Describe.', images: [_image()]);
        final error = expectLater(
          pending,
          throwsA(isA<VisionRequestCancelled>()),
        );
        expect(() => session.clear(), throwsStateError);
        session.cancel();
        await error;
        expect(session.messages, isEmpty);
        expect(
          (await session.send('Retry.', images: [_image()])).generatedTokens,
          3,
        );
      },
    );

    test('returning false from onText commits a partial reply', () async {
      final session = model.chat(maxNewTokens: 8);
      final turn = await session.send(
        'Describe.',
        images: [_image()],
        onText: (_) => false,
      );
      expect(turn.generatedTokens, 1);
      expect(session.messages, hasLength(2));
      expect(session.messages.last['content'], turn.text);
    });

    test(
      'cancellation interrupts an outstanding asynchronous text callback',
      () async {
        final session = model.chat(maxNewTokens: 8);
        final reached = Completer<void>();
        final answer = Completer<bool>();
        final pending = session.send(
          'Describe.',
          images: [_image()],
          onText: (_) {
            reached.complete();
            return answer.future;
          },
        );
        final error = expectLater(
          pending,
          throwsA(isA<VisionRequestCancelled>()),
        );
        await reached.future;
        session.cancel();
        await error.timeout(const Duration(seconds: 5));
        answer.complete(true);
        expect(session.messages, isEmpty);
      },
    );

    test(
      'a cancel before an asynchronous text callback still ends the wait',
      () async {
        final session = model.chat(maxNewTokens: 8);
        var observed = 0;
        await expectLater(
          session
              .send(
                'Describe.',
                images: [_image()],
                // The first call sees the prompt; cancel after the first step.
                onLogits: (_) {
                  if (++observed == 2) session.cancel();
                },
                onText: (_) => Completer<bool>().future,
              )
              .timeout(const Duration(seconds: 5)),
          throwsA(isA<VisionRequestCancelled>()),
        );
        expect(observed, 2);
        expect(session.messages, isEmpty);
      },
    );

    test(
      'rejecting an overlapping session does not cancel its owner',
      () async {
        final a = model.chat(maxNewTokens: 4), b = model.chat(maxNewTokens: 4);
        final reached = Completer<void>();
        final answer = Completer<bool>();
        final pending = a.send(
          'Describe.',
          images: [_image()],
          onText: (_) {
            reached.complete();
            return answer.future;
          },
        );
        await reached.future;
        final rejected = b.send('Overlap.');
        b.cancel();
        await expectLater(rejected, throwsStateError);
        answer.complete(false);
        expect((await pending).generatedTokens, 1);
        expect(a.messages, hasLength(2));
        expect(b.messages, isEmpty);
        expect((await b.send('After.')).generatedTokens, 4);
      },
    );

    test('callback errors roll back the draft', () async {
      final session = model.chat(maxNewTokens: 4);
      await expectLater(
        session.send(
          'Describe.',
          images: [_image()],
          onText: (_) => throw StateError('callback failed'),
        ),
        throwsStateError,
      );
      expect(session.messages, isEmpty);
      expect((await session.send('Retry.')).generatedTokens, 4);
    });

    test('closing cancels active work and rejects further model use', () async {
      final session = model.chat(maxNewTokens: 4);
      final pending = session.send('Describe.', images: [_image()]);
      final error = expectLater(
        pending,
        throwsA(isA<VisionRequestCancelled>()),
      );
      await model.close();
      await error;
      await model.close();
      expect(model.isClosed, isTrue);
      expect(session.messages, isEmpty);
      expect(() => model.chat(), throwsStateError);
      await expectLater(session.send('Closed.'), throwsStateError);
    });

    test(
      'close awaited inside onText does not deadlock or commit a draft',
      () async {
        final session = model.chat(maxNewTokens: 4);
        final pending = session.send(
          'Describe.',
          images: [_image()],
          onText: (_) async {
            await model.close();
            return true;
          },
        );
        await expectLater(
          pending.timeout(const Duration(seconds: 5)),
          throwsA(isA<VisionRequestCancelled>()),
        );
        await model.close();
        expect(model.isClosed, isTrue);
        expect(session.messages, isEmpty);
      },
    );

    test(
      'invalid bundle tokenizer is rejected without retaining weight mappings',
      () async {
        await model.close();
        expect(fixtureBackboneIsMapped(root), isFalse);
        final file = File('${root.path}/backbone/tokenizer.json');
        final original = file.readAsStringSync();
        try {
          file.writeAsStringSync('invalid JSON');
          await expectLater(
            LocalVisionModel.load(
              '${root.path}/model',
              backboneDir: '${root.path}/backbone',
              backend: InferenceBackend.cpu,
            ),
            throwsFormatException,
          );
          expect(fixtureBackboneIsMapped(root), isFalse);
        } finally {
          file.writeAsStringSync(original);
        }
      },
    );

    test('rejects mismatched checkpoint provenance before loading', () async {
      final file = File('${root.path}/backbone/vl_backbone.json');
      final original = file.readAsStringSync();
      try {
        final identity = jsonDecode(original) as Map<String, Object?>;
        identity['source_key'] = 'wrong';
        file.writeAsStringSync(jsonEncode(identity));
        await expectLater(
          LocalVisionModel.load(
            '${root.path}/model',
            backboneDir: '${root.path}/backbone',
            backend: InferenceBackend.cpu,
          ),
          throwsArgumentError,
        );
      } finally {
        file.writeAsStringSync(original);
      }
    });
  }, skip: LocalModel.isSupported ? false : 'requires native kernels');
}
