import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/gguf_fixture.dart';
import 'support/synthetic_tokenizer.dart';
import 'support/vision_model_fixture.dart';

void main() {
  final oracle =
      jsonDecode(
            File(
              'test/fixtures/audio/conversation_oracle.json',
            ).readAsStringSync(),
          )
          as Map;
  test('voice prompt matches the pinned official processor template', () {
    expect(
      '$audioConversationHead$audioConversationUser<|audio|>$audioConversationAssistant',
      oracle['first_prompt'],
    );
  });
  for (final c in oracle['cases'] as List) {
    test('upstream interleaving schedule: ${c['name']}', () {
      final state = AudioInterleavingState(
        textEnd: 102,
        turnEnd: 101,
        mixedStart: 103,
        audioStart: 104,
      );
      for (final row in c['rows'] as List) {
        final value = row['input'];
        expect(state.nextIsAudio, value is String);
        if (value is String) {
          state.acceptAudio(eos: value == 'eos');
        } else {
          state.acceptText(value as int);
        }
        expect(state.mode.name, row['mode']);
        expect(state.slotsLeft, row['slots_left']);
        expect(state.speaking, row['speaking']);
        expect(state.textDone, row['text_done']);
        expect(state.finished, row['finished']);
      }
      expect(() => state.acceptText(7), throwsStateError);
    });
  }
  group('conversation ownership and rollback', () {
    late Directory root;
    late Lfm3AudioGenerator generator;
    late Lfm3Model model;
    late AudioConversationSession session;
    setUp(() async {
      root = Directory.systemTemp.createTempSync('dianoia-conversation-test-');
      NativeKernels.load(null, threads: 2);
      final g =
          jsonDecode(
                File(
                  'test/fixtures/audio/generation_oracle.json',
                ).readAsStringSync(),
              )
              as Map;
      writeTestGguf(
        '${root.path}/audio.gguf',
        (g['metadata'] as Map).cast<String, Object>(),
        [
          for (final t in g['tensors'] as List)
            (
              t['name'] as String,
              (t['shape'] as List).cast<int>(),
              GgufType.values.byName(t['type'] as String),
              base64Decode(t['bytes'] as String),
            ),
        ],
      );
      generator = Lfm3AudioGenerator.load('${root.path}/audio.gguf');
      await prepareVisionFixture(root);
      final dir = '${root.path}/backbone';
      writeSyntheticTokenizer(
        dir,
        additionalSpecials: [
          '<|audio|>',
          '<|audio_out|>',
          '<|text_end|>',
          '<|mixed_start|>',
          '<|audio_start|>',
        ],
      );
      model = await Lfm3Model.load(
        dir,
        maxSeq: 512,
        weights: WeightFormat.q8,
        q8CachePath: '$dir/lfm3-q8.cache',
      );
      session = AudioConversationSession(
        model,
        Lfm3Tokenizer.load(dir),
        generator,
      );
    });
    tearDown(() async {
      session.close();
      generator.close();
      await model.close();
      root.deleteSync(recursive: true);
    });
    test(
      'input validation is retryable; cancellation discards only pending state',
      () async {
        await session.begin(maxNewSlots: 20, maxFrames: 5);
        final length = model.length;
        await expectLater(
          session.addProjected(Float32List(31)),
          throwsArgumentError,
        );
        await expectLater(
          session.addProjected(Float32List(512 * 32)),
          throwsStateError,
        );
        expect(model.length, length);
        await session.addProjected(Float32List(32));
        expect(model.length, length + 1);
        await expectLater(session.begin(maxNewSlots: 20), throwsStateError);
        session.cancel();
        expect(session.historyTokens, 0);
        await session.begin(maxNewSlots: 20);
        await session.addProjected(Float32List(32));
        final finishing = session.finish();
        session.cancel();
        await expectLater(finishing, throwsA(isA<AudioSynthesisCancelled>()));
        expect(session.turns, 0);
        session.clear();
        expect(session.retainedEmbeddingBytes, 0);
      },
    );
    test(
      'callback failure rolls back; foreign model state and closed generators fail cleanly',
      () async {
        await session.begin(maxNewSlots: 20);
        await session.addProjected(Float32List(32));
        await expectLater(
          session.finish(onSlot: (_, _) => throw StateError('capture failed')),
          throwsA(isA<StateError>()),
        );
        expect(session.turns, 0);
        await session.begin(maxNewSlots: 20);
        await model.reset();
        await expectLater(
          session.addProjected(Float32List(32)),
          throwsStateError,
        );
        await session.begin(maxNewSlots: 20);
        await session.addProjected(Float32List(32));
        generator.close();
        await expectLater(session.finish(), throwsStateError);
        session.cancel();
        session.clear();
      },
    );
  }, skip: NativeKernels.isAvailable ? false : 'native kernels unavailable');

  final dir = Platform.environment['DIANOIA_AUDIO_CONVERSATION_DIR'];
  final first = Platform.environment['DIANOIA_AUDIO_CONVERSATION_FIRST'];
  final followup = Platform.environment['DIANOIA_AUDIO_CONVERSATION_FOLLOWUP'];
  test(
    'real Q4 voice conversation preserves audio history and acknowledges replies transactionally',
    () async {
      NativeKernels.load(null, threads: 2);
      final generator = Lfm3AudioGenerator.load(
        '$dir/${Lfm3AudioQ4Source.audio.name}',
      );
      final encoder = Lfm3AudioEncoder.load(
        '$dir/${Lfm3AudioQ4Source.audio.name}',
      );
      final model = await Lfm3Model.load(
        '$dir/language',
        maxSeq: 2048,
        weights: WeightFormat.q4,
        q8CachePath: '$dir/language/lfm3-q4.cache',
      );
      final tokenizer = Lfm3Tokenizer.load('$dir/language');
      final session = AudioConversationSession(model, tokenizer, generator);
      Future<void> input(String path) async {
        final bytes = ByteData.sublistView(File(path).readAsBytesSync());
        final pcm = Float32List(bytes.lengthInBytes ~/ 4);
        for (var i = 0; i < pcm.length; i++) {
          pcm[i] = bytes.getFloat32(i * 4, Endian.little);
        }
        final rows = encoder.encode(
          Lfm3AudioFrontend().extract(pcm, samplingRate: 16000),
        );
        await session.begin();
        await session.addProjected(rows);
      }

      try {
        await input(first!);
        final rejected = await session.finish(deferCommit: true);
        expect(rejected.text.toLowerCase(), contains('four'));
        expect(session.turns, 0);
        expect(session.historyTokens, 0);
        await expectLater(session.begin(), throwsStateError);
        session.cancel();
        await input(first);
        final text = <String>[], audio = <double>[];
        var sawEos = false;
        final accepted = await session.finish(
          deferCommit: true,
          onText: text.add,
          onAudio: audio.addAll,
          onSlot: (_, codes) {
            if (codes != null && codes.contains(generator.eosCode)) {
              expect(codes.every((c) => c == generator.eosCode), isTrue);
              sawEos = true;
            }
          },
        );
        expect(sawEos, isTrue);
        expect(audio, accepted.samples);
        expect(text.last.trim(), accepted.text);
        session.acceptReply();
        expect(session.turns, 1);
        final tokens = session.historyTokens;
        expect(session.retainedEmbeddingBytes, greaterThan(0));
        // Standalone ASR/TTS may use the backbone between turns; replay restores it.
        await model.reset();
        await model.forward(tokenizer.encode('An unrelated task.'));
        await input(followup!);
        final reply = await session.finish(deferCommit: true);
        expect(reply.text.toLowerCase(), contains('seven'));
        expect(reply.samples.every((v) => v.isFinite), isTrue);
        session.acceptReply();
        expect(session.turns, 2);
        expect(session.historyTokens, greaterThan(tokens));
        await expectLater(session.begin(maxNewSlots: 1), completes);
        await expectLater(
          session.addProjected(Float32List(2048 * model.config.hiddenSize)),
          throwsStateError,
        );
        session.cancel();
        expect(session.turns, 2);
        session.clear();
        expect(session.historyTokens, 0);
        expect(session.retainedEmbeddingBytes, 0);
      } finally {
        session.close();
        encoder.close();
        generator.close();
        await model.close();
      }
    },
    skip: dir == null || first == null || followup == null
        ? 'set DIANOIA_AUDIO_CONVERSATION_DIR/FIRST/FOLLOWUP'
        : false,
    timeout: const Timeout(Duration(minutes: 2)),
  );
}
