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
            File('test/fixtures/audio/live_oracle.json').readAsStringSync(),
          )
          as Map;
  final policy = AudioLiveTokenPolicy(
    (oracle['markers'] as Map).cast<String, int>(),
    32,
  );
  for (final c in oracle['cases'] as List) {
    test('official live token protocol: ${c['name']}', () {
      for (final chunk in c['chunks'] as List) {
        final finalChunk = chunk['final'] as bool;
        final generated = <int>[];
        var closed = false;
        if (chunk['rows'] != 0 || finalChunk) {
          final scores = chunk['logits'] as List;
          for (var i = 0; i < chunk['limit']; i++) {
            final logits = Float32List(32)..fillRange(0, 32, -1000);
            final ranked = scores[i] as List;
            for (var j = 0; j < ranked.length; j++) {
              logits[ranked[j] as int] = 100.0 - j;
            }
            final token = policy.pick(logits, isFinal: finalChunk);
            if (policy.closesBlock(token, isFinal: finalChunk)) {
              closed = true;
              break;
            }
            generated.add(token);
          }
        }
        expect(generated, chunk['ids']);
        expect(closed, chunk['closed']);
      }
    });
  }
  test('live token policy validates vocabulary and finite candidates', () {
    expect(() => AudioLiveTokenPolicy({}, 32), throwsFormatException);
    final duplicate = Map<String, int>.from(oracle['markers'] as Map)
      ..['<|text_end|>'] = policy.audio;
    expect(() => AudioLiveTokenPolicy(duplicate, 32), throwsFormatException);
    expect(
      () => policy.pick(Float32List(1), isFinal: false),
      throwsArgumentError,
    );
    expect(
      () => policy.pick(
        Float32List(32)..fillRange(0, 32, double.nan),
        isFinal: false,
      ),
      throwsStateError,
    );
  });
  group('live session ownership', () {
    late Directory root;
    late Lfm3Model model;
    late Lfm3Tokenizer tokenizer;
    late Lfm3AudioEncoder encoder;
    setUp(() async {
      root = Directory.systemTemp.createTempSync('dianoia-live-');
      NativeKernels.load(null, threads: 2);
      await prepareVisionFixture(root);
      final dir = '${root.path}/backbone';
      writeSyntheticTokenizer(
        dir,
        additionalSpecials: AudioLiveTokenPolicy.names
            .where((n) => !['<|im_end|>', '<|im_start|>'].contains(n))
            .toList(),
      );
      tokenizer = Lfm3Tokenizer.load(dir);
      model = await Lfm3Model.load(
        dir,
        maxSeq: 512,
        weights: WeightFormat.q8,
        q8CachePath: '$dir/lfm3-q8.cache',
      );
      final e =
          jsonDecode(
                File(
                  'test/fixtures/audio/encoder_oracle.json',
                ).readAsStringSync(),
              )
              as Map;
      writeTestGguf(
        '${root.path}/audio.gguf',
        (e['metadata'] as Map).cast<String, Object>(),
        [
          for (final t in e['tensors'] as List)
            (
              t['name'] as String,
              (t['shape'] as List).cast<int>(),
              GgufType.values.byName(t['type'] as String),
              base64Decode(t['bytes'] as String),
            ),
        ],
      );
      encoder = Lfm3AudioEncoder.load('${root.path}/audio.gguf');
    });
    tearDown(() async {
      encoder.close();
      await model.close();
      root.deleteSync(recursive: true);
    });
    test(
      'defers prompt/text-end; validates before state changes and finalizes empty tails',
      () async {
        final session = await AudioLiveSession.start(model, tokenizer);
        expect(model.length, 0);
        final empty = await session.addProjected(Float32List(0));
        expect(empty.ids, isEmpty);
        expect(model.length, 0);
        await expectLater(
          session.addProjected(Float32List(31)),
          throwsArgumentError,
        );
        await expectLater(
          session.addProjected(Float32List(512 * 32)),
          throwsStateError,
        );
        expect(model.length, 0);
        final first = await session.addProjected(
          Float32List(12 * 32),
          maxNewTokens: 1,
        );
        expect(
          model.length,
          tokenizer.encode(audioLivePrompt).length + 12 + first.ids.length,
        );
        final before = model.length;
        await session.addProjected(Float32List(0));
        expect(model.length, before);
        final next = await session.addProjected(
          Float32List(32),
          maxNewTokens: 1,
        );
        expect(model.length, before + 2 + next.ids.length);
        final prior = model.length;
        final last = await session.addProjected(
          Float32List(0),
          isFinal: true,
          maxNewTokens: 1,
        );
        expect(model.length, prior + 1 + last.ids.length);
        expect(session.isFinished, isTrue);
        await expectLater(
          session.addProjected(Float32List(32)),
          throwsStateError,
        );
        session.close();
      },
    );
    test(
      'cancellation and external backbone changes terminate the live state',
      () async {
        final session = await AudioLiveSession.start(model, tokenizer);
        final pending = session.addProjected(Float32List(12 * 32));
        await expectLater(
          session.addProjected(Float32List(32)),
          throwsStateError,
        );
        session.close();
        await expectLater(pending, throwsStateError);
        final next = await AudioLiveSession.start(model, tokenizer);
        await model.forward([1]);
        await expectLater(next.addProjected(Float32List(32)), throwsStateError);
        next.close();
      },
    );
    test(
      'PCM wrapper rejects bad input, bounds features and closes on callback errors',
      () async {
        final live = await Lfm3LiveTranscriber.start(
          model,
          tokenizer,
          encoder,
          onText: (_) => throw StateError('callback failed'),
        );
        await expectLater(
          live.addPcm(Float32List.fromList([double.nan])),
          throwsArgumentError,
        );
        await expectLater(
          live.addPcm(Float32List(30 * 16000 + 1)),
          throwsArgumentError,
        );
        await live.addPcm(Float32List(1280));
        expect(live.pendingFeatureFrames, lessThan(96));
        await expectLater(live.addPcm(Float32List(16000)), throwsStateError);
        await expectLater(live.addPcm(Float32List(1)), throwsStateError);
        live.close();
      },
    );
  }, skip: NativeKernels.isAvailable ? false : 'native kernels unavailable');

  final dir = Platform.environment['DIANOIA_AUDIO_EXAMPLE_DIR'];
  final file = Platform.environment['DIANOIA_AUDIO_EXAMPLE_PCM'];
  test(
    'official Q4 English LiveASR updates before finish and flushes zero-row tails',
    () async {
      NativeKernels.load(null, threads: 2);
      final model = await Lfm3Model.load(
        '$dir/language',
        maxSeq: 2048,
        weights: WeightFormat.q4,
        q8CachePath: '$dir/language/lfm3-q4.cache',
      );
      final tokenizer = Lfm3Tokenizer.load('$dir/language');
      final encoder = Lfm3AudioEncoder.load(
        '$dir/${Lfm3AudioQ4Source.audio.name}',
      );
      final bytes = ByteData.sublistView(File(file!).readAsBytesSync());
      final pcm = Float32List(bytes.lengthInBytes ~/ 4);
      for (var i = 0; i < pcm.length; i++) {
        pcm[i] = bytes.getFloat32(i * 4, Endian.little);
      }
      try {
        List<int>? baseline;
        for (final chunk in [1280, 1600]) {
          final updates = <String>[],
              ids = <int>[],
              blocks = <AudioLiveBlock>[];
          var beforeFinish = true, early = false;
          final live = await Lfm3LiveTranscriber.start(
            model,
            tokenizer,
            encoder,
            onText: (text) {
              updates.add(text);
              if (text.isNotEmpty && beforeFinish) early = true;
            },
            onBlock: (b) {
              blocks.add(b);
              ids.addAll(b.ids);
            },
          );
          for (var at = 0; at < pcm.length; at += chunk) {
            await live.addPcm(
              pcm.sublist(at, (at + chunk).clamp(0, pcm.length)),
            );
            expect(live.pendingFeatureFrames, lessThan(96));
          }
          expect(early, isTrue);
          beforeFinish = false;
          final text = await live.finish();
          expect(text, 'The quick brown fox jumps over the lazy dog.');
          expect(updates.last, text);
          expect(blocks.map((b) => b.audioTokens), [12, 12, 10]);
          expect(blocks.every((b) => b.closed), isTrue);
          if (baseline != null) {
            expect(ids, baseline);
          } else {
            baseline = ids;
          }
        }
        final blocks = <AudioLiveBlock>[];
        final exact = await Lfm3LiveTranscriber.start(
          model,
          tokenizer,
          encoder,
          onBlock: blocks.add,
        );
        await exact.addPcm(
          pcm.sublist(0, 15600),
        ); // Exactly 96 complete mel frames.
        await exact.finish();
        expect(blocks.last.isFinal, isTrue);
        expect(blocks.last.audioTokens, 0);
      } finally {
        encoder.close();
        await model.close();
      }
    },
    skip: dir == null || file == null
        ? 'set DIANOIA_AUDIO_EXAMPLE_DIR/PCM'
        : false,
    timeout: const Timeout(Duration(minutes: 2)),
  );
}
