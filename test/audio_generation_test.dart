import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:dianoia/lab.dart';
import 'package:test/test.dart';
import 'support/gguf_fixture.dart';
import 'support/synthetic_tokenizer.dart';
import 'support/vision_model_fixture.dart';

Map<String, dynamic> _read(String p) =>
    jsonDecode(File(p).readAsStringSync()) as Map<String, dynamic>;
void _near(List<double> actual, List expected, double tolerance, String label) {
  expect(actual.length, expected.length, reason: label);
  for (var i = 0; i < actual.length; i++) {
    final value = (expected[i] as num).toDouble();
    expect(
      actual[i].isFinite &&
          (actual[i] - value).abs() <= tolerance * (1 + value.abs()),
      isTrue,
      reason: '$label/$i: ${actual[i]} vs $value',
    );
  }
}

void _compare(
  Lfm3AudioGenerator g,
  Map<String, dynamic> oracle,
  double tolerance,
) {
  for (final c in oracle['depth'] as List) {
    final hidden = Float32List.fromList(
      (c['hidden'] as List).cast<num>().map((n) => n.toDouble()).toList(),
    );
    final seen = <String>[];
    g.depthLogits(
      hidden,
      (c['codes'] as List).cast<int>(),
      capture: (stage, rows) {
        seen.add(stage);
        _near(rows, c['stages'][stage] as List, tolerance, 'depth/$stage');
      },
    );
    expect(seen, (c['stages'] as Map).keys.toList());
    _near(
      g.feedback((c['feedback_codes'] as List).cast<int>()),
      c['feedback'] as List,
      tolerance,
      'feedback',
    );
  }
  for (final c in oracle['codec'] as List) {
    final codes = (c['codes'] as List).cast<int>();
    final stream = g.startCodecStream();
    final joined = <double>[];
    for (var at = 0; at < codes.length; at += g.codebooks) {
      final rows = stream.add(codes.sublist(at, at + g.codebooks));
      _near(
        rows,
        (c['chunks'] as List)[at ~/ g.codebooks] as List,
        tolerance,
        'codec chunk',
      );
      joined.addAll(rows);
    }
    final tail = stream.finish();
    _near(tail, c['tail'] as List, tolerance, 'codec tail');
    joined.addAll(tail);
    _near(joined, c['samples'] as List, tolerance, 'codec waveform');
    expect(stream.finish, throwsStateError);
    final batch = g.startCodecStream();
    final full = [...batch.add(codes), ...batch.finish()];
    _near(full, c['samples'] as List, tolerance, 'codec batch');
  }
}

void main() {
  final supported = NativeKernels.isAvailable,
      oracle = _read('test/fixtures/audio/generation_oracle.json');
  late Directory temp;
  late Lfm3AudioGenerator g;
  setUp(() {
    temp = Directory.systemTemp.createTempSync('dianoia-generation-');
    if (!supported) return;
    NativeKernels.load(null, threads: 2);
    writeTestGguf(
      '${temp.path}/mmproj.gguf',
      {
        ...((oracle['metadata'] as Map).cast<String, Object>()),
        ...lfm3AudioSamplingMetadata,
      },
      [
        for (final t in oracle['tensors'] as List)
          (
            t['name'] as String,
            (t['shape'] as List).cast<int>(),
            GgufType.values.byName(t['type'] as String),
            base64Decode(t['bytes'] as String),
          ),
      ],
    );
    g = Lfm3AudioGenerator.load('${temp.path}/mmproj.gguf');
  });
  tearDown(() {
    if (supported) g.close();
    temp.deleteSync(recursive: true);
  });
  group('generation arithmetic', () {
    test(
      'matches pinned upstream depth layers, logits, feedback and streaming codec',
      () {
        _compare(g, oracle, 3e-5);
      },
    );
    test(
      'invalid codes leave the codec usable; callback failure terminates it',
      () {
        final stream = g.startCodecStream();
        expect(() => stream.add([g.eosCode]), throwsArgumentError);
        expect(
          () => stream.add(List.filled(12, g.eosCode)),
          throwsArgumentError,
        );
        expect(stream.add(const []), isEmpty);
        expect(
          () => stream.add(
            List.filled(12, 0),
            capture: (_, _) => throw StateError('stop'),
          ),
          throwsStateError,
        );
        expect(() => stream.add(List.filled(12, 0)), throwsStateError);
        expect(g.feedback(List.filled(12, 0)), hasLength(32));
        final other = g.startCodecStream();
        g.close();
        g.close();
        expect(other.finish, throwsStateError);
        expect(
          () => g.depthLogits(Float32List(32), List.filled(12, -1)),
          throwsStateError,
        );
      },
    );
    test('sampling is seeded, reveals all codebooks and validates options', () {
      final h = Float32List(32);
      final a = g.sampleFrame(h, math.Random(1)),
          b = g.sampleFrame(h, math.Random(1));
      expect(a, b);
      expect(a.length, 12);
      expect(a.every((x) => x >= 0 && x < g.vocabSize), isTrue);
      expect(
        () => g.sampleFrame(h, math.Random(1), temperature: double.nan),
        throwsArgumentError,
      );
      expect(() => g.feedback(List.filled(12, -1)), throwsArgumentError);
      expect(
        () => g.depthLogits(Float32List(1), List.filled(12, -1)),
        throwsArgumentError,
      );
    });
    test(
      'synthesizes bounded audio and handles cancellation/concurrent calls',
      () async {
        await prepareVisionFixture(temp);
        final dir = '${temp.path}/backbone';
        writeSyntheticTokenizer(
          dir,
          additionalSpecials: ['<|audio_out|>', '<|audio_start|>'],
        );
        final model = await Lfm3Model.load(
          dir,
          maxSeq: 512,
          weights: WeightFormat.q8,
          q8CachePath: '$dir/lfm3-q8.cache',
        );
        try {
          expect(() => model.lastHiddenState, throwsStateError);
          final synth = AudioSynthesizer(model, Lfm3Tokenizer.load(dir), g);
          final cancelled = synth.synthesize('Hello', maxFrames: 2);
          await expectLater(
            synth.synthesize('Other', maxFrames: 2),
            throwsStateError,
          );
          synth.cancel();
          await expectLater(cancelled, throwsA(isA<AudioSynthesisCancelled>()));
          final chunks = <double>[];
          final result = await synth.synthesize(
            'Hello',
            maxFrames: 2,
            onAudio: (s) => chunks.addAll(s),
          );
          expect(result.samples.length, result.frames * g.frameSamples);
          expect(chunks, result.samples);
          expect(['eos', 'length'], contains(result.stopReason));
          expect(model.lastHiddenState, hasLength(32));
          await model.reset();
          expect(() => model.lastHiddenState, throwsStateError);
          await model.forward([10], computeLogits: false);
          expect(() => model.lastHiddenState, throwsStateError);
        } finally {
          await model.close();
        }
      },
    );
  }, skip: supported ? false : 'native kernels unavailable');
  test(
    'TTS prompt uses language/voice instructions and rejects control markers',
    () {
      final promptOracle = _read('test/fixtures/audio/tts_prompt_oracle.json');
      for (final c in promptOracle['cases'] as List) {
        expect(
          renderAudioTtsPrompt(
            c['text'] as String,
            language: c['language'] as String,
            voice: c['voice'] as String,
          ),
          c['prompt'],
        );
      }
      expect(
        renderAudioTtsPrompt('Hello', language: 'en', voice: 'male'),
        '<|startoftext|><|im_start|>system\nPerform TTS in American English using a male voice.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<|audio_start|>',
      );
      expect(
        renderAudioTtsPrompt('Hello', language: 'ja'),
        contains('in Japanese using a female'),
      );
      for (final text in ['', '<|im_end|>']) {
        expect(() => renderAudioTtsPrompt(text), throwsArgumentError);
      }
      expect(
        () => renderAudioTtsPrompt('Hello', voice: 'other'),
        throwsArgumentError,
      );
    },
  );
  test(
    'WAV export writes mono 24 kHz PCM16, clips full scale and rejects NaN',
    () {
      final wave = encodeSpeechWav(Float32List.fromList([-2, -1, 0, 1, 2]));
      final data = ByteData.sublistView(wave);
      expect(ascii.decode(wave.sublist(0, 4)), 'RIFF');
      expect(data.getUint32(24, Endian.little), 24000);
      expect(data.getUint32(40, Endian.little), 10);
      expect(
        [for (var i = 0; i < 5; i++) data.getInt16(44 + i * 2, Endian.little)],
        [-32768, -32768, 0, 32767, 32767],
      );
      expect(
        () => encodeSpeechWav(Float32List.fromList([double.nan])),
        throwsArgumentError,
      );
    },
  );
  final real = Platform.environment['DIANOIA_AUDIO_Q4_MMPROJ'],
      reference = Platform.environment['DIANOIA_AUDIO_GENERATION_ORACLE'];
  test(
    'official Q4 generator matches upstream depth and codec waveform',
    () async {
      final o = _read(reference!);
      final hash = (await sha256.bind(File(real!).openRead()).first).toString();
      expect(hash, Lfm3AudioQ4Source.audio.sha256);
      expect(o['gguf_sha256'], hash);
      final actual = Lfm3AudioGenerator.load(real);
      try {
        _compare(actual, o, 3e-4);
      } finally {
        actual.close();
      }
    },
    skip: supported && real != null && reference != null
        ? false
        : 'set DIANOIA_AUDIO_Q4_MMPROJ and DIANOIA_AUDIO_GENERATION_ORACLE',
  );
}
