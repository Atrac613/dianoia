import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/gguf_fixture.dart';
import 'support/synthetic_tokenizer.dart';
import 'support/vision_model_fixture.dart';

const _speechSpecials = [
  '<|audio|>',
  '<|audio_out|>',
  '<|audio_start|>',
  '<|mixed_start|>',
  '<|text_end|>',
];

void main() {
  final oracle =
      jsonDecode(
            File('test/fixtures/omni/prompt_oracle.json').readAsStringSync(),
          )
          as Map;
  String rendered(String name) =>
      (oracle['cases'] as List).firstWhere((c) => c['name'] == name)['text']
          as String;

  group('Omni prompts match the pinned template', () {
    late Directory temp;
    late Lfm3Tokenizer tokenizer;
    setUp(() {
      temp = Directory.systemTemp.createTempSync('dianoia-omni-prompt-');
      writeSyntheticTokenizer(temp.path, additionalSpecials: _speechSpecials);
      tokenizer = Lfm3Tokenizer.load(temp.path);
    });
    tearDown(() => temp.deleteSync(recursive: true));

    for (final (name, language) in [
      ('asr_english', 'English'),
      ('asr_auto', null),
    ]) {
      test('ASR: $name', () {
        final prompt = prepareAudioTranscriptionPrompt(
          tokenizer: tokenizer,
          audioTokens: 5,
          hiddenSize: 32,
          language: language,
          variant: Lfm3SpeechVariant.omni,
        );
        expect(prompt.text, rendered(name));
        final id = tokenizer.addedTokenId;
        expect(prompt.ids[prompt.audioStart - 1], id('<|vision_start|>'));
        expect(
          prompt.ids.sublist(prompt.audioStart, prompt.audioStart + 5),
          List.filled(5, id('<|audio|>')),
        );
        expect(prompt.ids[prompt.audioStart + 5], id('<|vision_end|>'));
        expect(prompt.ids.sublist(prompt.ids.length - 2), [
          id('<think>'),
          id('</think>'),
        ]);
        for (final marker in ['<|image|>', '<|video|>', '<|audio_out|>']) {
          expect(prompt.suppressedTokenIds, contains(id(marker)));
        }
      });
    }

    test('ASR for LFM3-Audio keeps its bare placeholder', () {
      final prompt = prepareAudioTranscriptionPrompt(
        tokenizer: tokenizer,
        audioTokens: 1,
        hiddenSize: 32,
        language: 'English',
      );
      expect(prompt.text, isNot(contains('<|vision_start|>')));
      expect(prompt.text, isNot(contains('<think>')));
      expect(
        prompt.suppressedTokenIds,
        isNot(contains(tokenizer.addedTokenId('<|image|>'))),
      );
    });

    test('TTS opens speech after the empty reasoning block', () {
      expect(
        renderAudioTtsPrompt(
          'The quick brown fox jumps over the lazy dog',
          language: 'en-us',
          variant: Lfm3SpeechVariant.omni,
        ),
        '${rendered('tts')}<|audio_start|>',
      );
    });

    test('voice conversation wraps user audio and opens a mixed reply', () {
      const v = Lfm3SpeechVariant.omni;
      expect(
        '$audioConversationHead${audioConversationUserOpen(v)}<|audio|>'
            '${audioConversationAssistantOpen(v)}',
        '${rendered('interleaved_audio')}<|mixed_start|>',
      );
      const a = Lfm3SpeechVariant.audio;
      expect(audioConversationUserOpen(a), audioConversationUser);
      expect(audioConversationAssistantOpen(a), audioConversationAssistant);
    });

    test('image chat renders the non-reasoning assistant opening', () {
      expect(
        tokenizer.decode(
          tokenizer.encode(
            renderVisionChat([
              {
                'role': 'user',
                'content': [
                  {'type': 'image'},
                  {'type': 'text', 'text': 'Describe this image.'},
                ],
              },
            ], enableThinking: false),
          ),
        ),
        rendered('image_text'),
      );
    });
  });

  group('mixed Omni prompts', () {
    late Directory temp;
    late Lfm3Tokenizer tokenizer;
    late Lfm3VlConfig config;
    setUp(() {
      temp = Directory.systemTemp.createTempSync('dianoia-omni-mixed-');
      writeSyntheticTokenizer(temp.path, additionalSpecials: _speechSpecials);
      tokenizer = Lfm3Tokenizer.load(temp.path);
      final json =
          jsonDecode(File('test/fixtures/omni/config.json').readAsStringSync())
              as Map<String, Object?>;
      final id = tokenizer.addedTokenId;
      config = Lfm3VlConfig.fromOmniJson(
        {
          ...json,
          'image_token_id': id('<|image|>'),
          'video_token_id': id('<|video|>'),
        },
        visionStartTokenId: id('<|vision_start|>')!,
        visionEndTokenId: id('<|vision_end|>')!,
      );
    });
    tearDown(() => temp.deleteSync(recursive: true));

    List<Map<String, Object?>> messages(String name) => [
      for (final m
          in (oracle['cases'] as List).firstWhere(
                (c) => c['name'] == name,
              )['messages']
              as List)
        (m as Map).cast<String, Object?>(),
    ];

    test('renders every oracle case with its reply marker', () {
      for (final (name, mode, marker) in [
        ('asr_english', AudioReplyMode.text, ''),
        ('image_text', AudioReplyMode.text, ''),
        ('image_spoken_question', AudioReplyMode.text, ''),
        ('tts', AudioReplyMode.audio, '<|audio_start|>'),
        ('interleaved_audio', AudioReplyMode.mixed, '<|mixed_start|>'),
      ]) {
        expect(
          renderOmniChat(messages(name), mode: mode),
          '${rendered(name)}$marker',
          reason: name,
        );
      }
    });

    test('expands image and audio runs in content order', () {
      final prompt = prepareOmniPrompt(
        messages('image_spoken_question'),
        tokenizer: tokenizer,
        config: config,
        imageTokenCounts: [4],
        audioTokenCounts: [3],
      );
      final id = tokenizer.addedTokenId;
      final [image, audio] = prompt.spans;
      expect([image.kind, image.count], [OmniMediaKind.image, 4]);
      expect([audio.kind, audio.count], [OmniMediaKind.audio, 3]);
      expect(prompt.ids.sublist(image.start - 1, image.start + 5), [
        id('<|vision_start|>'),
        ...List.filled(4, id('<|image|>')),
        id('<|vision_end|>'),
      ]);
      expect(audio.start, image.start + 4 + 2);
      expect(prompt.ids.sublist(audio.start, audio.start + 4), [
        ...List.filled(3, id('<|audio|>')),
        id('<|vision_end|>'),
      ]);
      expect(
        tokenizer.decode(prompt.ids),
        rendered('image_spoken_question')
            .replaceFirst('<|image|>', '<|image|>' * 4)
            .replaceFirst('<|audio|>', '<|audio|>' * 3),
      );
    });

    test('rejects missing counts, bare placeholders and video', () {
      final question = messages('image_spoken_question');
      expect(
        () => prepareOmniPrompt(
          question,
          tokenizer: tokenizer,
          config: config,
          imageTokenCounts: [4],
        ),
        throwsArgumentError,
      );
      expect(
        () => prepareOmniPrompt(
          [
            {'role': 'user', 'content': 'say <|audio|>'},
          ],
          tokenizer: tokenizer,
          config: config,
          audioTokenCounts: [1],
        ),
        throwsArgumentError,
      );
    });

    test('expands a video into timestamped groups with its soundtrack', () {
      final layout = OmniVideoLayout(
        timestamps: [0.25, 1.0],
        groupStarts: [0, 0.75],
        tokensPerGroup: 3,
        soundtrackTokens: 20,
      );
      // round(0.75 s / 80 ms) = 9 rows go with the first group.
      expect(layout.soundtrackBounds, [0, 9, 20]);
      final prompt = prepareOmniPrompt(
        [
          {
            'role': 'user',
            'content': [
              {'type': 'video'},
              {'type': 'text', 'text': 'What happens?'},
            ],
          },
        ],
        tokenizer: tokenizer,
        config: config,
        videos: [layout],
      );
      const v = '<|vision_start|><|video|><|video|><|video|><|vision_end|>';
      expect(
        tokenizer.decode(prompt.ids),
        '<|startoftext|><|im_start|>user\n<|vision_start|>'
        '<0.2 seconds>$v${'<|audio|>' * 9}'
        '<1.0 seconds>$v${'<|audio|>' * 11}'
        '<|vision_end|>What happens?<|im_end|>\n'
        '<|im_start|>assistant\n<think></think>',
      );
      expect(
        [for (final s in prompt.spans) (s.kind, s.index, s.count, s.rowOffset)],
        [
          (OmniMediaKind.video, 0, 3, 0),
          (OmniMediaKind.soundtrack, 0, 9, 0),
          (OmniMediaKind.video, 0, 3, 3),
          (OmniMediaKind.soundtrack, 0, 11, 9),
        ],
      );
      final silent = prepareOmniPrompt(
        [
          {
            'role': 'user',
            'content': [
              {'type': 'video'},
            ],
          },
        ],
        tokenizer: tokenizer,
        config: config,
        videos: [
          OmniVideoLayout(
            timestamps: [0.25, 1.0],
            groupStarts: [0, 0.75],
            tokensPerGroup: 3,
          ),
        ],
      );
      expect(
        silent.spans.map((s) => s.kind),
        everyElement(OmniMediaKind.video),
      );
    });

    test('formats seconds like Python, ties to even', () {
      for (final (x, text) in [
        (0.25, '0.2'),
        (0.75, '0.8'),
        (1.25, '1.2'),
        (0.15, '0.1'),
        (2.05, '2.0'),
        (0.5, '0.5'),
        (10.45, '10.4'),
      ]) {
        expect(formatOmniSeconds(x), text, reason: '$x');
      }
    });
  });

  test('video frames are sampled like numpy.linspace', () {
    final p = Lfm3VideoProcessor();
    expect(p.sampleIndices(300, 30), [
      0, 16, 31, 47, 63, 79, 94, 110, 126, 142, //
      157, 173, 189, 205, 220, 236, 252, 268, 283, 299,
    ]);
    expect(p.sampleIndices(91, 29.97), [0, 18, 36, 54, 72, 90]);
    expect(p.sampleIndices(5, 25), [0, 1, 3, 4]);
    expect(p.sampleIndices(48, 24), [0, 16, 31, 47]);
    expect(p.sampleIndices(1000, 60).length, 33);
  });

  test('video frames pair into groups with mean timestamps', () {
    final p = Lfm3VideoProcessor(maxPixels: 4 * 64 * 64);
    final frames = [
      for (var i = 0; i < 3; i++)
        Uint8List(64 * 64 * 3)..fillRange(0, 64 * 64 * 3, i * 100),
    ];
    final video = p.prepareFrames(
      frames,
      height: 64,
      width: 64,
      frameIndices: [0, 15, 30],
      sourceFps: 30,
    );
    expect(video.groups.length, 2);
    expect(video.timestamps, [0.25, 1.0]);
    expect(video.groupStarts, [0.0, 1.0]);
    final g = video.groups.first;
    final area = g.patchSize * g.patchSize;
    // Channel 0 of the first patch: frame 0 slot, then frame 1 slot.
    expect(g.pixelValues[0], -1.0);
    expect(g.pixelValues[area], closeTo((100 - 127.5) / 127.5, 1e-6));
    // The odd last frame is repeated into its pair.
    final last = video.groups.last;
    expect(last.pixelValues[0], last.pixelValues[area]);
  });

  test('Omni config parses the shared towers with tokenizer span IDs', () {
    final json =
        jsonDecode(File('test/fixtures/omni/config.json').readAsStringSync())
            as Map<String, Object?>;
    final markers = oracle['markers'] as Map;
    final c = Lfm3VlConfig.fromOmniJson(
      json,
      visionStartTokenId: markers['<|vision_start|>'] as int,
      visionEndTokenId: markers['<|vision_end|>'] as int,
    );
    expect(c.text.hiddenSize, 2048);
    expect(c.text.numLayers, 24);
    expect(c.vision.hiddenSize, 1024);
    expect(c.vision.numLayers, 14);
    expect(c.imageTokenId, markers['<|image|>']);
    expect(c.videoTokenId, markers['<|video|>']);
    expect(c.projectorInputSize, 4096);
    expect(() => Lfm3VlConfig.fromJson(json), throwsFormatException);
    expect(
      () => Lfm3VlConfig.fromOmniJson(
        json,
        visionStartTokenId: c.imageTokenId,
        visionEndTokenId: markers['<|vision_end|>'] as int,
      ),
      throwsFormatException,
    );
    final vl =
        jsonDecode(File('test/fixtures/vision/config.json').readAsStringSync())
            as Map<String, Object?>;
    expect(
      () => Lfm3VlConfig.fromOmniJson(
        vl,
        visionStartTokenId: 1,
        visionEndTokenId: 2,
      ),
      throwsFormatException,
    );
  });

  group('mmproj metadata', () {
    late Directory temp;
    setUp(() => temp = Directory.systemTemp.createTempSync('dianoia-omni-'));
    tearDown(() => temp.deleteSync(recursive: true));

    GgufFile open(Map<String, Object> metadata) {
      final path = '${temp.path}/${metadata.length}.gguf';
      writeTestGguf(
        path,
        {'general.architecture': 'clip', ...metadata},
        [
          ('x', [32], GgufType.f32, Uint8List(128)),
        ],
      );
      return GgufFile.open(path);
    }

    test('variant follows the vision projector', () {
      final audio = open(lfm3AudioSamplingMetadata);
      final omni = open({'clip.vision.projector_type': 'lfm3vl'});
      addTearDown(audio.close);
      addTearDown(omni.close);
      expect(Lfm3SpeechVariant.of(audio), Lfm3SpeechVariant.audio);
      expect(Lfm3SpeechVariant.of(omni), Lfm3SpeechVariant.omni);
    });

    test('sampling defaults come from the published metadata', () {
      final audio = open(lfm3AudioSamplingMetadata);
      final omni = open({
        'clip.gen.audio.sampling.temp': 0.4,
        'clip.gen.audio.sampling.penalty_repeat': 1.2,
        'clip.gen.audio.sampling.penalty_last_n': 64,
      });
      final broken = open({'clip.gen.audio.sampling.temp': 0.4});
      addTearDown(audio.close);
      addTearDown(omni.close);
      addTearDown(broken.close);
      final a = Lfm3SpeechSampling.fromGguf(audio);
      expect(
        [a.temperature, a.topK, a.repetitionPenalty, a.repetitionWindow],
        [1.0, 128, 1.1, 4096],
      );
      final o = Lfm3SpeechSampling.fromGguf(omni);
      expect(
        [o.temperature, o.topK, o.repetitionPenalty, o.repetitionWindow],
        [0.4, null, 1.2, 64],
      );
      expect(() => Lfm3SpeechSampling.fromGguf(broken), throwsFormatException);
    });
  });

  test('Q4_0 blocks widen exactly, low nibbles first', () {
    final block = Uint8List(18)
      ..[0] = 0x00
      ..[1] = 0x38; // half 0.5
    for (var k = 0; k < 16; k++) {
      block[2 + k] = k | (15 - k) << 4;
    }
    final values = dequantizeQ4_0(block, 32);
    for (var k = 0; k < 16; k++) {
      expect(values[k], 0.5 * (k - 8));
      expect(values[k + 16], 0.5 * (7 - k));
    }
    expect(() => dequantizeQ4_0(block, 64), throwsArgumentError);
  });

  test('vision weights load from an mmproj like the imported bundle', () async {
    if (!NativeKernels.isAvailable) {
      markTestSkipped('requires native kernels');
      return;
    }
    final root = Directory.systemTemp.createTempSync('dianoia-omni-vision-');
    addTearDown(() => root.deleteSync(recursive: true));
    await prepareVisionFixture(root, hiddenSize: 256);
    final model = '${root.path}/model';
    final projector = '${root.path}/vision.gguf';
    writeVisionGgufFixture(model, '${root.path}/language.gguf', projector);
    final config = Lfm3VlConfig.load(model);
    importVlVisionGguf(projector, config, '${root.path}/imported.safetensors');
    final imported = SafetensorsFile.open('${root.path}/imported.safetensors');
    addTearDown(imported.close);
    final gguf = GgufFile.open(projector);
    addTearDown(gguf.close);
    final weights = readVisionGgufWeights(gguf, config);
    final shapes = lfm3VisionTensorShapes(config);
    expect(weights.keys.toSet(), shapes.keys.toSet());
    for (final e in shapes.entries) {
      expect(weights[e.key], imported.readF32(e.key, e.value), reason: e.key);
    }
    expect(
      () => Lfm3VisionEncoder.fromWeights(
        config,
        {...weights}..remove(shapes.keys.first),
      ),
      throwsArgumentError,
    );
  });
}
