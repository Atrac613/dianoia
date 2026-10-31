import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/synthetic_vision.dart';

const _specials = [
  '<|startoftext|>',
  '<|im_start|>',
  '<|im_end|>',
  '<think>',
  '</think>',
  '<|vision_start|>',
  '<|vision_end|>',
  '<|image|>',
  '<|video|>',
];

void _tokenizer(String dir) {
  final visible = [
    for (var i = 33; i <= 126; i++) i,
    for (var i = 161; i <= 172; i++) i,
    for (var i = 174; i <= 255; i++) i,
  ];
  var extra = 256;
  final vocab = <String, int>{};
  for (var byte = 0; byte < 256; byte++) {
    vocab[String.fromCharCode(visible.contains(byte) ? byte : extra++)] = byte;
  }
  File('$dir/tokenizer.json').writeAsStringSync(
    jsonEncode({
      'model': {
        'type': 'BPE',
        'ignore_merges': true,
        'vocab': vocab,
        'merges': [],
      },
      'added_tokens': [
        for (var i = 0; i < _specials.length; i++)
          {'content': _specials[i], 'id': 256 + i},
      ],
    }),
  );
}

Map<String, Object?> _config({int? hidden}) {
  final json =
      jsonDecode(File('test/fixtures/vision/config.json').readAsStringSync())
          as Map<String, Object?>;
  json.addAll({
    'vision_start_token_id': 261,
    'vision_end_token_id': 262,
    'image_token_id': 263,
    'video_token_id': 264,
  });
  (json['text_config'] as Map<String, Object?>).addAll({
    'vocab_size': 272,
    'eos_token_id': 258,
  });
  if (hidden != null) {
    (json['text_config'] as Map<String, Object?>)['hidden_size'] = hidden;
  }
  return json;
}

List<Map<String, Object?>> _messages(Object? input) => [
  for (final message in input as List) (message as Map).cast<String, Object?>(),
];

void main() {
  late Directory root;
  late Lfm3Tokenizer tokenizer;
  late Lfm3VlConfig config;
  setUp(() {
    root = Directory.systemTemp.createTempSync('dianoia_vl_prompt');
    _tokenizer(root.path);
    tokenizer = Lfm3Tokenizer.load(root.path);
    config = Lfm3VlConfig.fromJson(_config());
  });
  tearDown(() => root.deleteSync(recursive: true));
  final oracle =
      jsonDecode(
            File('test/fixtures/vision/prompt_oracle.json').readAsStringSync(),
          )
          as Map<String, Object?>;
  for (final item in oracle['cases'] as List) {
    final c = item as Map<String, Object?>;
    test('official VL template and processor: ${c['name']}', () {
      final messages = _messages(c['messages']);
      final tools = c['tools'] == null ? null : _messages(c['tools']);
      final thinking = c['enable_thinking'] as bool;
      final preserve = c['preserve_thinking'] == true;
      expect(
        renderVisionChat(
          messages,
          tools: tools,
          enableThinking: thinking,
          preserveThinking: preserve,
        ),
        c['prompt'],
      );
      final counts = (c['counts'] as List).cast<int>();
      final prompt = prepareVisionPrompt(
        messages,
        tokenizer: tokenizer,
        config: config,
        imageTokenCounts: counts,
        tools: tools,
        enableThinking: thinking,
        preserveThinking: preserve,
      );
      expect(prompt.ids, tokenizer.encode(c['expanded'] as String));
      expect(prompt.images.map((s) => s.count), counts);
      for (final span in prompt.images) {
        expect(
          prompt.ids.sublist(span.start, span.start + span.count),
          List.filled(span.count, config.imageTokenId),
        );
        expect(prompt.ids[span.start - 1], config.visionStartTokenId);
        expect(prompt.ids[span.start + span.count], config.visionEndTokenId);
      }
    });
  }

  test(
    'count mismatch, video, raw placeholders and tokenizer mismatch fail',
    () {
      final message = [
        {
          'role': 'user',
          'content': [
            {'type': 'image'},
          ],
        },
      ];
      for (final counts in <List<int>>[
        [],
        [0],
        [-1],
        [1, 2],
      ]) {
        expect(
          () => prepareVisionPrompt(
            message,
            tokenizer: tokenizer,
            config: config,
            imageTokenCounts: counts,
          ),
          throwsArgumentError,
        );
      }
      for (final content in [
        '<|image|>',
        '<|vision_start|><|image|><|vision_end|>',
      ]) {
        expect(
          () => prepareVisionPrompt(
            [
              {'role': 'user', 'content': content},
            ],
            tokenizer: tokenizer,
            config: config,
            imageTokenCounts: [],
          ),
          throwsArgumentError,
        );
      }
      for (final type in ['video', 'image_url']) {
        expect(
          () => renderVisionChat([
            {
              'role': 'user',
              'content': {'type': type},
            },
          ]),
          throwsUnsupportedError,
        );
      }
      final nested = [
        {
          'role': 'tool',
          'content': [
            [
              {'type': 'image'},
            ],
          ],
        },
      ];
      expect(
        renderVisionChat(nested),
        renderVisionChat([
          {
            'role': 'tool',
            'content': [
              {'type': 'image'},
            ],
          },
        ]),
      );
      expect(
        prepareVisionPrompt(
          nested,
          tokenizer: tokenizer,
          config: config,
          imageTokenCounts: [1],
        ).images,
        hasLength(1),
      );
      final wrong = Lfm3VlConfig.fromJson(_config()..['image_token_id'] = 265);
      expect(
        () => prepareVisionPrompt(
          message,
          tokenizer: tokenizer,
          config: wrong,
          imageTokenCounts: [1],
        ),
        throwsFormatException,
      );
    },
  );

  final modelDir = Platform.environment['DIANOIA_LFM3_DIR'];
  for (final gpu in [false, true]) {
    test(
      'encoder rows reach language prefill (${gpu ? 'GPU' : 'CPU'})',
      () async {
        NativeKernels.load(null, threads: 2);
        final model = await Lfm3Model.load(
          modelDir!,
          weights: WeightFormat.q4,
          q8CachePath: '$modelDir/lfm3-q4.cache',
          maxSeq: 256,
          maxChunk: 2,
          gpu: gpu,
        );
        addTearDown(model.close);
        final reference =
            jsonDecode(
                  File(
                    'test/fixtures/vision/encoder_oracle.json',
                  ).readAsStringSync(),
                )
                as Map<String, Object?>;
        final json = reference['config'] as Map<String, Object?>;
        json.addAll({
          'vision_start_token_id': 261,
          'vision_end_token_id': 262,
          'image_token_id': 263,
          'video_token_id': 264,
        });
        (json['text_config'] as Map<String, Object?>).addAll({
          'hidden_size': model.config.hiddenSize,
          'vocab_size': 272,
          'eos_token_id': 258,
        });
        File('${root.path}/config.json').writeAsStringSync(jsonEncode(json));
        final shapes = reference['shapes'] as Map<String, Object?>;
        shapes['model.multi_modal_projector.linear_2.weight'] = [
          model.config.hiddenSize,
          12,
        ];
        shapes['model.multi_modal_projector.linear_2.bias'] = [
          model.config.hiddenSize,
        ];
        writeSyntheticVisionWeights('${root.path}/model.safetensors', shapes);
        final encoder = Lfm3VisionEncoder.load(root.path);
        final images = [syntheticVisionImage(2, 2), syntheticVisionImage(2, 4)];
        final features = [for (final image in images) encoder.encode(image)];
        final prompt = prepareVisionPrompt(
          [
            {
              'role': 'user',
              'content': [
                {'type': 'image'},
                {'type': 'text', 'text': 'Compare'},
                {'type': 'image'},
              ],
            },
          ],
          tokenizer: tokenizer,
          config: encoder.config,
          imageTokenCounts: [for (final image in images) image.imageTokens],
        );
        await model.forward([10]);
        final before = model.history;
        await expectLater(
          prompt.prefill(model, [Float32List(1)]),
          throwsArgumentError,
        );
        expect(model.history, before);
        final bad = [for (final f in features) Float32List.fromList(f)];
        bad[0][0] = double.nan;
        await expectLater(prompt.prefill(model, bad), throwsArgumentError);
        expect(model.history, before);
        await prompt.prefill(model, features);
        final actual = Float32List.fromList(model.logits);
        expect(model.length, prompt.ids.length);
        for (final span in prompt.images) {
          expect(
            model.history.sublist(span.start, span.start + span.count),
            List.filled(span.count, -1),
          );
        }
        // Independent manual placement at every image token in the expanded IDs.
        final rows = <Float32List>[];
        for (final values in features) {
          for (var at = 0; at < values.length; at += model.config.hiddenSize) {
            rows.add(
              Float32List.fromList(
                values.sublist(at, at + model.config.hiddenSize),
              ),
            );
          }
        }
        var row = 0;
        final overrides = {
          for (var i = 0; i < prompt.ids.length; i++)
            if (prompt.ids[i] == encoder.config.imageTokenId) i: rows[row++],
        };
        await model.reset();
        await model.forward(prompt.ids, embeddingOverrides: overrides);
        expect(model.logits, orderedEquals(actual));
      },
      skip: modelDir == null ? 'set DIANOIA_LFM3_DIR to a Q4 bundle' : false,
    );
  }
}
