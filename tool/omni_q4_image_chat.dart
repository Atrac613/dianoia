/// Research image question answering on a prepared LFM3-Omni Q4_0 directory.
/// The vision tower runs on the CPU in float32 from the dequantized mmproj
/// blocks; the language backbone runs with Q4 weights on the CPU or GPU.
/// Usage: `dart run tool/omni_q4_image_chat.dart <prepared-dir> <image>`
///        `<question> [--gpu] [--max-image-tokens=256] [--max-new-tokens=256]`
library;

import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';

Future<void> main(List<String> args) async {
  final positional = <String>[];
  var gpu = false;
  int? maxImageTokens;
  var maxNewTokens = 256;
  for (final arg in args) {
    if (arg == '--gpu') {
      gpu = true;
    } else if (arg.startsWith('--max-image-tokens=')) {
      maxImageTokens = int.parse(arg.substring('--max-image-tokens='.length));
    } else if (arg.startsWith('--max-new-tokens=')) {
      maxNewTokens = int.parse(arg.substring('--max-new-tokens='.length));
    } else if (arg.startsWith('--')) {
      throw ArgumentError('unknown option $arg');
    } else {
      positional.add(arg);
    }
  }
  if (positional.length != 3 || maxNewTokens < 1) {
    throw ArgumentError(
      'usage: omni_q4_image_chat.dart <prepared-dir> <image> <question> '
      '[--gpu] [--max-image-tokens=N] [--max-new-tokens=N]',
    );
  }
  final [dir, imagePath, question] = positional;
  NativeKernels.load(null);
  final clock = Stopwatch()..start();
  double lap() {
    final seconds = clock.elapsedMicroseconds / 1e6;
    clock.reset();
    return seconds;
  }

  final tokenizer = Lfm3Tokenizer.load('$dir/language');
  final config = loadOmniVisionConfig(
    '$dir/${Lfm3OmniQ4Source.config.name}',
    tokenizer,
  );
  var processor = Lfm3ImageProcessor.load(dir);
  if (maxImageTokens != null) {
    processor = processor.withMaxImageTokens(maxImageTokens);
  }
  final image = await readImageRgb(imagePath);
  final input = processor.prepareRgb(
    image.pixels,
    height: image.height,
    width: image.width,
  );
  final preprocessSeconds = lap();
  final encoder = loadVisionEncoderGguf(speechMmprojIn(dir), config);
  final visionLoadSeconds = lap();
  final rows = encoder.encode(input);
  final visionSeconds = lap();
  final prompt = prepareVisionPrompt(
    [
      {
        'role': 'user',
        'content': [
          {'type': 'image'},
          {'type': 'text', 'text': question},
        ],
      },
    ],
    tokenizer: tokenizer,
    config: config,
    imageTokenCounts: [input.imageTokens],
    // LFM3-Omni is a non-reasoning model: its replies open with an empty
    // <think></think>, as the official template's default renders it.
    enableThinking: false,
  );
  final model = await Lfm3Model.load(
    '$dir/language',
    maxSeq: prompt.ids.length + maxNewTokens + 1,
    weights: WeightFormat.q4,
    q8CachePath: '$dir/language/lfm3-q4.cache',
    gpu: gpu,
  );
  final suppressed = {
    for (final marker in [
      '<|image|>',
      '<|video|>',
      '<|audio|>',
      '<|audio_out|>',
      '<|audio_start|>',
      '<|mixed_start|>',
    ])
      tokenizer.addedTokenId(marker)!,
  };
  final stops = {
    tokenizer.addedTokenId('<|im_end|>')!,
    tokenizer.addedTokenId('<|text_end|>')!,
  };
  final generated = <int>[];
  var stopReason = 'length';
  late final double prefillSeconds;
  try {
    lap();
    await prompt.prefill(model, [rows]);
    prefillSeconds = lap();
    while (generated.length < maxNewTokens) {
      final logits = model.logits;
      var token = -1, best = double.negativeInfinity;
      for (var i = 0; i < logits.length; i++) {
        if (!suppressed.contains(i) && logits[i] > best) {
          token = i;
          best = logits[i];
        }
      }
      if (token < 0) throw StateError('no finite text logits');
      if (stops.contains(token)) {
        stopReason = 'eos';
        break;
      }
      generated.add(token);
      await model.step(token);
    }
  } finally {
    await model.close();
  }
  stdout.writeln(
    const JsonEncoder.withIndent('  ').convert({
      'model': 'LFM3-Omni-2B-Q4_0',
      'source_revision': Lfm3OmniQ4Source.revision,
      'backend': gpu ? 'gpu' : 'cpu',
      'image': {'width': image.width, 'height': image.height},
      'resized': {'width': input.width, 'height': input.height},
      'image_tokens': input.imageTokens,
      'prompt_tokens': prompt.ids.length,
      'generated_ids': generated,
      'text': tokenizer.decode(generated).trim(),
      'stop_reason': stopReason,
      'preprocess_seconds': preprocessSeconds,
      'vision_load_seconds': visionLoadSeconds,
      'vision_encode_seconds': visionSeconds,
      'language_prefill_seconds': prefillSeconds,
      'decode_seconds': lap(),
    }),
  );
}
