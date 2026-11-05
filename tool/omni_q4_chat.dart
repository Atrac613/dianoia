/// One LFM3-Omni Q4_0 turn mixing images, audio clips and text, answered in
/// text, speech or interleaved text and speech.
/// Usage: `dart run tool/omni_q4_chat.dart <prepared-dir>`
///        `[--image=photo.png]... [--audio=clip.f32]... [--text=question]`
///        `[--system=prompt] [--mode=text|audio|mixed] [--out=reply.wav]`
///        `[--gpu] [--max-image-tokens=N] [--max-new-tokens=N] [--seed=0]`
/// Audio clips are mono 16 kHz f32le PCM. Parts are ordered: images, audio,
/// then text. Speech modes need --out.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';

Float32List _pcm(String path) {
  final bytes = File(path).readAsBytesSync();
  if (bytes.isEmpty || bytes.length % 4 != 0 || bytes.length > 30 * 64000) {
    throw ArgumentError('$path: expected up to 30 s of mono 16 kHz f32le');
  }
  final data = ByteData.sublistView(bytes);
  return Float32List.fromList([
    for (var i = 0; i < bytes.length ~/ 4; i++)
      data.getFloat32(i * 4, Endian.little),
  ]);
}

Future<void> main(List<String> args) async {
  final images = <String>[], clips = <String>[], positional = <String>[];
  final options = <String, String>{};
  var gpu = false;
  for (final arg in args) {
    if (arg == '--gpu') {
      gpu = true;
    } else if (arg.startsWith('--image=')) {
      images.add(arg.substring(8));
    } else if (arg.startsWith('--audio=')) {
      clips.add(arg.substring(8));
    } else if (arg.startsWith('--') && arg.contains('=')) {
      final at = arg.indexOf('=');
      options[arg.substring(2, at)] = arg.substring(at + 1);
    } else {
      positional.add(arg);
    }
  }
  const known = {
    'text',
    'system',
    'mode',
    'out',
    'max-image-tokens',
    'max-new-tokens',
    'seed',
  };
  final mode = AudioReplyMode.values.byName(options['mode'] ?? 'text');
  final text = options['text'];
  if (positional.length != 1 ||
      options.keys.any((k) => !known.contains(k)) ||
      (images.isEmpty && clips.isEmpty && text == null) ||
      (mode != AudioReplyMode.text && options['out'] == null)) {
    throw ArgumentError(
      'usage: omni_q4_chat.dart <prepared-dir> [--image=..] [--audio=..] '
      '[--text=..] [--system=..] [--mode=text|audio|mixed] [--out=reply.wav] '
      '[--gpu]',
    );
  }
  final dir = positional.single;
  NativeKernels.load(null);
  final timings = <String, double>{};
  final clock = Stopwatch()..start();
  void lap(String name) {
    timings[name] = clock.elapsedMicroseconds / 1e6;
    clock.reset();
  }

  final mmproj = speechMmprojIn(dir);
  final tokenizer = Lfm3Tokenizer.load('$dir/language');
  final config = loadOmniVisionConfig(
    '$dir/${Lfm3OmniQ4Source.config.name}',
    tokenizer,
  );
  var processor = Lfm3ImageProcessor.load(dir);
  if (options['max-image-tokens'] case final n?) {
    processor = processor.withMaxImageTokens(int.parse(n));
  }
  final imageRows = <Float32List>[], imageTokens = <int>[];
  if (images.isNotEmpty) {
    final vision = loadVisionEncoderGguf(mmproj, config);
    lap('vision_load_seconds');
    for (final path in images) {
      final rgb = await readImageRgb(path);
      final input = processor.prepareRgb(
        rgb.pixels,
        height: rgb.height,
        width: rgb.width,
      );
      imageRows.add(vision.encode(input));
      imageTokens.add(input.imageTokens);
    }
    vision.close();
    lap('vision_encode_seconds');
  }
  final audioRows = <Float32List>[], audioTokens = <int>[];
  if (clips.isNotEmpty) {
    final encoder = Lfm3AudioEncoder.load(mmproj);
    try {
      if (encoder.config.variant != Lfm3SpeechVariant.omni) {
        throw StateError('$dir is not an LFM3-Omni directory');
      }
      final frontend = Lfm3AudioFrontend();
      for (final path in clips) {
        final features = frontend.extract(_pcm(path), samplingRate: 16000);
        audioRows.add(encoder.encode(features));
        audioTokens.add(features.encoderTokens);
      }
    } finally {
      encoder.close();
    }
    lap('audio_encode_seconds');
  }
  final messages = [
    if (options['system'] case final system?)
      {'role': 'system', 'content': system},
    {
      'role': 'user',
      'content': [
        for (final _ in images) {'type': 'image'},
        for (final _ in clips) {'type': 'audio'},
        if (text != null) {'type': 'text', 'text': text},
      ],
    },
  ];
  final prompt = prepareOmniPrompt(
    messages,
    tokenizer: tokenizer,
    config: config,
    imageTokenCounts: imageTokens,
    audioTokenCounts: audioTokens,
    mode: mode,
  );
  final maxNew = int.parse(options['max-new-tokens'] ?? '768');
  final generator = mode == AudioReplyMode.text
      ? null
      : Lfm3AudioGenerator.load(mmproj);
  final model = await Lfm3Model.load(
    '$dir/language',
    maxSeq: prompt.ids.length + maxNew + 1,
    weights: WeightFormat.q4,
    q8CachePath: '$dir/language/lfm3-q4.cache',
    gpu: gpu,
  );
  late final SpeechReply reply;
  try {
    lap('model_load_seconds');
    await prompt.prefill(model, images: imageRows, audio: audioRows);
    lap('prefill_seconds');
    reply = await decodeSpeechReply(
      model: model,
      tokenizer: tokenizer,
      generator: generator,
      variant: Lfm3SpeechVariant.omni,
      mode: mode,
      maxSlots: maxNew,
      seed: int.parse(options['seed'] ?? '0'),
      feed: (id, embedding, {required logits}) => model.forward(
        [id],
        computeLogits: logits,
        embeddingOverrides: {0: ?embedding},
      ),
    );
    lap('decode_seconds');
  } finally {
    generator?.close();
    await model.close();
  }
  if (options['out'] case final out? when reply.samples.isNotEmpty) {
    File(out).writeAsBytesSync(
      encodeSpeechWav(reply.samples, sampleRate: reply.sampleRate),
    );
  }
  stdout.writeln(
    const JsonEncoder.withIndent('  ').convert({
      'model': 'LFM3-Omni-2B-Q4_0',
      'source_revision': Lfm3OmniQ4Source.revision,
      'backend': gpu ? 'gpu' : 'cpu',
      'mode': mode.name,
      'image_tokens': imageTokens,
      'audio_tokens': audioTokens,
      'prompt_tokens': prompt.ids.length,
      'text': reply.text,
      'speech_seconds': reply.seconds,
      'generated_slots': reply.generatedSlots,
      'output': options['out'],
      ...timings,
    }),
  );
}
