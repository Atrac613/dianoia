/// Unpaced file simulation of English LiveASR; no audio devices are opened.
/// `dart run tool/audio_q4_live_transcribe.dart <prepared-dir> <16k-mono.f32> [--gpu]`
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:dianoia/lab.dart';

Future<void> main(List<String> args) async {
  final gpu = args.contains('--gpu'),
      pos = args.where((a) => a != '--gpu').toList();
  if (pos.length != 2) {
    throw ArgumentError(
      'usage: audio_q4_live_transcribe.dart <prepared-dir> <16k-mono.f32> [--gpu]',
    );
  }
  final dir = pos[0], source = File(pos[1]);
  if (source.lengthSync() == 0 || source.lengthSync() % 4 != 0) {
    throw ArgumentError('expected nonempty mono 16 kHz f32le PCM');
  }
  NativeKernels.load(null);
  final encoder = Lfm3AudioEncoder.load(speechMmprojIn(dir));
  if (encoder.config.variant != Lfm3SpeechVariant.audio) {
    encoder.close();
    // Omni's GGUF publishes no live-ASR template, unlike LFM3-Audio's.
    throw UnsupportedError('LiveASR is defined for LFM3-Audio only');
  }
  RandomAccessFile? reader;
  Lfm3Model? model;
  Lfm3LiveTranscriber? live;
  var fed = 0, block = 0;
  try {
    model = await Lfm3Model.load(
      '$dir/language',
      maxSeq: 2048,
      weights: WeightFormat.q4,
      q8CachePath: '$dir/language/lfm3-q4.cache',
      gpu: gpu,
    );
    live = await Lfm3LiveTranscriber.start(
      model,
      Lfm3Tokenizer.load('$dir/language'),
      encoder,
      onBlock: (b) {
        block++;
        stdout.writeln(
          jsonEncode({
            'event': 'block',
            'block': block,
            'input_seconds': fed / 16000,
            'audio_tokens': b.audioTokens,
            'ids': b.ids,
            'closed': b.closed,
            'final': b.isFinal,
            'context': b.context,
            'restarted': b.restarted,
          }),
        );
      },
      onText: (text) => stdout.writeln(
        jsonEncode({
          'event': 'text',
          'input_seconds': fed / 16000,
          'text': text,
        }),
      ),
    );
    reader = source.openSync();
    while (true) {
      final chunk = reader.readSync(1280 * 4);
      if (chunk.isEmpty) break;
      if (chunk.length % 4 != 0) {
        throw const FormatException('Truncated f32le PCM');
      }
      final bytes = ByteData.sublistView(chunk),
          pcm = Float32List(chunk.length ~/ 4);
      for (var i = 0; i < pcm.length; i++) {
        pcm[i] = bytes.getFloat32(i * 4, Endian.little);
      }
      fed += pcm.length;
      await live.addPcm(pcm);
    }
    final text = await live.finish();
    stdout.writeln(
      jsonEncode({
        'event': 'done',
        'text': text,
        'blocks': block,
        'contexts': live.contexts,
        'encoded_audio_tokens': live.encodedAudioTokens,
        'retained_text_tokens': live.retainedTextTokens,
        'language_context_tokens': model.length,
        'input_seconds': fed / 16000,
        'simulation': 'unpaced file',
      }),
    );
  } finally {
    reader?.closeSync();
    live?.close();
    encoder.close();
    await model?.close();
  }
}
