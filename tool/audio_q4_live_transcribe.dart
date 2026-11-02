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
  final dir = pos[0],
      bytes = ByteData.sublistView(File(pos[1]).readAsBytesSync());
  if (bytes.lengthInBytes == 0 ||
      bytes.lengthInBytes % 4 != 0 ||
      bytes.lengthInBytes > 30 * 16000 * 4) {
    throw ArgumentError('expected at most 30 seconds of f32le PCM');
  }
  final pcm = Float32List(bytes.lengthInBytes ~/ 4);
  for (var i = 0; i < pcm.length; i++) {
    pcm[i] = bytes.getFloat32(i * 4, Endian.little);
  }
  NativeKernels.load(null);
  final encoder = Lfm3AudioEncoder.load('$dir/${Lfm3AudioQ4Source.audio.name}');
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
    for (var at = 0; at < pcm.length; at += 1280) {
      fed = (at + 1280).clamp(0, pcm.length);
      await live.addPcm(pcm.sublist(at, fed));
    }
    final text = await live.finish();
    stdout.writeln(
      jsonEncode({
        'event': 'done',
        'text': text,
        'blocks': block,
        'input_seconds': fed / 16000,
        'simulation': 'unpaced file',
      }),
    );
  } finally {
    live?.close();
    encoder.close();
    await model?.close();
  }
}
