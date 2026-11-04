/// Voice conversation without opening audio devices. Input: mono 16 kHz f32le.
/// `dart run tool/audio_q4_converse.dart <prepared-dir> <out-dir> <clip.f32>... [--gpu]`
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:dianoia/lab.dart';

Future<void> main(List<String> args) async {
  final gpu = args.contains('--gpu');
  final pos = args.where((a) => a != '--gpu').toList();
  if (pos.length < 3) {
    throw ArgumentError(
      'usage: audio_q4_converse.dart <prepared-dir> <out-dir> <clip.f32>... [--gpu]',
    );
  }
  final dir = pos[0], out = Directory(pos[1])..createSync(recursive: true);
  NativeKernels.load(null);
  final generator = Lfm3AudioGenerator.load(
    '$dir/${Lfm3AudioQ4Source.audio.name}',
  );
  final encoder = Lfm3AudioEncoder.load('$dir/${Lfm3AudioQ4Source.audio.name}');
  Lfm3Model? model;
  AudioConversationSession? session;
  try {
    model = await Lfm3Model.load(
      '$dir/language',
      maxSeq: 2048,
      weights: WeightFormat.q4,
      q8CachePath: '$dir/language/lfm3-q4.cache',
      gpu: gpu,
    );
    session = AudioConversationSession(
      model,
      Lfm3Tokenizer.load('$dir/language'),
      generator,
    );
    for (var i = 2; i < pos.length; i++) {
      final data = ByteData.sublistView(File(pos[i]).readAsBytesSync());
      if (data.lengthInBytes % 4 != 0 ||
          data.lengthInBytes == 0 ||
          data.lengthInBytes > 30 * 16000 * 4) {
        throw ArgumentError(
          'expected at most 30 seconds of mono 16 kHz f32le PCM',
        );
      }
      final pcm = Float32List(data.lengthInBytes ~/ 4);
      for (var j = 0; j < pcm.length; j++) {
        pcm[j] = data.getFloat32(j * 4, Endian.little);
      }
      final frontend = Lfm3AudioFrontend().startStream(),
          stream = encoder.startStream();
      final watch = Stopwatch()..start();
      await session.begin();
      try {
        for (var at = 0; at < pcm.length; at += 1280) {
          await session.addProjected(
            stream.add(
              frontend.add(
                pcm.sublist(at, (at + 1280).clamp(0, pcm.length)),
                samplingRate: 16000,
              ),
            ),
          );
        }
        await session.addProjected(stream.add(frontend.finish()));
        await session.addProjected(stream.finish());
        final reply = await session.finish();
        final path = '${out.path}/reply-${i - 1}.wav';
        File(path).writeAsBytesSync(
          encodeSpeechWav(reply.samples, sampleRate: reply.sampleRate),
        );
        final json = const JsonEncoder.withIndent('  ').convert({
          'turn': session.turns,
          'text': reply.text,
          'frames': reply.frames,
          'seconds': reply.seconds,
          'generated_slots': reply.generatedSlots,
          'history_tokens': reply.historyTokens,
          'retained_embedding_bytes': session.retainedEmbeddingBytes,
          'wall_seconds': watch.elapsedMicroseconds / 1e6,
          'wav': File(path).absolute.path,
        });
        File('${out.path}/reply-${i - 1}.json').writeAsStringSync(json);
        stdout.writeln(json);
      } finally {
        stream.close();
      }
    }
  } finally {
    session?.close();
    generator.close();
    encoder.close();
    await model?.close();
  }
}
