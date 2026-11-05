import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia_example/src/audio_worker.dart';
import 'package:dianoia_example/src/speech_model.dart';
import 'package:flutter_test/flutter_test.dart';

Float32List _pcm(String path) {
  final data = ByteData.sublistView(File(path).readAsBytesSync());
  return Float32List.fromList([
    for (var i = 0; i < data.lengthInBytes ~/ 4; i++)
      data.getFloat32(i * 4, Endian.little),
  ]);
}

/// A prepared research folder or an installed Omni bundle, a PNG or JPEG
/// photo, and mono 16 kHz f32le PCM of an English question about it.
void main() {
  final directory = Platform.environment['DIANOIA_OMNI_EXAMPLE_DIR'];
  final image = Platform.environment['DIANOIA_OMNI_EXAMPLE_IMAGE'];
  final question = Platform.environment['DIANOIA_OMNI_EXAMPLE_PCM'];
  final gpu = Platform.environment['DIANOIA_OMNI_EXAMPLE_GPU'] == '1';
  test(
    'Omni worker answers typed and spoken photo questions and transcribes',
    () async {
      final backend = AudioIsolateBackend(model: SpeechModel.omni, useGpu: gpu);
      try {
        expect(await backend.load(directory!), gpu ? 'GPU' : 'CPU');
        final photo = File(image!).readAsBytesSync();
        final updates = <String>[];
        final typed = await backend.askAboutPhoto(
          photo,
          text: 'What colors are the shapes? Answer briefly.',
          spoken: false,
          speak: false,
          onText: updates.add,
        );
        expect(typed.text.toLowerCase(), contains('red'));
        expect(typed.wav, isEmpty);
        expect(updates.last, typed.text);

        final pcm = _pcm(question!);
        await backend.begin(null, question: true);
        for (var at = 0; at < pcm.length; at += 1280) {
          await backend.addPcm(
            pcm.sublist(at, (at + 1280).clamp(0, pcm.length)),
          );
        }
        final spoken = await backend.askAboutPhoto(
          photo,
          spoken: true,
          speak: true,
          onText: (_) {},
        );
        expect(spoken.text, isNotEmpty);
        expect(String.fromCharCodes(spoken.wav.sublist(0, 4)), 'RIFF');

        await backend.begin('English');
        for (var at = 0; at < pcm.length; at += 1280) {
          await backend.addPcm(
            pcm.sublist(at, (at + 1280).clamp(0, pcm.length)),
          );
        }
        expect((await backend.finish()).toLowerCase(), contains('shapes'));
        await expectLater(
          backend.begin('English', live: true),
          throwsStateError,
        );
      } finally {
        await backend.close();
      }
    },
    skip: directory == null || image == null || question == null
        ? 'set DIANOIA_OMNI_EXAMPLE_DIR/IMAGE/PCM'
        : false,
    timeout: const Timeout(Duration(minutes: 10)),
  );
}
