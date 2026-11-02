@Timeout(Duration(minutes: 3))
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia_example/src/audio_worker.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final dir = Platform.environment['DIANOIA_AUDIO_CONVERSATION_DIR'];
  final first = Platform.environment['DIANOIA_AUDIO_CONVERSATION_FIRST'];
  final second = Platform.environment['DIANOIA_AUDIO_CONVERSATION_FOLLOWUP'];
  test(
    'voice worker discards unaccepted replies, remembers accepted turns and resets',
    () async {
      final backend = AudioIsolateBackend(useGpu: false);
      Future<void> record(String path) async {
        final bytes = ByteData.sublistView(File(path).readAsBytesSync());
        final pcm = Float32List(bytes.lengthInBytes ~/ 4);
        for (var i = 0; i < pcm.length; i++) {
          pcm[i] = bytes.getFloat32(i * 4, Endian.little);
        }
        await backend.begin(null, conversation: true);
        for (var at = 0; at < pcm.length; at += 1280) {
          await backend.addPcm(
            pcm.sublist(at, (at + 1280).clamp(0, pcm.length)),
          );
        }
      }

      try {
        await backend.load(dir!);
        await record(first!);
        final discarded = await backend.finishConversation((_) {});
        expect(discarded.text.toLowerCase(), contains('four'));
        await backend.cancel();
        await expectLater(backend.acceptConversationReply(), throwsStateError);
        await record(first);
        final progress = <String>[];
        final accepted = await backend.finishConversation(progress.add);
        expect(accepted.turn, 1);
        expect(progress.last.trim(), accepted.text);
        expect(String.fromCharCodes(accepted.wav.sublist(0, 4)), 'RIFF');
        await backend.acceptConversationReply();
        await record(second!);
        final followup = await backend.finishConversation((_) {});
        expect(followup.turn, 2);
        expect(followup.text.toLowerCase(), contains('seven'));
        await backend.acceptConversationReply();
        await record(first);
        final failed = expectLater(
          backend.finishConversation((_) {}),
          throwsStateError,
        );
        await Future<void>.delayed(Duration.zero);
        await backend.cancel();
        await failed;
        await backend.newConversation();
        await record(first);
        final reset = await backend.finishConversation((_) {});
        expect(reset.turn, 1);
        expect(reset.text.toLowerCase(), contains('four'));
        await backend.acceptConversationReply();
      } finally {
        await backend.close();
      }
    },
    skip: dir == null || first == null || second == null
        ? 'set DIANOIA_AUDIO_CONVERSATION_DIR/FIRST/FOLLOWUP'
        : false,
  );
}
