import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';

import 'package:dianoia_example/src/chat_protocol.dart';
import 'package:dianoia_example/src/chat_worker.dart';
import 'package:dianoia_example/src/example_model.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final bundle = Platform.environment['DIANOIA_VL_BUNDLE'];
  test(
    'default QAD Q4 VL worker loads, replies and closes',
    () async {
      expect(
        (jsonDecode(File('$bundle/vl_bundle.json').readAsStringSync())
            as Map)['weights'],
        'q4',
      );
      expect(
        (jsonDecode(File('$bundle/vl_bundle.json').readAsStringSync())
            as Map)['input_embedding'],
        'q8-cache',
      );
      final events = ReceivePort();
      final exited = ReceivePort();
      final iterator = StreamIterator<dynamic>(events);
      final worker = await Isolate.spawn(runChatWorker, (
        events.sendPort,
        ModelLocation(
          storeDir: '$bundle/unused-store',
          dropDir: bundle!,
          copyDir: bundle,
        ),
        ExampleModel.lfm3Vl,
      ), onExit: exited.sendPort);
      addTearDown(() async {
        worker.kill(priority: Isolate.immediate);
        await iterator.cancel();
        events.close();
        exited.close();
      });

      expect(await iterator.moveNext(), isTrue);
      final commands = (iterator.current as WorkerStarted).commands;
      while (await iterator.moveNext()) {
        final event = iterator.current;
        expect(event, isNot(isA<WorkerFailed>()), reason: '$event');
        if (event is WorkerReady) break;
      }
      commands.send(const SendMessage('Reply with only OK.'));
      ReplyDone? reply;
      while (await iterator.moveNext()) {
        final event = iterator.current;
        expect(event, isNot(isA<ReplyFailed>()), reason: '$event');
        if (event is ReplyDone) {
          reply = event;
          break;
        }
      }
      expect(reply, isNotNull);
      expect(reply!.tokens, greaterThan(0));
      expect(reply.text.trim(), isNotEmpty);
      commands.send(const NewChat());
      final imagePath = Platform.environment['DIANOIA_VL_IMAGE'];
      if (imagePath != null) {
        commands.send(
          SendMessage(
            '画像の文字だけを書き起こしてください。',
            image: ImageAttachment(
              'ocr.png',
              File(imagePath).readAsBytesSync(),
            ),
          ),
        );
        ReplyDone? visualReply;
        while (await iterator.moveNext()) {
          final event = iterator.current;
          expect(event, isNot(isA<ReplyFailed>()));
          if (event is ReplyDone) {
            visualReply = event;
            break;
          }
        }
        expect(visualReply, isNotNull);
        expect(visualReply!.text, '東京 123');
        commands.send(const NewChat());
        commands.send(
          SendMessage(
            'Describe this image.',
            image: ImageAttachment(
              'ocr.png',
              File(imagePath).readAsBytesSync(),
            ),
          ),
        );
        commands.send(const StopReply());
        ReplyFailed? stopped;
        while (await iterator.moveNext()) {
          final event = iterator.current;
          if (event is ReplyFailed) {
            stopped = event;
            break;
          }
          expect(event, isNot(isA<ReplyDone>()));
        }
        expect(stopped?.message, 'Reply stopped.');
      }
      commands.send(const CloseWorker());
      await exited.first.timeout(const Duration(seconds: 15));
    },
    skip: bundle == null ? 'set DIANOIA_VL_BUNDLE to a pack-vl bundle' : false,
    timeout: const Timeout(Duration(minutes: 2)),
  );
}
