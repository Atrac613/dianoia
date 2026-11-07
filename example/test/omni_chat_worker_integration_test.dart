import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:dianoia_example/src/chat_protocol.dart';
import 'package:dianoia_example/src/chat_worker.dart';
import 'package:dianoia_example/src/example_model.dart';
import 'package:flutter_test/flutter_test.dart';

/// A prepared research folder or an installed Omni bundle, and a PNG or
/// JPEG of a red circle left of a yellow square.
void main() {
  final directory = Platform.environment['DIANOIA_OMNI_EXAMPLE_DIR'];
  final image = Platform.environment['DIANOIA_OMNI_EXAMPLE_IMAGE'];
  test(
    'Omni chat worker answers text, images and follow-ups, and stops',
    () async {
      final events = ReceivePort();
      final exited = ReceivePort();
      final iterator = StreamIterator<dynamic>(events);
      final worker = await Isolate.spawn(runChatWorker, (
        events.sendPort,
        ModelLocation(
          storeDir: '$directory/unused-store',
          dropDir: directory!,
          copyDir: directory,
        ),
        ExampleModel.omni,
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

      Future<(ReplyDone, List<String>)> answer(SendMessage message) async {
        commands.send(message);
        final progress = <String>[];
        while (await iterator.moveNext()) {
          final event = iterator.current;
          expect(event, isNot(isA<ReplyFailed>()), reason: '$event');
          if (event is ReplyProgress) progress.add(event.status);
          if (event is ReplyDone) return (event, progress);
        }
        fail('the worker closed before replying');
      }

      final (ok, _) = await answer(const SendMessage('Reply with only OK.'));
      expect(ok.tokens, greaterThan(0));
      expect(ok.text.trim(), isNotEmpty);

      final photo = ImageAttachment(
        'shapes.png',
        File(image!).readAsBytesSync(),
      );
      final (shapes, progress) = await answer(
        SendMessage(
          'What shapes are in this image? Answer briefly.',
          image: photo,
        ),
      );
      expect(shapes.text.toLowerCase(), contains('circle'));
      expect(progress.first, 'Reading image…');
      expect(progress.any((s) => s.startsWith('Analyzing image…')), isTrue);
      expect(progress, containsAll(['Preparing reply…', 'Generating reply…']));

      // The follow-up sends no image: the worker replays the encoded one.
      final (color, _) = await answer(
        const SendMessage('What color is the square? Answer with one word.'),
      );
      expect(color.text.toLowerCase(), contains('yellow'));

      commands.send(const NewChat());
      commands.send(SendMessage('Describe this image.', image: photo));
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

      // A stopped reply leaves the cleared conversation empty.
      final (fresh, _) = await answer(
        const SendMessage('Is there an image in this chat? Answer yes or no.'),
      );
      expect(fresh.text.toLowerCase(), contains('no'));

      commands.send(const CloseWorker());
      await exited.first.timeout(const Duration(seconds: 15));
    },
    skip: directory == null || image == null
        ? 'set DIANOIA_OMNI_EXAMPLE_DIR and DIANOIA_OMNI_EXAMPLE_IMAGE'
        : false,
    timeout: const Timeout(Duration(minutes: 5)),
  );
}
