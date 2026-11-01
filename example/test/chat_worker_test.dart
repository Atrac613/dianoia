import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:dianoia/dianoia.dart';
import 'package:dianoia_example/src/chat_protocol.dart';
import 'package:dianoia_example/src/chat_worker.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test(
    'an idle worker reports the selected model and exits on close',
    () async {
      final directory = Directory.systemTemp.createTempSync('dianoia-worker-');
      final events = ReceivePort();
      final exited = ReceivePort();
      final iterator = StreamIterator<dynamic>(events);
      final worker = await Isolate.spawn(runChatWorker, (
        events.sendPort,
        ModelLocation(
          storeDir: '${directory.path}/models',
          dropDir: '${directory.path}/drop',
          copyDir: '${directory.path}/drop',
        ),
        ModelSpec.lfm3Q4,
      ), onExit: exited.sendPort);
      addTearDown(() async {
        worker.kill(priority: Isolate.immediate);
        await iterator.cancel();
        events.close();
        exited.close();
        directory.deleteSync(recursive: true);
      });

      expect(await iterator.moveNext(), isTrue);
      final started = iterator.current as WorkerStarted;
      expect(await iterator.moveNext(), isTrue);
      expect((iterator.current as ModelNeeded).spec.id, ModelSpec.lfm3Q4.id);
      started.commands.send(const CloseWorker());
      await exited.first.timeout(const Duration(seconds: 5));
      // Selecting a model must not download anything before the user asks.
      expect(directory.listSync(), isEmpty);
    },
  );
}
