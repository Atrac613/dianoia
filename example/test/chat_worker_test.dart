import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:dianoia_example/src/example_model.dart';
import 'package:dianoia_example/src/chat_protocol.dart';
import 'package:dianoia_example/src/chat_worker.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  for (final selected in ExampleModel.values) {
    test(
      'an idle worker reports ${selected.name} and exits on close',
      () async {
        final directory = Directory.systemTemp.createTempSync(
          'dianoia-worker-',
        );
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
          selected,
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
        expect(
          (iterator.current as ModelNeeded).spec.id,
          selected.downloadSpec.id,
        );
        started.commands.send(const CloseWorker());
        await exited.first.timeout(const Duration(seconds: 5));
        // Selecting a model must not download anything before the user asks.
        expect(directory.listSync(), isEmpty);
      },
    );
  }
  test(
    'VL sideload refreshes nested files and reports an invalid bundle',
    () async {
      final directory = Directory.systemTemp.createTempSync('dianoia-vl-copy-');
      final source = Directory('${directory.path}/source')..createSync();
      final destination = Directory('${directory.path}/copy')..createSync();
      File('${source.path}/vl_bundle.json').writeAsStringSync('{}');
      final sourceFile = File('${source.path}/language/model.safetensors');
      sourceFile.parent.createSync();
      sourceFile.writeAsStringSync('source');
      final copyFile = File('${destination.path}/language/model.safetensors');
      copyFile.parent.createSync();
      copyFile.writeAsStringSync('broken');
      // Left by an earlier bundle: mirroring must not keep it.
      final stale = File('${destination.path}/language/lfm3-q8.cache')
        ..writeAsStringSync('stale');
      final events = ReceivePort();
      final exited = ReceivePort();
      final iterator = StreamIterator<dynamic>(events);
      final worker = await Isolate.spawn(runChatWorker, (
        events.sendPort,
        ModelLocation(
          storeDir: '${directory.path}/models',
          dropDir: source.path,
          copyDir: destination.path,
        ),
        ExampleModel.lfm3Vl,
      ), onExit: exited.sendPort);
      addTearDown(() async {
        worker.kill(priority: Isolate.immediate);
        await iterator.cancel();
        events.close();
        exited.close();
        directory.deleteSync(recursive: true);
      });
      expect(await iterator.moveNext(), isTrue);
      final commands = (iterator.current as WorkerStarted).commands;
      while (await iterator.moveNext()) {
        if (iterator.current is WorkerFailed) break;
      }
      expect(iterator.current, isA<WorkerFailed>());
      expect(copyFile.readAsStringSync(), 'source');
      expect(stale.existsSync(), isFalse);
      // The failed copy loses its manifest, so the next launch copies again.
      expect(File('${destination.path}/vl_bundle.json').existsSync(), isFalse);
      commands.send(const CloseWorker());
      await exited.first.timeout(const Duration(seconds: 5));
    },
  );
}
