// The chat worker with real control vectors. Needs a text bundle with
// lfm3-q4.cache in DIANOIA_LFM3_DIR and a control vector for it
// (.safetensors) in DIANOIA_CONTROL_VECTOR; skipped otherwise. Checks only
// the steering state and that replies complete, never their text.
import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:dianoia_example/src/chat_protocol.dart';
import 'package:dianoia_example/src/chat_worker.dart';
import 'package:dianoia_example/src/example_model.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final bundle = Platform.environment['DIANOIA_LFM3_DIR'];
  final vector = Platform.environment['DIANOIA_CONTROL_VECTOR'];
  test(
    'the worker lists, sets, rejects and clears control vectors',
    () async {
      final vectors = Directory.systemTemp.createTempSync('dianoia-vectors-');
      File(vector!).copySync('${vectors.path}/refusal.safetensors');
      File(
        '${vectors.path}/refusal.json',
      ).writeAsStringSync('{"mode": "project", "scale": 1, "first_layer": 8}');
      File(vector).copySync('${vectors.path}/broken.safetensors');
      File('${vectors.path}/broken.json').writeAsStringSync('{"mode": 1}');
      final events = ReceivePort();
      final exited = ReceivePort();
      final iterator = StreamIterator<dynamic>(events);
      final worker = await Isolate.spawn(runChatWorker, (
        events.sendPort,
        ModelLocation(
          storeDir: '${vectors.path}/models',
          dropDir: bundle!,
          copyDir: bundle,
          controlVectorDir: vectors.path,
        ),
        ExampleModel.lfm3,
      ), onExit: exited.sendPort);
      addTearDown(() async {
        worker.kill(priority: Isolate.immediate);
        await iterator.cancel();
        events.close();
        exited.close();
        vectors.deleteSync(recursive: true);
      });

      Future<T> next<T>() async {
        while (await iterator.moveNext().timeout(const Duration(minutes: 3))) {
          if (iterator.current case final T event) return event;
          if (iterator.current case WorkerFailed(:final message)) {
            fail(message);
          }
        }
        fail('the worker stopped');
      }

      final commands = (await next<WorkerStarted>()).commands;
      await next<WorkerReady>();
      var state = await next<SteeringState>();
      expect(state.available, ['broken', 'refusal']);
      expect(state.active, isNull);

      commands.send(const SetSteering('broken'));
      state = await next<SteeringState>();
      expect(state.active, isNull);
      expect(state.error, contains('broken'));

      commands.send(const SetSteering('refusal'));
      state = await next<SteeringState>();
      expect((state.active, state.error), ('refusal', null));
      commands.send(const SendMessage('Say hello.'));
      expect((await next<ReplyDone>()).tokens, greaterThan(0));

      // Removing the files must not suppress the acknowledgement of Off.
      vectors.deleteSync(recursive: true);
      commands.send(const SetSteering(null));
      state = await next<SteeringState>();
      expect((state.active, state.error), (null, null));
      expect(state.available, isEmpty);
      commands.send(const SendMessage('Say hello.'));
      expect((await next<ReplyDone>()).tokens, greaterThan(0));

      vectors.createSync();
      File(vector).copySync('${vectors.path}/refusal.safetensors');
      commands.send(const RescanSteering());
      expect((await next<SteeringState>()).available, ['refusal']);
      // Closing during a switch must wait for the warm-up before disposal.
      commands.send(const SetSteering('refusal'));
      commands.send(const CloseWorker());
      await exited.first.timeout(const Duration(seconds: 30));
    },
    timeout: const Timeout(Duration(minutes: 10)),
    skip: bundle == null || vector == null
        ? 'set DIANOIA_LFM3_DIR and DIANOIA_CONTROL_VECTOR'
        : false,
  );
}
