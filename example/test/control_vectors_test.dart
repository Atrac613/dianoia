// Experimental control vectors in the text chat: their settings files, the
// folder listing, loading, the worker messages and the chat page's menu.
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:dianoia/lab.dart' show ControlMode, ControlVector;
import 'package:dianoia_example/src/chat_controller.dart';
import 'package:dianoia_example/src/chat_page.dart';
import 'package:dianoia_example/src/chat_protocol.dart';
import 'package:dianoia_example/src/control_vectors.dart';
import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('SteeringSettings.parse', () {
    test('reads every key', () {
      final s = SteeringSettings.parse(
        '{"mode": "add", "scale": 0.15, "first_layer": 8, "last_layer": 16}',
      );
      expect(s.mode, ControlMode.add);
      expect(s.scale, 0.15);
      expect((s.firstLayer, s.lastLayer), (8, 16));
    });

    test('defaults missing keys to project, scale 1, every layer', () {
      final s = SteeringSettings.parse('{"last_layer": null}');
      expect(s.mode, ControlMode.project);
      expect(s.scale, 1);
      expect((s.firstLayer, s.lastLayer), (0, null));
      expect(SteeringSettings.parse('{"scale": 1}').scale, 1.0);
    });

    test('rejects what it cannot use', () {
      for (final bad in [
        '{',
        '[]',
        '{"mode": "remove"}',
        '{"scale": "big"}',
        '{"first_layer": -1}',
        '{"first_layer": 1.5}',
        '{"first_layer": 9, "last_layer": 8}',
        '{"layers": "8-"}',
      ]) {
        expect(
          () => SteeringSettings.parse(bad),
          throwsFormatException,
          reason: bad,
        );
      }
    });
  });

  group('folder', () {
    late Directory dir;
    setUp(() => dir = Directory.systemTemp.createTempSync('dianoia-cv-'));
    tearDown(() => dir.deleteSync(recursive: true));

    void vector(String name, {int hiddenSize = 4}) => ControlVector({
      1: Float32List.fromList([for (var j = 0; j < hiddenSize; j++) j + 1.0]),
    }, hiddenSize: hiddenSize).save('${dir.path}/$name.safetensors');

    test('lists the vectors by name, sorted', () {
      vector('short');
      vector('casual');
      File('${dir.path}/casual.json').writeAsStringSync('{}');
      File('${dir.path}/notes.txt').writeAsStringSync('');
      expect(listControlVectors(dir.path), ['casual', 'short']);
      expect(listControlVectors('${dir.path}/missing'), isEmpty);
      expect(listControlVectors(null), isEmpty);
    });

    test('loads a vector with its settings, or the defaults', () {
      vector('lang');
      File(
        '${dir.path}/lang.json',
      ).writeAsStringSync('{"mode": "add", "scale": 0.2, "first_layer": 1}');
      final v = loadControlVector(dir.path, 'lang');
      expect(v.mode, ControlMode.add);
      expect(v.scale, closeTo(0.2, 1e-12));
      expect(v.steers(1), isTrue);
      vector('refusal');
      final d = loadControlVector(dir.path, 'refusal');
      expect((d.mode, d.scale, d.firstLayer), (ControlMode.project, 1.0, 0));
    });

    test('a broken file or settings throws instead of crashing', () {
      vector('bad');
      File('${dir.path}/bad.json').writeAsStringSync('{"mode": 3}');
      expect(() => loadControlVector(dir.path, 'bad'), throwsFormatException);
      File('${dir.path}/junk.safetensors').writeAsStringSync('not a file');
      expect(() => loadControlVector(dir.path, 'junk'), throwsA(anything));
    });
  });

  test('the steering messages cross isolates', () async {
    final state = await Isolate.run(
      () => const SteeringState(['a', 'b'], 'a', error: 'e'),
    );
    expect(state.available, ['a', 'b']);
    expect((state.active, state.error), ('a', 'e'));
    final command = await Isolate.run(() => const SetSteering(null));
    expect(command.name, isNull);
  });

  testWidgets('the menu offers Off and each vector, the active one checked', (
    tester,
  ) async {
    final chat = ChatController()
      ..setup = Setup.ready
      ..backend = 'GPU'
      ..steeringOptions = ['casual', 'refusal']
      ..steering = 'refusal';
    addTearDown(chat.dispose);
    await tester.pumpWidget(MaterialApp(home: ChatPage(controller: chat)));
    expect(find.textContaining('steered: refusal'), findsOneWidget);
    await tester.tap(find.byTooltip('Control vector (experimental)'));
    await tester.pumpAndSettle();
    expect(find.text('Off'), findsOneWidget);
    expect(find.text('casual'), findsOneWidget);
    final checked = tester.widget<CheckedPopupMenuItem<Object>>(
      find.ancestor(
        of: find.text('refusal'),
        matching: find.byType(CheckedPopupMenuItem<Object>),
      ),
    );
    expect(checked.checked, isTrue);
  });

  testWidgets('without vectors there is no menu', (tester) async {
    final chat = ChatController()..setup = Setup.ready;
    addTearDown(chat.dispose);
    await tester.pumpWidget(MaterialApp(home: ChatPage(controller: chat)));
    expect(find.byTooltip('Control vector (experimental)'), findsNothing);
  });
}
