// What a damaged weight cache, a forward pass that fails partway and a reply
// that fails mid-generation leave behind. Runs on the synthetic backbone of
// the vision fixture, on the native CPU kernels.
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/vision_model_fixture.dart';

void main() {
  late Directory root;
  late String backbone;
  late String cachePath;

  setUpAll(() async {
    root = Directory.systemTemp.createTempSync('dianoia-failure-');
    await prepareVisionFixture(root);
    backbone = '${root.path}/backbone';
    cachePath = '$backbone/lfm3-q8.cache';
  });

  tearDownAll(() => root.deleteSync(recursive: true));

  group('Q8Cache.open', () {
    late Uint8List original;
    setUp(() => original = File(cachePath).readAsBytesSync());
    tearDown(() => File(cachePath).writeAsBytesSync(original));

    String key() => Q8Cache.keyFor(backbone);

    test('a truncated cache throws and releases its mapping', () {
      File(
        cachePath,
      ).writeAsBytesSync(original.sublist(0, original.length - 1));
      expect(() => Q8Cache.open(cachePath, key()), throwsFormatException);
      expect(MappedFile.isMapped(cachePath), isFalse);
    });

    test('a header length past the end of the file throws', () {
      final damaged = Uint8List.fromList(original);
      ByteData.sublistView(
        damaged,
      ).setUint64(8, original.length, Endian.little);
      File(cachePath).writeAsBytesSync(damaged);
      expect(() => Q8Cache.open(cachePath, key()), throwsFormatException);
      expect(MappedFile.isMapped(cachePath), isFalse);
    });

    test('a matrix offset outside the file throws', () {
      final cache = Q8Cache.open(cachePath, key())!;
      final (name, rows, cols) = cache.shapes.first;
      final qs = cache.offsets(name, rows, cols)!.qs;
      cache.close();
      // Offsets are written as decimal JSON numbers; a digit more moves
      // the matrix far past the end of the file.
      final text = String.fromCharCodes(original.sublist(16, 16 + 4096));
      final at = text.indexOf('"qs":$qs');
      expect(at, isNonNegative);
      final damaged = Uint8List.fromList(original);
      damaged[16 + at + '"qs":'.length] = '9'.codeUnitAt(0);
      File(cachePath).writeAsBytesSync(damaged);
      expect(() => Q8Cache.open(cachePath, key()), throwsFormatException);
      expect(MappedFile.isMapped(cachePath), isFalse);
    });

    test('an intact cache still opens', () {
      final cache = Q8Cache.open(cachePath, key());
      expect(cache, isNotNull);
      cache!.close();
    });
  }, skip: NativeKernels.isAvailable ? false : 'requires native kernels');

  group('after a failure', () {
    late Lfm3Model model;
    setUpAll(() async {
      model = await Lfm3Model.load(
        backbone,
        maxSeq: 64,
        maxChunk: 4,
        weights: WeightFormat.q8,
        q8CachePath: cachePath,
      );
    });
    tearDownAll(() => model.close());
    setUp(() => model.reset());

    test('a forward pass that fails partway forgets the sequence', () async {
      await model.forward([10, 20, 30]);
      await model.checkpoint();
      // Fails in the second chunk, after the first has advanced the state.
      await expectLater(
        model.forward(
          [40, 50, 60, 70, 80, 90],
          capture: (slot, first, rows, count) {
            if (first > 3) throw StateError('injected failure');
          },
        ),
        throwsStateError,
      );
      expect(model.length, 0);
      expect(model.history, isEmpty);

      // The same tokens fed again match a fresh model.
      await model.forward([10, 20, 30, 40]);
      final again = Float32List.fromList(model.logits);
      await model.reset();
      await model.forward([10, 20, 30, 40]);
      expect(again, orderedEquals(model.logits));
    });

    test('a reply that fails mid-generation leaves the messages', () async {
      final session = ChatSession(
        model,
        Lfm3Tokenizer.load(backbone),
        system: 'S',
        maxNew: 8,
      );
      await session.warmUp();
      final before = [...session.messages];
      var calls = 0;
      await expectLater(
        session.send(
          {'role': 'user', 'content': 'hi'},
          onText: (_) {
            if (++calls == 2) throw StateError('injected failure');
            return true;
          },
        ),
        throwsStateError,
      );
      expect(session.messages, before);

      // The next turn renders one user message, as if the failed one had
      // never been sent.
      final turn = await session.send({'role': 'user', 'content': 'hi'});
      expect(session.messages, hasLength(2));
      expect(
        turn.promptTokens,
        session.promptIds([
          {'role': 'user', 'content': 'hi'},
        ]).length,
      );
    });
  }, skip: NativeKernels.isAvailable ? false : 'requires native kernels');
}
