import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/vision_model_fixture.dart';

void main() {
  late Directory root;
  late String bundle;
  setUpAll(() async {
    root = Directory.systemTemp.createTempSync('dianoia_vision_bench');
    await prepareVisionFixture(root);
    bundle = '${root.path}/packed';
    await packVlBundle('${root.path}/model', '${root.path}/backbone', bundle);
  });
  tearDownAll(() => root.deleteSync(recursive: true));

  Future<Map<String, Object?>> run({
    String? expected,
    String? imagePath,
    void Function(Map<String, Object?>)? onRecord,
  }) => benchmarkVisionBundle(
    bundle: bundle,
    imagePath: imagePath ?? 'test/fixtures/image_input/gradient.png',
    prompt: 'Describe.',
    backend: InferenceBackend.cpu,
    runs: 2,
    maxNewTokens: 2,
    expectedText: expected,
    onRecord: onRecord,
  );

  test(
    'records timing, process memory, fresh turns and retained-image follow-up',
    () async {
      final phases = <String>[];
      final record = await run(
        onRecord: (r) {
          phases.add(r['phase'] as String);
          expect(jsonDecode(jsonEncode(r)), isA<Map>());
          if (r['phase'] == 'loading') expect(r['finished'], isFalse);
        },
      );
      expect(record['passed'], isTrue);
      expect(record['finished'], isTrue);
      expect(record['phase'], 'done');
      expect(record['backend'], 'cpu');
      expect(record['requested_max_image_tokens'], 1024);
      // The fixture checkpoint's own budget is lower than the request.
      expect(record['max_image_tokens'], 256);
      expect(record['image_width'], 8);
      expect(phases.first, 'loading');
      expect(
        phases,
        containsAllInOrder([
          'loaded',
          'image_decoded',
          'fresh_1',
          'fresh_1_done',
          'fresh_2',
          'follow_up',
          'closing',
          'done',
        ]),
      );
      final turns = (record['turns'] as List).cast<Map>();
      expect(turns, hasLength(3));
      expect(turns.map((t) => t['new_image']), [true, true, false]);
      expect(turns.first['text'], turns[1]['text']);
      expect(
        turns.last['prompt_tokens'] as int,
        greaterThan(turns[1]['prompt_tokens'] as int),
      );
      for (final turn in turns) {
        expect(turn['generated_tokens'], 2);
        expect(turn['first_text_s'] as double, greaterThanOrEqualTo(0));
        expect(
          turn['wall_s'] as double,
          greaterThanOrEqualTo(turn['first_text_s'] as double),
        );
      }
      final memory = (record['memory'] as List).cast<Map>();
      expect(memory.first['phase'], 'loading');
      expect(memory.last['phase'], 'done');
      expect(memory.every((s) => (s['rss_bytes'] as int) > 0), isTrue);
      expect(MappedFile.isMapped('$bundle/language/lfm3-q8.cache'), isFalse);
    },
  );

  test('quality mismatch remains a completed failed result', () async {
    final record = await run(
      expected: 'This cannot match the synthetic model.',
    );
    expect(record['finished'], isTrue);
    expect(record['passed'], isFalse);
    expect(
      (record['turns'] as List).cast<Map>().every(
        (t) => t['expected_text_matches'] == false,
      ),
      isTrue,
    );
  });

  test(
    'a differently worded follow-up does not fail the fresh turns',
    () async {
      final fresh = ((await run())['turns'] as List).cast<Map>();
      expect(fresh.last['text'], isNot(fresh.first['text']));
      final record = await run(expected: fresh.first['text'] as String);
      final turns = (record['turns'] as List).cast<Map>();
      expect(turns.map((t) => t['expected_text_matches']), [true, true, false]);
      expect(record['passed'], isTrue);
    },
  );

  test('decode failure records an error and closes the loaded model', () async {
    Map<String, Object?>? finalRecord;
    await expectLater(
      run(
        imagePath: '${root.path}/missing.png',
        onRecord: (r) => finalRecord = r,
      ),
      throwsA(isA<FileSystemException>()),
    );
    expect(finalRecord!['finished'], isTrue);
    expect(finalRecord!['phase'], 'failed');
    expect(finalRecord!['passed'], isFalse);
    expect(finalRecord!['error'], isNotNull);
    expect(MappedFile.isMapped('$bundle/language/lfm3-q8.cache'), isFalse);
  });

  test('checkpoint write failure still releases model resources', () async {
    await expectLater(
      run(
        onRecord: (r) {
          if (r['phase'] == 'closing') {
            throw const FileSystemException('checkpoint failed');
          }
        },
      ),
      throwsA(isA<FileSystemException>()),
    );
    expect(MappedFile.isMapped('$bundle/language/lfm3-q8.cache'), isFalse);
  });
}
