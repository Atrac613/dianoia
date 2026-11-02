import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/dianoia.dart';
import 'package:dianoia/lab.dart' show decodeImageRgb, readImageRgb;
import 'package:test/test.dart';

Uint8List _fixture(String name) =>
    File('test/fixtures/image_input/$name').readAsBytesSync();

// Reproduces a file growing after its initial length check, without timing races.
final class _GrowingFile implements File {
  _GrowingFile(this.original);
  final File original;
  @override
  Future<int> length() async {
    final before = await original.length();
    final bytes = original.readAsBytesSync();
    original.writeAsBytesSync([...bytes, 1, 2, 3]);
    return before;
  }

  @override
  Stream<List<int>> openRead([int? start, int? end]) =>
      original.openRead(start, end);
  @override
  Future<RandomAccessFile> open({FileMode mode = FileMode.read}) =>
      original.open(mode: mode);
  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  final png = <String, List<int>>{
    'rgb.png': [
      255,
      0,
      0,
      0,
      255,
      0,
      0,
      0,
      255,
      10,
      20,
      30,
      128,
      64,
      32,
      255,
      255,
      255,
    ],
    'rgba.png': [
      255,
      127,
      127,
      255,
      255,
      255,
      0,
      0,
      255,
      191,
      191,
      191,
      255,
      255,
      255,
      255,
      255,
      255,
    ],
    'gray.png': [0, 17, 64, 128, 200, 255],
    'gray16.png': [0, 1, 128, 255, 170, 2],
    'gray-alpha.png': [255, 143, 80, 231, 200, 255],
    'indexed.png': [
      255,
      255,
      255,
      127,
      255,
      127,
      0,
      0,
      255,
      0,
      0,
      255,
      127,
      255,
      127,
      255,
      255,
      255,
    ],
  };
  for (final entry in png.entries) {
    test('PNG RGB8 conversion and white alpha composite: ${entry.key}', () {
      final decoded = decodeImageRgb(_fixture(entry.key));
      expect(decoded.width, 3);
      expect(decoded.height, 2);
      final expected = entry.key.startsWith('gray')
          ? [
              for (final n in entry.value) ...[n, n, n],
            ]
          : entry.value;
      expect(decoded.pixels, orderedEquals(expected));
      final public = ChatImage.encoded(_fixture(entry.key));
      expect(public.width, decoded.width);
      expect(public.height, decoded.height);
    });
  }

  final orientations = [
    [
      [20, 50, 80],
      [110, 140, 170],
    ],
    [
      [80, 50, 20],
      [170, 140, 110],
    ],
    [
      [170, 140, 110],
      [80, 50, 20],
    ],
    [
      [110, 140, 170],
      [20, 50, 80],
    ],
    [
      [20, 110],
      [50, 140],
      [80, 170],
    ],
    [
      [110, 20],
      [140, 50],
      [170, 80],
    ],
    [
      [170, 80],
      [140, 50],
      [110, 20],
    ],
    [
      [80, 170],
      [50, 140],
      [20, 110],
    ],
  ];
  for (var i = 0; i < orientations.length; i++) {
    test('JPEG EXIF orientation ${i + 1} is applied exactly once', () {
      final decoded = decodeImageRgb(_fixture('orientation-${i + 1}.jpg'));
      final expected = orientations[i];
      expect(decoded.width, expected.first.length * 8);
      expect(decoded.height, expected.length * 8);
      for (var y = 0; y < decoded.height; y++) {
        for (var x = 0; x < decoded.width; x++) {
          for (var c = 0; c < 3; c++) {
            expect(
              decoded.pixels[(y * decoded.width + x) * 3 + c],
              closeTo(expected[y ~/ 8][x ~/ 8], 1),
            );
          }
        }
      }
    });
  }

  for (final name in [
    'animated.png',
    'one-frame-apng.png',
    'unsupported.gif',
  ]) {
    test('rejects $name instead of silently selecting a frame', () {
      expect(() => ChatImage.encoded(_fixture(name)), throwsFormatException);
    });
  }

  test('encoded views honor their byte offset and length', () {
    for (final name in ['rgb.png', 'orientation-6.jpg']) {
      final source = _fixture(name);
      final buffer = Uint8List(source.length + 24)
        ..setRange(11, 11 + source.length, source);
      final view = Uint8List.sublistView(buffer, 11, 11 + source.length);
      expect(decodeImageRgb(view).pixels, decodeImageRgb(source).pixels);
    }
  });

  test('a growing local file cannot bypass the byte budget', () async {
    final dir = Directory.systemTemp.createTempSync('dianoia_image_growth');
    addTearDown(() => dir.deleteSync(recursive: true));
    final file = File('${dir.path}/image.png')
      ..writeAsBytesSync(_fixture('rgb.png'));
    final limit = file.lengthSync();
    await IOOverrides.runZoned(() async {
      await expectLater(
        ChatImage.fromFile(file.path, maxEncodedBytes: limit),
        throwsFormatException,
      );
    }, createFile: (_) => _GrowingFile(file));
  });

  test('rejects invalid data and bad PNG checksums consistently', () {
    final crc = _fixture('rgb.png')..[29] ^= 1;
    for (final bytes in [
      Uint8List(0),
      Uint8List.fromList([1, 2, 3]),
      crc,
      Uint8List.sublistView(_fixture('orientation-1.jpg'), 0, 20),
      Uint8List.sublistView(_fixture('rgb.png'), 0, 24),
    ]) {
      expect(() => ChatImage.encoded(bytes), throwsFormatException);
    }
  });

  test('enforces encoded-byte and decoded-pixel budgets', () {
    final data = _fixture('rgb.png');
    expect(
      () => ChatImage.encoded(data, maxEncodedBytes: data.length - 1),
      throwsFormatException,
    );
    expect(() => ChatImage.encoded(data, maxPixels: 5), throwsFormatException);
    expect(
      ChatImage.encoded(data, maxPixels: 6, maxEncodedBytes: data.length).width,
      3,
    );
    expect(() => ChatImage.encoded(data, maxPixels: 0), throwsArgumentError);
    expect(
      () => ChatImage.encoded(data, maxEncodedBytes: -1),
      throwsArgumentError,
    );
    for (final name in [
      'large-header.png',
      'large-header.jpg',
      'overflow-header.png',
    ]) {
      expect(
        () => ChatImage.encoded(_fixture(name)),
        throwsA(
          isA<FormatException>().having(
            (e) => e.message.toLowerCase(),
            'message',
            contains('pixels'),
          ),
        ),
      );
    }
  });

  test(
    'bounded local file reading uses content rather than its extension',
    () async {
      final root = Directory.systemTemp.createTempSync('dianoia_image_input');
      addTearDown(() => root.deleteSync(recursive: true));
      final file = File('${root.path}/not-a-jpeg.jpg')
        ..writeAsBytesSync(_fixture('rgb.png'));
      final decoded = await readImageRgb(file.path);
      expect(decoded.pixels, png['rgb.png']);
      final public = await ChatImage.fromFile(file.path);
      expect(public.width, 3);
      expect(public.height, 2);
      await expectLater(
        ChatImage.fromFile(file.path, maxEncodedBytes: 10),
        throwsFormatException,
      );
      await expectLater(
        ChatImage.fromFile(file.path, maxPixels: 5),
        throwsFormatException,
      );
      await expectLater(
        ChatImage.fromFile('${root.path}/missing', maxPixels: 0),
        throwsArgumentError,
      );
      await expectLater(
        ChatImage.fromFile('${root.path}/missing'),
        throwsA(isA<FileSystemException>()),
      );
      file.writeAsBytesSync([]);
      await expectLater(ChatImage.fromFile(file.path), throwsFormatException);
    },
  );
}
