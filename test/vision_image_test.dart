import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

void main() {
  final oracle =
      jsonDecode(
            File('test/fixtures/vision/image_oracle.json').readAsStringSync(),
          )
          as Map<String, Object?>;
  for (final item in oracle['cases'] as List) {
    final c = item as Map<String, Object?>;
    // A token cap on a larger budget must reproduce upstream's own budget.
    for (final capped in [false, true]) {
      final label = capped ? ' via a token cap' : '';
      test('upstream preprocessing bytes: ${c['name']}$label', () {
        final h = c['height'] as int;
        final w = c['width'] as int;
        final patch = c['patch_size'] as int;
        final maxPixels = c['max_pixels'] as int;
        final processor = capped
            ? Lfm3ImageProcessor(
                patchSize: patch,
                minPixels: c['min_pixels'] as int,
                maxPixels: maxPixels * 4,
              ).withMaxImageTokens(maxPixels ~/ (4 * patch * patch))
            : Lfm3ImageProcessor(
                patchSize: patch,
                minPixels: c['min_pixels'] as int,
                maxPixels: maxPixels,
              );
        expect(processor.maxPixels, maxPixels);
        _expectOracle(processor, c, h, w);
      });
    }
  }

  test('a token budget downscales large photos with the upstream rule', () {
    final official = Lfm3ImageProcessor.load('test/fixtures/vision');
    expect(official.maxImageTokens, 16384);
    // The checkpoint budget keeps a 12 MP photo at 11,844 tokens.
    expect(official.resizedSize(3024, 4032), (3008, 4032));
    final capped = official.withMaxImageTokens(1024);
    expect(capped.maxImageTokens, 1024);
    expect(capped.resizedSize(3024, 4032), (864, 1152));
    expect(capped.resizedSize(256, 256), (256, 256));
    expect(official.withMaxImageTokens(16384), same(official));
    expect(official.withMaxImageTokens(1 << 40), same(official));
    expect(official.withMaxImageTokens(64).maxImageTokens, 64);
    for (final tokens in [0, -1, 63]) {
      expect(() => official.withMaxImageTokens(tokens), throwsArgumentError);
    }
  });
  test('official processor configuration is accepted', () {
    final p = Lfm3ImageProcessor.load('test/fixtures/vision');
    expect(p.patchSize, 16);
    expect(p.resizedSize(256, 256), (256, 256));
    expect(p.resizedSize(16, 16), (256, 256));
  });

  test('invalid RGB, dimensions, aspect ratio and budgets are rejected', () {
    final p = Lfm3ImageProcessor();
    expect(
      () => p.prepareRgb(Uint8List(3), height: 2, width: 2),
      throwsArgumentError,
    );
    for (final dims in [(0, 16), (16, -1), (1, 201)]) {
      expect(() => p.resizedSize(dims.$1, dims.$2), throwsArgumentError);
    }
    expect(() => Lfm3ImageProcessor(patchSize: 0), throwsArgumentError);
    expect(() => Lfm3ImageProcessor(minPixels: 1), throwsArgumentError);
    expect(() => Lfm3ImageProcessor(maxPixels: 1024), throwsArgumentError);
  });

  test('unsupported preprocessing settings are rejected', () {
    for (final change in <Map<String, Object?>>[
      {'resample': 2},
      {'do_resize': false},
      {
        'image_mean': [0.0, 0.0, 0.0],
      },
      {'rescale_factor': 1.0},
      {'temporal_patch_size': '2'},
    ]) {
      final json =
          jsonDecode(
                File(
                  'test/fixtures/vision/processor_config.json',
                ).readAsStringSync(),
              )
              as Map<String, Object?>;
      (json['image_processor'] as Map<String, Object?>).addAll(change);
      expect(() => Lfm3ImageProcessor.fromJson(json), throwsFormatException);
    }
  });
}

void _expectOracle(
  Lfm3ImageProcessor processor,
  Map<String, Object?> c,
  int h,
  int w,
) {
  final rgb = Uint8List.fromList([
    for (var i = 0; i < h * w * 3; i++) (i * 37 + (i ~/ 7) * 13) % 256,
  ]);
  final result = processor.prepareRgb(rgb, height: h, width: w);
  expect(result.height, c['resized_height']);
  expect(result.width, c['resized_width']);
  final (t, gh, gw) = result.gridThw;
  expect([t, gh, gw], c['grid']);
  expect(result.imageTokens, gh * gw ~/ 4);
  expect(result.pixelValues.length, result.patchCount * result.patchDim);
  final bytes = ByteData(result.pixelValues.lengthInBytes);
  for (var i = 0; i < result.pixelValues.length; i++) {
    bytes.setFloat32(i * 4, result.pixelValues[i], Endian.little);
  }
  expect(sha256.convert(bytes.buffer.asUint8List()).toString(), c['sha256']);
}
