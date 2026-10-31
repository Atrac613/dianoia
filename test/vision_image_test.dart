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
    test('upstream preprocessing bytes: ${c['name']}', () {
      final h = c['height'] as int;
      final w = c['width'] as int;
      final processor = Lfm3ImageProcessor(
        patchSize: c['patch_size'] as int,
        minPixels: c['min_pixels'] as int,
        maxPixels: c['max_pixels'] as int,
      );
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
      expect(
        sha256.convert(bytes.buffer.asUint8List()).toString(),
        c['sha256'],
      );
    });
  }

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
