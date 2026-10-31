import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

void _writeSyntheticWeights(String path, Map<String, Object?> shapes) {
  final header = <String, Object?>{};
  final data = BytesBuilder();
  var offset = 0;
  for (final entry in shapes.entries) {
    final shape = (entry.value as List).cast<int>();
    final count = shape.fold(1, (int a, b) => a * b);
    var seed = 0;
    for (var i = 0; i < entry.key.length; i++) {
      seed += (i + 1) * entry.key.codeUnitAt(i);
    }
    seed %= 997;
    final norm = entry.key.contains('norm') && entry.key.endsWith('.weight');
    final bytes = ByteData(count * 4);
    for (var i = 0; i < count; i++) {
      bytes.setFloat32(
        i * 4,
        (norm ? 1 : 0) +
            ((seed + i * 7 + (i ~/ 3) * 11) % 37 - 18) / (norm ? 256 : 64),
        Endian.little,
      );
    }
    header[entry.key] = {
      'dtype': 'F32',
      'shape': shape,
      'data_offsets': [offset, offset + count * 4],
    };
    data.add(bytes.buffer.asUint8List());
    offset += count * 4;
  }
  final json = utf8.encode(jsonEncode(header));
  final padded = (json.length + 7) ~/ 8 * 8;
  final prefix = ByteData(8)..setUint64(0, padded, Endian.little);
  File(path).writeAsBytesSync([
    ...prefix.buffer.asUint8List(),
    ...json,
    ...List.filled(padded - json.length, 32),
    ...data.takeBytes(),
  ]);
}

VisionImageInput _image(int gh, int gw, {int patch = 2}) {
  final height = gh * patch, width = gw * patch;
  return Lfm3ImageProcessor(
    patchSize: patch,
    minPixels: 16,
    maxPixels: 4096,
  ).prepareRgb(
    Uint8List.fromList([
      for (var i = 0; i < height * width * 3; i++)
        (i * 37 + (i ~/ 7) * 13) % 256,
    ]),
    height: height,
    width: width,
  );
}

void _near(Float32List actual, List expected, String stage) {
  expect(actual.length, expected.length, reason: stage);
  var maxError = 0.0;
  for (var i = 0; i < actual.length; i++) {
    final value = (expected[i] as num).toDouble();
    final error = (actual[i] - value).abs();
    maxError = math.max(maxError, error);
    expect(
      actual[i].isFinite && error <= 2e-5 + 2e-5 * value.abs(),
      isTrue,
      reason: '$stage index $i: ${actual[i]} vs $value, max error $maxError',
    );
  }
}

void main() {
  final oracle =
      jsonDecode(
            File('test/fixtures/vision/encoder_oracle.json').readAsStringSync(),
          )
          as Map<String, Object?>;
  late Directory directory;
  late Lfm3VisionEncoder encoder;
  setUp(() {
    directory = Directory.systemTemp.createTempSync('dianoia_vision_encoder');
    File(
      '${directory.path}/config.json',
    ).writeAsStringSync(jsonEncode(oracle['config']));
    _writeSyntheticWeights(
      '${directory.path}/model.safetensors',
      oracle['shapes'] as Map<String, Object?>,
    );
    encoder = Lfm3VisionEncoder.load(directory.path);
  });
  tearDown(() => directory.deleteSync(recursive: true));

  for (final item in oracle['cases'] as List) {
    final c = item as Map<String, Object?>;
    final grid = (c['grid'] as List).cast<int>();
    test('upstream encoder and projector intermediate outputs: $grid', () {
      final stages = c['stages'] as Map<String, Object?>;
      final seen = <String>[];
      final image = _image(grid[0], grid[1]);
      final input = Float32List.fromList(image.pixelValues);
      final result = encoder.encode(
        image,
        capture: (stage, rows) {
          _near(rows, stages[stage] as List, stage);
          seen.add(stage);
          // Observers receive snapshots, not mutable model state.
          rows.fillRange(0, rows.length, double.nan);
        },
      );
      expect(seen, ['embedding', 'layer.0', 'layer.1', 'layer.2', 'projected']);
      expect(result.length, image.imageTokens * encoder.config.text.hiddenSize);
      _near(result, stages['projected'] as List, 'result');
      expect(image.pixelValues, orderedEquals(input));
      expect(encoder.encode(image), orderedEquals(result));
    });
  }

  test('mismatched preprocessing and nonfinite pixels are rejected', () {
    expect(() => encoder.encode(_image(2, 2, patch: 4)), throwsArgumentError);
    final image = _image(2, 2);
    image.pixelValues[0] = double.nan;
    expect(() => encoder.encode(image), throwsArgumentError);
  });

  test('missing and incorrect weight shapes fail while loading', () {
    final shapes = Map<String, Object?>.from(oracle['shapes'] as Map);
    shapes.remove('model.multi_modal_projector.linear_2.bias');
    _writeSyntheticWeights('${directory.path}/model.safetensors', shapes);
    expect(() => Lfm3VisionEncoder.load(directory.path), throwsArgumentError);
    shapes['model.multi_modal_projector.linear_2.bias'] = [1];
    _writeSyntheticWeights('${directory.path}/model.safetensors', shapes);
    expect(() => Lfm3VisionEncoder.load(directory.path), throwsArgumentError);
  });
}
