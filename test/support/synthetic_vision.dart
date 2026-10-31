import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';

void writeSyntheticVisionWeights(String path, Map<String, Object?> shapes) {
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

VisionImageInput syntheticVisionImage(int gh, int gw, {int patch = 2}) {
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
