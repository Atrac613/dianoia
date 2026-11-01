import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';

void writeSyntheticVisionWeights(
  String path,
  Map<String, Object?> shapes, {
  bool bf16 = false,
  bool suppressSpecialEmbeddings = false,
}) {
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
    final bytes = ByteData(count * (bf16 ? 2 : 4));
    final scalar = ByteData(4);
    for (var i = 0; i < count; i++) {
      var value =
          (norm ? 1 : 0) +
          ((seed + i * 7 + (i ~/ 3) * 11) % 37 - 18) / (norm ? 256 : 64);
      if (suppressSpecialEmbeddings &&
          entry.key.endsWith('embed_tokens.weight')) {
        final row = i ~/ shape.last;
        if (row < 32 || row > 126) value = 0;
      }
      if (bf16) {
        scalar.setFloat32(0, value, Endian.little);
        bytes.setUint16(
          i * 2,
          scalar.getUint32(0, Endian.little) >> 16,
          Endian.little,
        );
      } else {
        bytes.setFloat32(i * 4, value, Endian.little);
      }
    }
    header[entry.key] = {
      'dtype': bf16 ? 'BF16' : 'F32',
      'shape': shape,
      'data_offsets': [offset, offset + bytes.lengthInBytes],
    };
    data.add(bytes.buffer.asUint8List());
    offset += bytes.lengthInBytes;
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
