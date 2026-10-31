import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

void main() {
  test(
    'renamed tensor copy preserves bytes across streaming buffer boundary',
    () {
      final dir = Directory.systemTemp.createTempSync('dianoia_subset');
      addTearDown(() => dir.deleteSync(recursive: true));
      final data = Uint8List(1024 * 1024 + 12);
      for (var i = 0; i < data.length; i++) {
        data[i] = i % 251;
      }
      final header = utf8.encode(
        jsonEncode({
          'source': {
            'dtype': 'F32',
            'shape': [data.length ~/ 4],
            'data_offsets': [0, data.length],
          },
        }),
      );
      final length = ByteData(8)..setUint64(0, header.length, Endian.little);
      final source = File('${dir.path}/source');
      source.writeAsBytesSync([
        ...length.buffer.asUint8List(),
        ...header,
        ...data,
      ]);
      final st = SafetensorsFile.open(source.path);
      try {
        st.writeSubset(
          '${dir.path}/out',
          (_, _) => true,
          rename: (_) => 'renamed',
        );
      } finally {
        st.close();
      }
      final out = SafetensorsFile.open('${dir.path}/out');
      try {
        expect(out.tensors.keys, ['renamed']);
        final bytes = File('${dir.path}/out').readAsBytesSync();
        expect(
          Uint8List.sublistView(bytes, out.dataStart),
          orderedEquals(data),
        );
      } finally {
        out.close();
      }
    },
  );

  test('renaming collisions are rejected before an output file is opened', () {
    final dir = Directory.systemTemp.createTempSync('dianoia_subset');
    addTearDown(() => dir.deleteSync(recursive: true));
    final header = utf8.encode(
      jsonEncode({
        for (final name in ['a', 'b'])
          name: {
            'dtype': 'F32',
            'shape': [1],
            'data_offsets': [0, 4],
          },
      }),
    );
    final length = ByteData(8)..setUint64(0, header.length, Endian.little);
    final source = File('${dir.path}/source');
    source.writeAsBytesSync([
      ...length.buffer.asUint8List(),
      ...header,
      0,
      0,
      0,
      0,
    ]);
    final st = SafetensorsFile.open(source.path);
    try {
      expect(
        () => st.writeSubset(
          '${dir.path}/out',
          (_, _) => true,
          rename: (_) => 'same',
        ),
        throwsArgumentError,
      );
      expect(File('${dir.path}/out').existsSync(), isFalse);
    } finally {
      st.close();
    }
  });
}
