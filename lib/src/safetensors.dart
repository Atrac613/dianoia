/// Minimal safetensors reader: an 8-byte little-endian header length, a JSON
/// header mapping tensor names to dtype, shape and byte range, then raw data.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

final class TensorInfo {
  TensorInfo(this.dtype, this.shape, this.begin, this.end);

  final String dtype;
  final List<int> shape;
  final int begin;
  final int end;

  int get numel => shape.fold(1, (a, b) => a * b);
}

final class SafetensorsFile {
  SafetensorsFile._(this._file, this._dataStart, this.tensors);

  factory SafetensorsFile.open(String path) {
    final file = File(path).openSync();
    final lengthBytes = Uint8List(8);
    _readFully(file, lengthBytes, 0);
    final headerLength = ByteData.sublistView(
      lengthBytes,
    ).getUint64(0, Endian.little);
    final headerBytes = Uint8List(headerLength);
    _readFully(file, headerBytes, 8);
    final header = jsonDecode(utf8.decode(headerBytes)) as Map<String, Object?>;
    header.remove('__metadata__');
    final tensors = {
      for (final MapEntry(:key, :value) in header.entries)
        key: () {
          final info = value as Map<String, Object?>;
          final offsets = (info['data_offsets'] as List).cast<int>();
          return TensorInfo(
            info['dtype'] as String,
            (info['shape'] as List).cast<int>(),
            offsets[0],
            offsets[1],
          );
        }(),
    };
    return SafetensorsFile._(file, 8 + headerLength, tensors);
  }

  final RandomAccessFile _file;
  final int _dataStart;

  /// File offset of the data section; tensor offsets are relative to it.
  int get dataStart => _dataStart;
  final Map<String, TensorInfo> tensors;
  Uint8List _scratch = Uint8List(0);

  /// Reads a tensor as float32, checking its shape. BF16 widens exactly:
  /// a bfloat16 is the top half of the float32 with the same value.
  ///
  /// [rowStart] and [rowEnd] select rows of the first dimension. Rows of a
  /// row-major tensor are contiguous, so a slice is a single read.
  Float32List readF32(
    String name,
    List<int> shape, {
    int rowStart = 0,
    int? rowEnd,
  }) {
    final info = tensors[name];
    if (info == null) throw ArgumentError('tensor $name not found');
    if (info.shape.length != shape.length ||
        !Iterable.generate(
          shape.length,
        ).every((i) => info.shape[i] == shape[i])) {
      throw ArgumentError(
        'tensor $name has shape ${info.shape}, expected $shape',
      );
    }
    final elementBytes = switch (info.dtype) {
      'F32' => 4,
      'BF16' => 2,
      _ => throw UnsupportedError('tensor $name has dtype ${info.dtype}'),
    };
    final rows = shape.isEmpty ? 1 : shape.first;
    final end = rowEnd ?? rows;
    if (rowStart < 0 || end > rows || rowStart > end) {
      throw RangeError('rows $rowStart..$end of $name with $rows rows');
    }
    final rowElements = rows == 0 ? 0 : info.numel ~/ rows;
    final count = (end - rowStart) * rowElements;
    final bytes = count * elementBytes;
    if (_scratch.length < bytes) _scratch = Uint8List(bytes);
    _readFully(
      _file,
      _scratch,
      _dataStart + info.begin + rowStart * rowElements * elementBytes,
      bytes,
    );

    final out = Float32List(count);
    if (elementBytes == 4) {
      out.setRange(0, count, Float32List.view(_scratch.buffer, 0, count));
    } else {
      _widenBf16(Uint16List.view(_scratch.buffer, 0, count), out);
    }
    return out;
  }

  /// Reads a BF16 tensor as its raw 16-bit patterns, half the size of
  /// [readF32]; widening a pattern p to float32 is `p << 16`.
  Uint16List readBf16Bits(String name, List<int> shape) {
    final info = tensors[name];
    if (info == null) throw ArgumentError('tensor $name not found');
    if (info.dtype != 'BF16' || info.shape.join(',') != shape.join(',')) {
      throw ArgumentError('tensor $name is ${info.dtype} ${info.shape}');
    }
    final out = Uint16List(info.numel);
    _readFully(
      _file,
      Uint8List.view(out.buffer),
      _dataStart + info.begin,
      info.end - info.begin,
    );
    return out;
  }

  void close() => _file.closeSync();

  /// The length prefix and JSON [header], padded with spaces so the data
  /// section starts 8-byte aligned, as the format suggests.
  static void _writeHeader(RandomAccessFile out, Map<String, Object?> header) {
    var json = utf8.encode(jsonEncode(header));
    final padded = (json.length + 7) ~/ 8 * 8;
    json = Uint8List(padded)
      ..fillRange(0, padded, 0x20)
      ..setAll(0, json);
    final length = ByteData(8)..setUint64(0, padded, Endian.little);
    out
      ..writeFromSync(length.buffer.asUint8List())
      ..writeFromSync(json);
  }

  /// Writes the tensors for which [keep] holds to a new safetensors file
  /// at [path], bytes unchanged and in this file's order.
  void writeSubset(
    String path,
    bool Function(String name, TensorInfo info) keep,
  ) {
    final kept = tensors.entries.where((e) => keep(e.key, e.value)).toList()
      ..sort((a, b) => a.value.begin.compareTo(b.value.begin));
    final header = <String, Object?>{
      '__metadata__': {'format': 'pt'},
    };
    var offset = 0;
    for (final MapEntry(:key, :value) in kept) {
      final bytes = value.end - value.begin;
      header[key] = {
        'dtype': value.dtype,
        'shape': value.shape,
        'data_offsets': [offset, offset + bytes],
      };
      offset += bytes;
    }
    final out = File(path).openSync(mode: FileMode.write);
    try {
      _writeHeader(out, header);
      var buffer = Uint8List(0);
      for (final MapEntry(:value) in kept) {
        final bytes = value.end - value.begin;
        if (buffer.length < bytes) buffer = Uint8List(bytes);
        _readFully(_file, buffer, _dataStart + value.begin, bytes);
        out.writeFromSync(buffer, 0, bytes);
      }
    } finally {
      out.closeSync();
    }
  }

  static void _widenBf16(Uint16List src, Float32List dst) {
    final bits = Uint32List.sublistView(dst);
    for (var i = 0; i < src.length; i++) {
      bits[i] = src[i] << 16;
    }
  }

  static void _readFully(
    RandomAccessFile file,
    Uint8List into,
    int position, [
    int? length,
  ]) {
    final end = length ?? into.length;
    file.setPositionSync(position);
    var done = 0;
    while (done < end) {
      final n = file.readIntoSync(into, done, end);
      if (n <= 0) {
        throw FileSystemException('unexpected end of file', file.path);
      }
      done += n;
    }
  }
}

/// Writes float32 [tensors] (name: shape and values) to a new safetensors
/// file at [path].
void writeSafetensorsF32(
  String path,
  Map<String, (List<int>, Float32List)> tensors,
) {
  final header = <String, Object?>{
    '__metadata__': {'format': 'pt'},
  };
  var offset = 0;
  for (final MapEntry(key: name, value: (shape, values)) in tensors.entries) {
    header[name] = {
      'dtype': 'F32',
      'shape': shape,
      'data_offsets': [offset, offset + values.lengthInBytes],
    };
    offset += values.lengthInBytes;
  }
  final out = File(path).openSync(mode: FileMode.write);
  try {
    SafetensorsFile._writeHeader(out, header);
    for (final (_, values) in tensors.values) {
      out.writeFromSync(
        values.buffer.asUint8List(values.offsetInBytes, values.lengthInBytes),
      );
    }
  } finally {
    out.closeSync();
  }
}
