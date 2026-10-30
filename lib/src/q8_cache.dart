/// A file of native Q8 (and Q4) matrices, written once and memory-mapped
/// afterwards.
///
/// Quantizing LFM3 at start-up widened every bf16 tensor to float32 and
/// quantized it again: 4.8 s and a 5.2 GB peak on each start. The cache
/// stores the result. Mapping it costs no copy, and its pages stay clean,
/// so the OS can evict and re-read them instead of swapping.
///
/// Layout: magic `LFM3Q8C1`, a little-endian u64 header length, a JSON
/// header, zero padding to 4096 bytes, then each matrix's quantized values
/// and scales at 64-byte aligned offsets. An entry's `format` is `q8` (int8
/// values, float scales; the default, as in caches written before Q4) or
/// `q4` (nibbles, half scales; see native.dart). The header's key
/// fingerprints the checkpoint (safetensors header and size, config.json)
/// and the weight format asked for, so a cache built from another model,
/// revision or format is ignored.
library;

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'native.dart';

const _magic = 'LFM3Q8C1';
const _version = 1;

final class Q8Cache {
  Q8Cache._(this._file, this._entries);

  final MappedFile _file;
  final Map<String, Map<String, Object?>> _entries;

  /// Fingerprint of the checkpoint in [modelDir] and of this cache format
  /// for [weights] (`q8` or `q4`).
  static String keyFor(String modelDir, {String weights = 'q8'}) {
    final st = File('$modelDir/model.safetensors').openSync();
    try {
      final lengthBytes = st.readSync(8);
      final headerLength = ByteData.sublistView(
        lengthBytes,
      ).getUint64(0, Endian.little);
      var hash = _fnv(_fnvOffset, lengthBytes);
      hash = _fnv(hash, st.readSync(headerLength));
      hash = _fnv(hash, utf8.encode('${st.lengthSync()}'));
      hash = _fnv(hash, File('$modelDir/config.json').readAsBytesSync());
      // Q8 keeps its original key text, so existing Q8 caches stay valid.
      hash = _fnv(hash, utf8.encode('$_magic v$_version $weights block 32'));
      return hash.toUnsigned(64).toRadixString(16).padLeft(16, '0');
    } finally {
      st.closeSync();
    }
  }

  /// Maps [path] if it is a cache with [key]; otherwise returns null. A
  /// cache that exists but cannot be mapped is an error rather than a miss:
  /// a miss would quantize from model.safetensors, which a device bundle
  /// (`lfm3 pack`) does not hold.
  static Q8Cache? open(String path, String key) {
    final length = File(path).existsSync() ? File(path).lengthSync() : 0;
    if (length < 16) return null;
    final file =
        MappedFile.open(path) ??
        (throw StateError('mapping $path ($length bytes) failed'));
    if (utf8.decode(file.bytes(0, 8), allowMalformed: true) != _magic) {
      file.release();
      return null;
    }
    final headerLength = ByteData.sublistView(
      file.bytes(8, 8),
    ).getUint64(0, Endian.little);
    final header =
        jsonDecode(utf8.decode(file.bytes(16, headerLength)))
            as Map<String, Object?>;
    if (header['version'] != _version || header['key'] != key) {
      file.release();
      return null;
    }
    return Q8Cache._(file, {
      for (final MapEntry(:key, :value)
          in (header['tensors'] as Map<String, Object?>).entries)
        key: value as Map<String, Object?>,
    });
  }

  /// The mapping, for handing the whole file to the GPU in one piece.
  MappedFile get file => _file;

  /// Unmaps the cache once no other [open] of it holds the mapping. The
  /// matrices from [matrix] point into it and must not be used afterwards.
  void close() => _file.release();

  /// Every cached matrix: name, rows and columns.
  Iterable<(String, int, int)> get shapes => _entries.entries.map(
    (e) => (e.key, e.value['rows'] as int, e.value['cols'] as int),
  );

  /// Matrix [name]'s format and the byte offsets and lengths in [file] of
  /// its values and scales, or null if it is absent or of another shape.
  ({String format, int qs, int scales, int end})? offsets(
    String name,
    int rows,
    int cols,
  ) {
    final e = _entries[name];
    if (e == null || e['rows'] != rows || e['cols'] != cols) return null;
    final format = (e['format'] ?? 'q8') as String;
    final scales = e['scales'] as int;
    final scaleBytes = rows * cols ~/ 32 * (format == 'q4' ? 2 : 4);
    return (
      format: format,
      qs: e['qs'] as int,
      scales: scales,
      end: scales + scaleBytes,
    );
  }

  /// The cached matrix [name], or null if absent or of another shape.
  NativeMatrix? matrix(String name, int rows, int cols) {
    final o = offsets(name, rows, cols);
    if (o == null) return null;
    return switch (o.format) {
      'q8' => NativeQ8Matrix.fromPointers(
        rows,
        cols,
        (_file.address + o.qs).cast<Int8>(),
        (_file.address + o.scales).cast<Float>(),
      ),
      'q4' => NativeQ4Matrix.fromPointers(
        rows,
        cols,
        _file.address + o.qs,
        (_file.address + o.scales).cast<Uint16>(),
      ),
      _ => throw StateError('unknown format ${o.format} of $name'),
    };
  }

  /// Writes [matrices] to [path] under [key], through a temporary file and
  /// a rename so a reader never maps a half-written cache.
  static void write(
    String path,
    String key,
    Map<String, CachedMatrix> matrices,
  ) {
    int align(int v, int a) => (v + a - 1) ~/ a * a;
    // Data offsets depend on the header size, which depends on the offsets:
    // reserve a generous fixed header region instead.
    const headerRegion = 1 << 20;
    var offset = headerRegion;
    final tensors = <String, Object?>{};
    for (final MapEntry(:key, :value) in matrices.entries) {
      final qs = offset = align(offset, 64);
      offset += value.qsBytes.length;
      final scales = offset = align(offset, 64);
      offset += value.scaleBytes.length;
      tensors[key] = {
        'format': value.format,
        'rows': value.rows,
        'cols': value.cols,
        'qs': qs,
        'scales': scales,
      };
    }
    final header = utf8.encode(
      jsonEncode({'version': _version, 'key': key, 'tensors': tensors}),
    );
    if (16 + header.length > headerRegion) {
      throw StateError('Q8 cache header does not fit');
    }
    final temp = File('$path.partial');
    final out = temp.openSync(mode: FileMode.write);
    try {
      final prefix = ByteData(16);
      final magic = utf8.encode(_magic);
      for (var i = 0; i < 8; i++) {
        prefix.setUint8(i, magic[i]);
      }
      prefix.setUint64(8, header.length, Endian.little);
      out
        ..writeFromSync(prefix.buffer.asUint8List())
        ..writeFromSync(header);
      for (final MapEntry(:key, :value) in matrices.entries) {
        final e = tensors[key] as Map<String, Object?>;
        out
          ..setPositionSync(e['qs'] as int)
          ..writeFromSync(value.qsBytes)
          ..setPositionSync(e['scales'] as int)
          ..writeFromSync(value.scaleBytes);
      }
    } finally {
      out.closeSync();
    }
    temp.renameSync(path);
  }

  static const _fnvOffset = -0x340d631b7bdddcdb; // 0xcbf29ce484222325
  static const _fnvPrime = 0x100000001b3;

  /// 64-bit FNV-1a; Dart's int arithmetic wraps at 64 bits.
  static int _fnv(int hash, List<int> bytes) {
    var h = hash;
    for (final b in bytes) {
      h ^= b;
      h *= _fnvPrime;
    }
    return h;
  }
}
