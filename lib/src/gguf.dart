/// Reading llama.cpp's GGUF files, and importing Liquid's quantization-aware
/// distilled (QAD) Q4_0 LFM3 into this engine's bundle layout.
///
/// GGUF: magic `GGUF`, version, tensor and metadata counts, metadata as
/// typed key-value pairs, tensor infos (name, dims with the fastest first,
/// type, offset), then tensor data from the next multiple of
/// `general.alignment` (32 by default). Only the types LFM3's files use are
/// supported: F32, Q4_0 and Q6_K.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'config.dart';
import 'native.dart';
import 'q8_cache.dart';
import 'safetensors.dart';

enum GgufType {
  f32(0, 1, 4),
  q4_0(2, 32, 18),
  q6K(14, 256, 210);

  const GgufType(this.id, this.blockValues, this.blockBytes);

  final int id;
  final int blockValues;
  final int blockBytes;
}

final class GgufTensor {
  GgufTensor(this.name, this.dims, this.type, this.offset);

  final String name;

  /// Fastest-varying first: a [rows, cols] matrix is `[cols, rows]`.
  final List<int> dims;
  final GgufType type;

  /// Absolute offset of the data in the file.
  final int offset;

  int get values => dims.fold(1, (a, b) => a * b);
  int get bytes => values ~/ type.blockValues * type.blockBytes;
}

final class GgufFile {
  GgufFile._(this._file, this.metadata, this.tensors);

  /// Maps and parses [path]. Needs the native kernels (for the mapping).
  factory GgufFile.open(String path) {
    final file =
        MappedFile.open(path) ??
        (throw FileSystemException('cannot map', path));
    final data = ByteData.sublistView(file.bytes(0, file.length));
    var at = 0;
    // Each reader returns the value at the cursor and moves past it.
    T read<T>(int size, T Function(int offset) get) {
      at += size;
      return get(at - size);
    }

    int u32() => read(4, (o) => data.getUint32(o, Endian.little));
    int u64() => read(8, (o) => data.getUint64(o, Endian.little));
    String str() {
      final n = u64();
      at += n;
      return utf8.decode(file.bytes(at - n, n), allowMalformed: true);
    }

    if (utf8.decode(file.bytes(0, 4)) != 'GGUF') {
      throw FormatException('$path is not GGUF');
    }
    at = 4;
    final version = u32();
    if (version < 2) throw FormatException('GGUF version $version');
    final tensorCount = u64();
    final kvCount = u64();

    // Scalars as their values; arrays are skipped and kept as their length.
    Object? value(int type) {
      switch (type) {
        case 0 || 1 || 7:
          return read(1, data.getUint8);
        case 2 || 3:
          return read(2, (o) => data.getUint16(o, Endian.little));
        case 4:
          return u32();
        case 5:
          return read(4, (o) => data.getInt32(o, Endian.little));
        case 6:
          return read(4, (o) => data.getFloat32(o, Endian.little));
        case 8:
          return str();
        case 9:
          final elementType = u32();
          final n = u64();
          for (var i = 0; i < n; i++) {
            value(elementType);
          }
          return n;
        case 10:
          return u64();
        case 11:
          return read(8, (o) => data.getInt64(o, Endian.little));
        case 12:
          return read(8, (o) => data.getFloat64(o, Endian.little));
      }
      throw FormatException('GGUF metadata type $type');
    }

    final metadata = <String, Object?>{};
    for (var i = 0; i < kvCount; i++) {
      final key = str();
      metadata[key] = value(u32());
    }
    final infos = <(String, List<int>, int, int)>[];
    for (var i = 0; i < tensorCount; i++) {
      final name = str();
      final dims = [for (var d = u32(); d > 0; d--) u64()];
      infos.add((name, dims, u32(), u64()));
    }
    final alignment = (metadata['general.alignment'] ?? 32) as int;
    final dataStart = (at + alignment - 1) ~/ alignment * alignment;
    final tensors = <String, GgufTensor>{};
    for (final (name, dims, typeId, offset) in infos) {
      final type = GgufType.values.where((t) => t.id == typeId).firstOrNull;
      if (type == null) continue; // a type this importer never needs
      tensors[name] = GgufTensor(name, dims, type, dataStart + offset);
    }
    return GgufFile._(file, metadata, tensors);
  }

  final MappedFile _file;
  final Map<String, Object?> metadata;
  final Map<String, GgufTensor> tensors;

  Uint8List bytes(GgufTensor t) => _file.bytes(t.offset, t.bytes);

  GgufTensor tensor(String name, GgufType type, List<int> dims) {
    final t = tensors[name] ?? (throw StateError('GGUF lacks $name'));
    if (t.type != type || t.dims.join(',') != dims.join(',')) {
      throw StateError('$name is ${t.type.name} ${t.dims}, expected $dims');
    }
    return t;
  }

  /// An F32 tensor's values, copied.
  Float32List f32(GgufTensor t) =>
      Float32List.fromList(Float32List.sublistView(bytes(t)));
}

/// Q4_0 blocks (half scale, then 16 nibble bytes) split into the engine's
/// nibble and scale arrays (lab_native.c); the nibble order is GGUF's.
CachedMatrix splitQ4_0(Uint8List blocks, int rows, int cols) {
  final count = rows * cols ~/ 32;
  final qs = Uint8List(count * 16);
  final scales = Uint8List(count * 2);
  for (var b = 0; b < count; b++) {
    scales[2 * b] = blocks[18 * b];
    scales[2 * b + 1] = blocks[18 * b + 1];
    qs.setRange(16 * b, 16 * b + 16, blocks, 18 * b + 2);
  }
  return _RawMatrix('q4', rows, cols, qs, scales);
}

/// Dequantizes [count] Q6_K super-blocks (256 values each) starting at
/// block [first] into [out] at [offset], as ggml's dequantize_row_q6_K.
void dequantizeQ6K(
  Uint8List data,
  int first,
  int count,
  Float32List out,
  int offset,
) {
  final sc = Int8List.sublistView(data);
  for (var b = first; b < first + count; b++) {
    final base = b * 210;
    final d = halfToDouble(data[base + 208] | data[base + 209] << 8);
    var y = offset + (b - first) * 256;
    var ql = base, qh = base + 128, s = base + 192;
    for (var half = 0; half < 2; half++) {
      for (var l = 0; l < 32; l++) {
        final i = l ~/ 16;
        final h = data[qh + l];
        final q1 = (data[ql + l] & 15 | (h & 3) << 4) - 32;
        final q2 = (data[ql + l + 32] & 15 | (h >> 2 & 3) << 4) - 32;
        final q3 = (data[ql + l] >> 4 | (h >> 4 & 3) << 4) - 32;
        final q4 = (data[ql + l + 32] >> 4 | (h >> 6 & 3) << 4) - 32;
        out[y + l] = d * sc[s + i] * q1;
        out[y + l + 32] = d * sc[s + i + 2] * q2;
        out[y + l + 64] = d * sc[s + i + 4] * q3;
        out[y + l + 96] = d * sc[s + i + 6] * q4;
      }
      y += 128;
      ql += 64;
      qh += 32;
      s += 8;
    }
  }
}

final class _RawMatrix implements CachedMatrix {
  _RawMatrix(this.format, this.rows, this.cols, this.qsBytes, this.scaleBytes);

  @override
  final String format;
  @override
  final int rows, cols;
  @override
  final Uint8List qsBytes, scaleBytes;
}

/// Writes an engine bundle for the GGUF checkpoint [ggufPath] into
/// [outDir]: config.json and tokenizer.json from the Hugging Face directory
/// [hfDir] (the same model's), a model.safetensors with the float tensors
/// under their Hugging Face names, and `lfm3-q4.cache` holding every matrix
/// as Q4 and the tied embedding, Q6_K in the file, re-quantized to Q8 for
/// the LM head and token lookup. Load it with `--weights=q4`.
///
/// Returns lines comparing each imported tensor family with [hfDir]'s
/// weights (relative L2 difference over layers 0-3), which shows the layout
/// is read right: a transposed or reordered tensor differs by about 1.4.
List<String> importGguf(String ggufPath, String hfDir, String outDir) {
  final gguf = GgufFile.open(ggufPath);
  final c = Lfm3Config.load(hfDir);
  final h = c.hiddenSize, f = c.intermediateSize, hd = c.headDim;
  if (gguf.metadata['general.architecture'] != 'lfm3' ||
      gguf.metadata['lfm3.block_count'] != c.numLayers) {
    throw StateError('$ggufPath is not this LFM3 model');
  }

  final small = <String, (List<int>, Float32List)>{};
  // The cache's matrices, made one at a time while it is written: on a
  // phone the whole set (1.1 GB) does not fit in memory at once.
  final shapes = <(String, String, int, int)>[];
  final produce = <String, CachedMatrix Function()>{};
  // Imported tensors keyed by their Hugging Face name, for the comparison.
  final imported = <String, Float32List Function()>{};

  void vector(String gguf_, String hf, List<int> shape) {
    final values = gguf.f32(
      gguf.tensor(gguf_, GgufType.f32, shape.reversed.toList()),
    );
    small[hf] = (shape, values);
    imported[hf] = () => values;
  }

  void matrix(
    String gguf_,
    String cacheName,
    String hfName,
    int rows,
    int cols,
  ) {
    final t = gguf.tensor(gguf_, GgufType.q4_0, [cols, rows]);
    CachedMatrix split() => splitQ4_0(gguf.bytes(t), rows, cols);
    shapes.add((cacheName, 'q4', rows, cols));
    produce[cacheName] = split;
    imported[hfName] = () => _dequantizeQ4(split());
  }

  for (var i = 0; i < c.numLayers; i++) {
    final g = 'blk.$i.';
    final p = 'model.layers.$i.';
    vector('${g}attn_norm.weight', '${p}operator_norm.weight', [h]);
    vector('${g}post_attention_norm.weight', '${p}post_operator_norm.weight', [
      h,
    ]);
    vector('${g}ffn_norm.weight', '${p}ffn_norm.weight', [h]);
    vector('${g}post_ffw_norm.weight', '${p}post_ffn_norm.weight', [h]);
    final ff = '${p}feed_forward.';
    matrix('${g}ffn_gate.weight', '${ff}w1', '${ff}w1.weight', f, h);
    matrix('${g}ffn_up.weight', '${ff}w3', '${ff}w3.weight', f, h);
    matrix('${g}ffn_down.weight', '${ff}w2', '${ff}w2.weight', h, f);
    if (c.layerTypes[i] == LayerType.conv) {
      final t = gguf.tensor('${g}shortconv.conv.weight', GgufType.f32, [
        c.convKernel,
        h,
      ]);
      final values = gguf.f32(t);
      small['${p}conv.conv.weight'] = ([h, 1, c.convKernel], values);
      imported['${p}conv.conv.weight'] = () => values;
      for (final (gName, hfName, rows) in [
        ('shortconv.in_proj', 'conv.in_proj.weight', 3 * h),
        ('shortconv.out_proj', 'conv.out_proj.weight', h),
      ]) {
        matrix('$g$gName.weight', '$p$hfName', '$p$hfName', rows, h);
      }
    } else {
      final a = '${p}self_attn.';
      vector('${g}attn_q_norm.weight', '${a}q_norm.weight', [hd]);
      vector('${g}attn_k_norm.weight', '${a}k_norm.weight', [hd]);
      for (final (gName, hfName, rows, cols) in [
        ('attn_q', 'q_proj', c.numHeads * hd, h),
        ('attn_k', 'k_proj', c.kvDim, h),
        ('attn_v', 'v_proj', c.kvDim, h),
        ('attn_gate', 'gate_proj', c.numHeads, h),
        ('attn_output', 'o_proj', h, c.numHeads * hd),
      ]) {
        matrix('$g$gName.weight', '$a$hfName', '$a$hfName.weight', rows, cols);
      }
    }
  }
  vector('output_norm.weight', 'model.embedding_norm.weight', [h]);
  const embed = 'model.embed_tokens.weight';
  final embedding = gguf.tensor('token_embd.weight', GgufType.q6K, [
    h,
    c.vocabSize,
  ]);
  final embedBytes = gguf.bytes(embedding);
  final perRow = h ~/ 256;
  shapes.add((embed, 'q8', c.vocabSize, h));
  produce[embed] = () => NativeQ8Matrix.quantizeRows(
    c.vocabSize,
    h,
    (first, count, out) =>
        dequantizeQ6K(embedBytes, first * perRow, count * perRow, out, 0),
  );
  imported[embed] = () {
    const rows = 4096;
    final out = Float32List(rows * h);
    dequantizeQ6K(embedBytes, 0, rows * perRow, out, 0);
    return out;
  };

  Directory(outDir).createSync(recursive: true);
  for (final name in ['config.json', 'tokenizer.json']) {
    File('$hfDir/$name').copySync('$outDir/$name');
  }
  writeSafetensorsF32('$outDir/model.safetensors', small);
  Q8Cache.writeEach(
    '$outDir/lfm3-q4.cache',
    Q8Cache.keyFor(outDir, weights: 'q4'),
    shapes,
    (name) => produce[name]!(),
    done: (m) {
      if (m is NativeQ8Matrix) m.free();
    },
  );
  return _compare(hfDir, c, imported);
}

Float32List _dequantizeQ4(CachedMatrix m) {
  final out = Float32List(m.rows * m.cols);
  final q = m.qsBytes, s = m.scaleBytes;
  for (var b = 0; b < out.length ~/ 32; b++) {
    final d = halfToDouble(s[2 * b] | s[2 * b + 1] << 8);
    for (var k = 0; k < 16; k++) {
      final byte = q[16 * b + k];
      out[32 * b + k] = d * ((byte & 15) - 8);
      out[32 * b + k + 16] = d * ((byte >> 4) - 8);
    }
  }
  return out;
}

/// Relative L2 differences from the Hugging Face weights, per tensor family
/// (layer numbers folded), over layers 0-3 and the first 4096 embedding
/// rows; empty if [hfDir] has no model.safetensors.
List<String> _compare(
  String hfDir,
  Lfm3Config c,
  Map<String, Float32List Function()> imported,
) {
  final path = '$hfDir/model.safetensors';
  if (!File(path).existsSync()) return [];
  final st = SafetensorsFile.open(path);
  try {
    final sums = <String, (double, double)>{};
    for (final MapEntry(key: name, value: read) in imported.entries) {
      final layer = RegExp(r'layers\.(\d+)\.').firstMatch(name);
      if (layer != null && int.parse(layer[1]!) > 3) continue;
      final info = st.tensors[name];
      if (info == null) continue;
      final ours = read();
      final rows = info.shape.first;
      final rowLength = info.numel ~/ rows;
      final theirs = st.readF32(
        name,
        info.shape,
        rowEnd: math.min(rows, ours.length ~/ rowLength),
      );
      var diff = 0.0, norm = 0.0;
      for (var j = 0; j < theirs.length; j++) {
        final d = ours[j] - theirs[j];
        diff += d * d;
        norm += theirs[j] * theirs[j];
      }
      final family = name.replaceAll(RegExp(r'layers\.\d+\.'), 'layers.N.');
      final (sd, sn) = sums[family] ?? (0.0, 0.0);
      sums[family] = (sd + diff, sn + norm);
    }
    return [
      for (final MapEntry(key: family, value: (d, n)) in sums.entries)
        '${(math.sqrt(d / n)).toStringAsExponential(2)}  $family',
    ]..sort();
  } finally {
    st.close();
  }
}
