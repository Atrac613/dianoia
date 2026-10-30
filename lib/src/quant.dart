/// Quantized weight matrices with kernels that Dart's SIMD can decode.
///
/// The formats come from the stage 3 experiment (stage3_quant/dart, see the
/// README). Each 32-bit lane packs `planes` unsigned fields of `bits` bits
/// inside the 23-bit mantissa range, and a field decodes without shifts:
///
///     u * 2^s = float_bits((word & mask) | 0x4B000000) - 2^23
///
/// Lane l, plane p of group g holds column g * 4 * planes + 4 * p + l.
/// Blocks of `groupsPerBlock` groups share an f32 scale, with
/// w = (u - offset) * scale. Columns are zero-padded to a whole block.
///
/// Decoding one vector ([gemv]) multiplies fields directly with an input
/// pre-scaled by 2^-s and removes the offset with per-block input sums. A
/// batch ([gemm]) instead dequantizes 16 rows at a time into a float tile
/// and reuses it for every vector, so the decode cost is paid once per
/// chunk rather than once per token.
library;

import 'dart:math' as math;
import 'dart:typed_data';

enum WeightFormat {
  /// Weights widened from the bf16 checkpoint and kept as float32.
  f32(0, 0, 0),

  /// Four 5-bit fields per lane, one scale per 64 weights: 8.5 bits.
  q5p4(5, 4, 4),

  /// Five 4-bit fields per lane, one scale per 80 weights: 6.8 bits.
  q4p5(4, 5, 4),

  /// Three 7-bit fields per lane, one scale per 48 weights: 11.3 bits.
  /// The only one of the three that keeps LFM3 close to float32 (KL 0.02 on
  /// the tools case, against 0.4 for q5p4; see tools/lfm3_quant_study.py).
  q7p3(7, 3, 4),

  /// int8 with one f32 scale per 32 weights (9 bits), in native memory and
  /// multiplied by the C kernels (native.dart). Needs --native-lib.
  q8(8, 0, 0),

  /// llama.cpp's Q4_0: 4-bit with one half scale per 32 weights (4.5 bits),
  /// native like [q8]. The tied LM head stays Q8, as Liquid's Q4_0 build
  /// keeps it at 6 bits.
  q4(4, 0, 0);

  /// Whether the format is decoded by Dart's SIMD kernels in this file.
  bool get isDartPacked => planes > 0;

  /// Whether the format lives in native memory and runs on the C kernels
  /// (and can be cached and run on the GPU).
  bool get isNative => this == q8 || this == q4;

  const WeightFormat(this.bits, this.planes, this.groupsPerBlock);

  final int bits;
  final int planes;
  final int groupsPerBlock;

  int get groupColumns => 4 * planes;
  int get blockColumns => groupColumns * groupsPerBlock;
  int get qmax => (1 << (bits - 1)) - 1;
  int get offset => 1 << (bits - 1);
}

final class QuantizedMatrix {
  QuantizedMatrix._(
    this.format,
    this.rows,
    this.cols,
    this.paddedCols,
    this._words,
    this._scales,
  ) : blocksPerRow = paddedCols ~/ format.blockColumns,
      groupsPerRow = paddedCols ~/ format.groupColumns;

  /// Symmetric per-block quantization of a row-major [rows, cols] matrix.
  factory QuantizedMatrix.quantize(
    Float32List w,
    int rows,
    int cols,
    WeightFormat format,
  ) {
    if (!format.isDartPacked || cols % 4 != 0) {
      throw ArgumentError('cannot quantize $rows x $cols as $format');
    }
    final block = format.blockColumns;
    final group = format.groupColumns;
    final padded = (cols + block - 1) ~/ block * block;
    final words = Int32List(rows * padded ~/ format.planes);
    final scales = Float32List(rows * padded ~/ block);
    final groupsPerRow = padded ~/ group;
    final blocksPerRow = padded ~/ block;
    for (var r = 0; r < rows; r++) {
      final row = r * cols;
      for (var b = 0; b < blocksPerRow; b++) {
        final end = math.min((b + 1) * block, cols);
        var maxAbs = 0.0;
        for (var c = b * block; c < end; c++) {
          maxAbs = math.max(maxAbs, w[row + c].abs());
        }
        final si = r * blocksPerRow + b;
        scales[si] = maxAbs == 0 ? 1.0 : maxAbs / format.qmax;
        final inverse = 1 / scales[si];
        for (
          var g = b * format.groupsPerBlock;
          g < (b + 1) * format.groupsPerBlock;
          g++
        ) {
          for (var p = 0; p < format.planes; p++) {
            for (var l = 0; l < 4; l++) {
              final c = g * group + 4 * p + l;
              final q = c < cols
                  ? (w[row + c] * inverse).round().clamp(
                      -format.qmax,
                      format.qmax,
                    )
                  : 0;
              words[(r * groupsPerRow + g) * 4 + l] |=
                  (q + format.offset) << (format.bits * p);
            }
          }
        }
      }
    }
    return QuantizedMatrix._(format, rows, cols, padded, words, scales);
  }

  final WeightFormat format;
  final int rows;
  final int cols;
  final int paddedCols;
  final int blocksPerRow;
  final int groupsPerRow;
  final Int32List _words;
  final Float32List _scales;

  int get bytes => _words.lengthInBytes + _scales.lengthInBytes;

  // Per-isolate scratch, grown on demand and reused across calls.
  Float32List _scaledX = Float32List(0);
  Float64List _blockSums = Float64List(0);
  Float32List _tile = Float32List(0);

  static const _tileRows = 16;

  /// y[0:rows] = W x.
  void gemv(Float32List x, Float32List y) {
    if (_scaledX.length < paddedCols) _scaledX = Float32List(paddedCols);
    if (_blockSums.length < blocksPerRow) {
      _blockSums = Float64List(blocksPerRow);
    }
    final group = format.groupColumns;
    final block = format.blockColumns;
    _blockSums.fillRange(0, blocksPerRow, 0);
    for (var c = 0; c < cols; c++) {
      final p = (c % group) ~/ 4;
      _scaledX[c] = x[c] / (1 << (format.bits * p));
      _blockSums[c ~/ block] += x[c];
    }
    _scaledX.fillRange(cols, paddedCols, 0);
    final xs = Float32x4List.sublistView(_scaledX, 0, paddedCols);
    switch (format.planes) {
      case 3:
        _gemv3(this, xs, _blockSums, y);
      case 4:
        _gemv4(this, xs, _blockSums, y);
      default:
        _gemv5(this, xs, _blockSums, y);
    }
  }

  /// ys[t] = W xs[t] for t < n, xs as [n, cols] and ys as [n, rows].
  void gemm(Float32List xs, Float32List ys, int n) {
    if (_tile.length < _tileRows * cols) _tile = Float32List(_tileRows * cols);
    final xv = Float32x4List.sublistView(xs, 0, n * cols);
    for (var r0 = 0; r0 < rows; r0 += _tileRows) {
      final count = math.min(_tileRows, rows - r0);
      _dequantize(this, r0, count, _tile);
      _tileProduct(
        Float32x4List.sublistView(_tile, 0, count * cols),
        count,
        xv,
        ys,
        rows,
        r0,
        cols ~/ 4,
        n,
      );
    }
  }

  /// The weight the kernels use at (r, c); for tests.
  double weightAt(int r, int c) {
    final group = format.groupColumns;
    final g = c ~/ group;
    final p = (c % group) ~/ 4;
    final l = c % 4;
    final word = _words[(r * groupsPerRow + g) * 4 + l];
    final u = (word >> (format.bits * p)) & ((1 << format.bits) - 1);
    return (u - format.offset) *
        _scales[r * blocksPerRow + c ~/ format.blockColumns];
  }

  List<Int32x4> _masks() => [
    for (var p = 0; p < format.planes; p++)
      () {
        final m = ((1 << format.bits) - 1) << (format.bits * p);
        return Int32x4(m, m, m, m);
      }(),
  ];
}

@pragma('vm:prefer-inline')
double _laneSum(Float32x4 v) {
  final t = v + v.shuffle(Float32x4.yxwz);
  return t.x + t.z;
}

// The magic constants are created inside each kernel. A top-level final is
// initialized lazily, and reading one in a hot loop compiled to an init
// check and a call that spilled every vector register (2.4x slower, stage 3).
@pragma('vm:prefer-inline')
Float32x4 _field(
  Int32x4 word,
  Int32x4 mask,
  Int32x4 magicBits,
  Float32x4 magic,
) => Float32x4.fromInt32x4Bits((word & mask) | magicBits) - magic;

@pragma('vm:unsafe:no-bounds-checks')
void _gemv3(
  QuantizedMatrix m,
  Float32x4List xs,
  Float64List sums,
  Float32List y,
) {
  final mb = Int32x4(0x4B000000, 0x4B000000, 0x4B000000, 0x4B000000);
  final mv = Float32x4.splat(8388608.0);
  final masks = m._masks();
  final m0 = masks[0], m1 = masks[1], m2 = masks[2];
  final wv = Int32x4List.sublistView(m._words);
  final scales = m._scales;
  final gpb = m.format.groupsPerBlock;
  final bpr = m.blocksPerRow;
  final offset = m.format.offset;
  for (var r = 0; r < m.rows; r++) {
    var acc = Float32x4.zero();
    var correction = 0.0;
    var word = r * m.groupsPerRow;
    var xi = 0;
    for (var b = 0; b < bpr; b++) {
      var s = Float32x4.zero();
      for (var g = 0; g < gpb; g++) {
        final w = wv[word++];
        s += _field(w, m0, mb, mv) * xs[xi];
        s += _field(w, m1, mb, mv) * xs[xi + 1];
        s += _field(w, m2, mb, mv) * xs[xi + 2];
        xi += 3;
      }
      final scale = scales[r * bpr + b];
      acc += s.scale(scale);
      correction += scale * sums[b];
    }
    y[r] = _laneSum(acc) - offset * correction;
  }
}

@pragma('vm:unsafe:no-bounds-checks')
void _gemv4(
  QuantizedMatrix m,
  Float32x4List xs,
  Float64List sums,
  Float32List y,
) {
  final mb = Int32x4(0x4B000000, 0x4B000000, 0x4B000000, 0x4B000000);
  final mv = Float32x4.splat(8388608.0);
  final masks = m._masks();
  final m0 = masks[0], m1 = masks[1], m2 = masks[2], m3 = masks[3];
  final wv = Int32x4List.sublistView(m._words);
  final scales = m._scales;
  final gpb = m.format.groupsPerBlock;
  final bpr = m.blocksPerRow;
  final offset = m.format.offset;
  for (var r = 0; r < m.rows; r++) {
    var acc = Float32x4.zero();
    var correction = 0.0;
    var word = r * m.groupsPerRow;
    var xi = 0;
    for (var b = 0; b < bpr; b++) {
      var s = Float32x4.zero();
      for (var g = 0; g < gpb; g++) {
        final w = wv[word++];
        s += _field(w, m0, mb, mv) * xs[xi];
        s += _field(w, m1, mb, mv) * xs[xi + 1];
        s += _field(w, m2, mb, mv) * xs[xi + 2];
        s += _field(w, m3, mb, mv) * xs[xi + 3];
        xi += 4;
      }
      final scale = scales[r * bpr + b];
      acc += s.scale(scale);
      correction += scale * sums[b];
    }
    y[r] = _laneSum(acc) - offset * correction;
  }
}

@pragma('vm:unsafe:no-bounds-checks')
void _gemv5(
  QuantizedMatrix m,
  Float32x4List xs,
  Float64List sums,
  Float32List y,
) {
  final mb = Int32x4(0x4B000000, 0x4B000000, 0x4B000000, 0x4B000000);
  final mv = Float32x4.splat(8388608.0);
  final masks = m._masks();
  final m0 = masks[0],
      m1 = masks[1],
      m2 = masks[2],
      m3 = masks[3],
      m4 = masks[4];
  final wv = Int32x4List.sublistView(m._words);
  final scales = m._scales;
  final gpb = m.format.groupsPerBlock;
  final bpr = m.blocksPerRow;
  final offset = m.format.offset;
  for (var r = 0; r < m.rows; r++) {
    var acc = Float32x4.zero();
    var correction = 0.0;
    var word = r * m.groupsPerRow;
    var xi = 0;
    for (var b = 0; b < bpr; b++) {
      var s = Float32x4.zero();
      for (var g = 0; g < gpb; g++) {
        final w = wv[word++];
        s += _field(w, m0, mb, mv) * xs[xi];
        s += _field(w, m1, mb, mv) * xs[xi + 1];
        s += _field(w, m2, mb, mv) * xs[xi + 2];
        s += _field(w, m3, mb, mv) * xs[xi + 3];
        s += _field(w, m4, mb, mv) * xs[xi + 4];
        xi += 5;
      }
      final scale = scales[r * bpr + b];
      acc += s.scale(scale);
      correction += scale * sums[b];
    }
    y[r] = _laneSum(acc) - offset * correction;
  }
}

/// Rows [r0, r0 + count) as floats into tile [count, cols]. A field
/// decodes as u * 2^(bits * p); scaling by scale * 2^-(bits * p) and
/// subtracting offset * scale gives the weight.
@pragma('vm:unsafe:no-bounds-checks')
void _dequantize(QuantizedMatrix m, int r0, int count, Float32List tile) {
  final mb = Int32x4(0x4B000000, 0x4B000000, 0x4B000000, 0x4B000000);
  final mv = Float32x4.splat(8388608.0);
  final masks = m._masks();
  final planes = m.format.planes;
  final bits = m.format.bits;
  final gpb = m.format.groupsPerBlock;
  final group = m.format.groupColumns;
  final cols = m.cols;
  final wv = Int32x4List.sublistView(m._words);
  final out = Float32x4List.sublistView(tile, 0, count * cols);
  final planeScale = Float64List(planes);
  for (var p = 0; p < planes; p++) {
    planeScale[p] = 1 / (1 << (bits * p));
  }
  for (var i = 0; i < count; i++) {
    final r = r0 + i;
    final rowOut = i * cols ~/ 4;
    for (var b = 0; b < m.blocksPerRow; b++) {
      final scale = m._scales[r * m.blocksPerRow + b];
      final bias = Float32x4.splat(m.format.offset * scale);
      for (var g = b * gpb; g < (b + 1) * gpb; g++) {
        final w = wv[r * m.groupsPerRow + g];
        for (var p = 0; p < planes; p++) {
          final c = g * group + 4 * p;
          if (c >= cols) break;
          out[rowOut + c ~/ 4] =
              _field(w, masks[p], mb, mv).scale(scale * planeScale[p]) - bias;
        }
      }
    }
  }
}

/// ys[t * stride + rowOffset + i] = tile[i] . xs[t] for i < tileRows and
/// t < n: two rows by four vectors per block, as in matmul.dart's gemm.
@pragma('vm:unsafe:no-bounds-checks')
void _tileProduct(
  Float32x4List wv,
  int tileRows,
  Float32x4List xv,
  Float32List ys,
  int stride,
  int rowOffset,
  int kv,
  int n,
) {
  final blocked = n - n % 4;
  for (var t0 = 0; t0 < blocked; t0 += 4) {
    final x0 = t0 * kv, x1 = x0 + kv, x2 = x1 + kv, x3 = x2 + kv;
    for (var r = 0; r + 1 < tileRows; r += 2) {
      final w0 = r * kv, w1 = w0 + kv;
      var a0 = Float32x4.zero(), a1 = Float32x4.zero();
      var a2 = Float32x4.zero(), a3 = Float32x4.zero();
      var b0 = Float32x4.zero(), b1 = Float32x4.zero();
      var b2 = Float32x4.zero(), b3 = Float32x4.zero();
      for (var j = 0; j < kv; j++) {
        final wa = wv[w0 + j], wb = wv[w1 + j];
        final xa = xv[x0 + j],
            xb = xv[x1 + j],
            xc = xv[x2 + j],
            xd = xv[x3 + j];
        a0 += wa * xa;
        a1 += wa * xb;
        a2 += wa * xc;
        a3 += wa * xd;
        b0 += wb * xa;
        b1 += wb * xb;
        b2 += wb * xc;
        b3 += wb * xd;
      }
      final o = rowOffset + r;
      ys[t0 * stride + o] = _laneSum(a0);
      ys[(t0 + 1) * stride + o] = _laneSum(a1);
      ys[(t0 + 2) * stride + o] = _laneSum(a2);
      ys[(t0 + 3) * stride + o] = _laneSum(a3);
      ys[t0 * stride + o + 1] = _laneSum(b0);
      ys[(t0 + 1) * stride + o + 1] = _laneSum(b1);
      ys[(t0 + 2) * stride + o + 1] = _laneSum(b2);
      ys[(t0 + 3) * stride + o + 1] = _laneSum(b3);
    }
  }
  // Leftover rows of an odd tile, and the last n % 4 vectors.
  for (var t = 0; t < n; t++) {
    final rStart = t < blocked ? tileRows - tileRows % 2 : 0;
    for (var r = rStart; r < tileRows; r++) {
      var acc = Float32x4.zero();
      for (var j = 0; j < kv; j++) {
        acc += wv[r * kv + j] * xv[t * kv + j];
      }
      ys[t * stride + rowOffset + r] = _laneSum(acc);
    }
  }
}
