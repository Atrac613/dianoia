/// One weight matrix, float32 or quantized, and its product kernels.
library;

import 'dart:typed_data';

import 'matmul.dart';
import 'native.dart';
import 'quant.dart';
import 'safetensors.dart';

final class Linear {
  Linear._(this.rows, this.cols, this._weights);

  /// Wraps a row-major [rows, cols] matrix, quantized to [format] unless
  /// that is f32.
  factory Linear.from(Float32List w, int rows, int cols, WeightFormat format) =>
      Linear._(rows, cols, switch (format) {
        _ when rows == 0 => w,
        WeightFormat.f32 => w,
        WeightFormat.q8 => NativeQ8Matrix.quantize(w, rows, cols),
        WeightFormat.q4 => NativeQ4Matrix.quantize(w, rows, cols),
        _ => QuantizedMatrix.quantize(w, rows, cols, format),
      });

  /// Wraps a native matrix that already exists (a mapped cache entry).
  factory Linear.native(NativeMatrix m) => Linear._(m.rows, m.cols, m);

  final int rows;
  final int cols;

  /// The native matrix behind this product, if it has one.
  NativeMatrix? get native => switch (_weights) {
    NativeMatrix m => m,
    _ => null,
  };

  /// A float32 [rows, cols] list, a [QuantizedMatrix] or a [NativeMatrix].
  final Object _weights;

  /// ys[t * rows + r] = (W xs[t])[r] for t < n, xs as [n, cols]. One vector
  /// uses a GEMV kernel, a batch a GEMM kernel.
  void apply(Float32List xs, Float32List ys, int n) {
    if (rows == 0) return;
    switch (_weights) {
      case Float32List w:
        n == 1 ? gemv(w, xs, ys, rows, cols) : gemm(w, xs, ys, rows, cols, n);
      case QuantizedMatrix q:
        n == 1 ? q.gemv(xs, ys) : q.gemm(xs, ys, n);
      case NativeMatrix q:
        q.apply(xs, ys, n);
    }
  }
}

/// Reads part of a 2-D [rows, cols] tensor as float32: the concatenation of
/// [rowRanges] (all rows by default), restricted to the columns
/// [colStart, colEnd).
Float32List readSlice(
  SafetensorsFile st,
  String name,
  int rows,
  int cols, {
  List<(int, int)>? rowRanges,
  int colStart = 0,
  int? colEnd,
}) {
  final ranges = rowRanges ?? [(0, rows)];
  final end = colEnd ?? cols;
  final width = end - colStart;
  final outRows = ranges.fold(0, (sum, r) => sum + r.$2 - r.$1);
  final out = Float32List(outRows * width);
  var row = 0;
  for (final (start, stop) in ranges) {
    if (stop == start) continue;
    final part = st.readF32(name, [rows, cols], rowStart: start, rowEnd: stop);
    if (width == cols) {
      out.setRange(row * width, (row + stop - start) * width, part);
    } else {
      for (var r = 0; r < stop - start; r++) {
        out.setRange(
          (row + r) * width,
          (row + r + 1) * width,
          part,
          r * cols + colStart,
        );
      }
    }
    row += stop - start;
  }
  return out;
}
