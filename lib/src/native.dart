/// Native Q8 and Q4 matrix products through dart:ffi (stage 5).
///
/// The kernels live in src/c/lab_native.c, built by hook/build.dart into the
/// code asset `package:dianoia/kernels` that the `@Native` functions below
/// bind to. [NativeKernels.load] can instead open a library at a path, with
/// its symbols global, which is how `@Native` falls back to finding them
/// when no asset provides them: `dlopen` does that by default on Apple
/// platforms, and on Android it is asked for explicitly. `@Native` leaf
/// functions accept `TypedData.address`,
/// so activations stay in Dart lists and reach C without a copy; only the
/// weights live in native memory.
///
/// Q8 is llama.cpp's Q8_0 idea in a plain layout: per row, blocks of 32
/// weights share a float scale and store int8 q = round(w / scale). With
/// int8 inputs (the default) each product first quantizes its inputs per
/// block and multiplies int8 pairs with `sdot`; otherwise inputs stay float
/// and int8 weights are widened. Rows are split across native threads that
/// share memory, so there are no isolates and no messages.
///
/// Q4 is llama.cpp's Q4_0 (the format of Liquid's quantization-aware
/// distilled LFM3): per block of 32, a half scale d and 4-bit q, worth
/// d * (q - 8). Nibbles and scales are stored apart (lab_native.c).
@DefaultAsset('package:dianoia/kernels')
library;

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

@Native<Pointer<Void> Function(Int64)>(symbol: 'lab_alloc', isLeaf: true)
external Pointer<Void> _alloc(int bytes);

@Native<
  Void Function(Pointer<Float>, Pointer<Int8>, Pointer<Float>, Int64, Int64)
>(symbol: 'lab_quantize_q8', isLeaf: true)
external void _quantizeQ8(
  Pointer<Float> w,
  Pointer<Int8> qs,
  Pointer<Float> scales,
  int rows,
  int cols,
);

@Native<
  Void Function(
    Pointer<Int8>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Int8>,
    Pointer<Float>,
    Pointer<Float>,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
  )
>(symbol: 'lab_gemm_q8', isLeaf: true)
external void _gemmQ8(
  Pointer<Int8> qs,
  Pointer<Float> scales,
  Pointer<Float> xs,
  Pointer<Int8> xq,
  Pointer<Float> xsc,
  Pointer<Float> ys,
  int rows,
  int cols,
  int n,
  int threads,
  int mode,
);

@Native<
  Void Function(Pointer<Float>, Pointer<Uint8>, Pointer<Uint16>, Int64, Int64)
>(symbol: 'lab_quantize_q4', isLeaf: true)
external void _quantizeQ4(
  Pointer<Float> w,
  Pointer<Uint8> qs,
  Pointer<Uint16> scales,
  int rows,
  int cols,
);

@Native<
  Void Function(
    Pointer<Uint8>,
    Pointer<Uint16>,
    Pointer<Float>,
    Pointer<Int8>,
    Pointer<Float>,
    Pointer<Float>,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
  )
>(symbol: 'lab_gemm_q4', isLeaf: true)
external void _gemmQ4(
  Pointer<Uint8> qs,
  Pointer<Uint16> scales,
  Pointer<Float> xs,
  Pointer<Int8> xq,
  Pointer<Float> xsc,
  Pointer<Float> ys,
  int rows,
  int cols,
  int n,
  int threads,
  int mode,
);

@Native<Pointer<Void> Function(Pointer<Uint8>, Pointer<Int64>)>(
  symbol: 'lab_map_file',
  isLeaf: true,
)
external Pointer<Void> _mapFile(Pointer<Uint8> path, Pointer<Int64> size);

@Native<Int64 Function()>(symbol: 'lab_phys_footprint', isLeaf: true)
external int _physFootprint();

/// The process's physical footprint in bytes (what iOS holds against an
/// app's memory limit); on Linux and Android, its resident set size from
/// /proc; -1 where neither is available.
int physFootprint() {
  final footprint = _physFootprint();
  if (footprint >= 0 || !(Platform.isAndroid || Platform.isLinux)) {
    return footprint;
  }
  try {
    final status = File('/proc/self/status').readAsStringSync();
    final kb = RegExp(r'VmRSS:\s*(\d+)').firstMatch(status)?[1];
    return kb == null ? -1 : int.parse(kb) * 1024;
  } on FileSystemException {
    return -1;
  }
}

/// A file mapped read-only into memory by the native library. Its pages are
/// read on first touch and stay clean, so the OS can drop them under memory
/// pressure. Mappings live for the rest of the process, and opening a path
/// again returns its existing mapping: an iOS app's address space has room
/// for the 1.95 GB Q8 cache, but not for two copies of it.
final class MappedFile {
  MappedFile._(this.address, this.length);

  static final _mapped = <String, MappedFile>{};

  /// Maps [path], or returns null if it cannot be opened, is empty, or
  /// does not fit in the address space.
  static MappedFile? open(String path) {
    if (_mapped[path] case final file?) return file;
    final bytes = utf8.encode(path);
    final cPath = NativeKernels.allocate<Uint8>(bytes.length + 1);
    cPath.asTypedList(bytes.length + 1)
      ..setAll(0, bytes)
      ..[bytes.length] = 0;
    final size = NativeKernels.allocate<Int64>(8);
    final p = _mapFile(cPath, size);
    if (p == nullptr) return null;
    return _mapped[path] = MappedFile._(p.cast<Uint8>(), size.value);
  }

  final Pointer<Uint8> address;
  final int length;

  Uint8List bytes(int offset, int count) =>
      (address + offset).asTypedList(count);
}

@Native<
  Void Function(
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
  )
>(symbol: 'lab_attention', isLeaf: true)
external void _attention(
  Pointer<Float> q,
  Pointer<Float> keys,
  Pointer<Float> values,
  Pointer<Float> gate,
  Pointer<Float> out,
  int n,
  int position,
  int heads,
  int kvHeads,
  int headDim,
  int window,
  int threads,
);

/// Grouped-query attention of [n] tokens at positions [position], ... in C
/// (src/c/lab_native.c, `lab_attention`), with the per-head
/// output gate. Shapes as in LayerShard: q and out [n, heads * headDim],
/// keys and values [maxSeq, kvHeads * headDim] already holding this chunk,
/// gate [n, heads] before the sigmoid. [window] 0 is full attention.
void nativeAttention(
  Float32List q,
  Float32List keys,
  Float32List values,
  Float32List gate,
  Float32List out, {
  required int n,
  required int position,
  required int heads,
  required int kvHeads,
  required int headDim,
  required int window,
}) {
  final kvDim = kvHeads * headDim;
  if (q.length < n * heads * headDim ||
      out.length < n * heads * headDim ||
      gate.length < n * heads ||
      keys.length < (position + n) * kvDim ||
      values.length < (position + n) * kvDim ||
      heads % kvHeads != 0 ||
      headDim % 4 != 0) {
    throw ArgumentError('native attention: bad shapes');
  }
  _attention(
    q.address,
    keys.address,
    values.address,
    gate.address,
    out.address,
    n,
    position,
    heads,
    kvHeads,
    headDim,
    window,
    NativeKernels.instance.threads,
  );
}

/// Process-wide settings of the native kernels.
final class NativeKernels {
  NativeKernels._(this.threads, this.int8Inputs, this.attention);

  static NativeKernels? _instance;

  static NativeKernels get instance =>
      _instance ??
      (throw StateError('native kernels are not loaded (NativeKernels.load)'));

  static bool get isLoaded => _instance != null;

  /// Enables the kernels: those the package's build hook compiled, or with
  /// [path] the library there (for kernels built outside the package).
  /// [threads] native threads share each product; [int8Inputs] selects the
  /// sdot path; [attention] moves attention to C (see [nativeAttention]).
  static void load(
    String? path, {
    int threads = 4,
    bool int8Inputs = true,
    bool attention = true,
  }) {
    if (path == null) {
      if (!isAvailable) {
        throw StateError(
          'the native kernels are not built for this platform '
          '(${Abi.current()})',
        );
      }
    } else if (Platform.isAndroid) {
      _openGlobal(path);
    } else {
      DynamicLibrary.open(path);
    }
    _instance = NativeKernels._(threads, int8Inputs, attention);
  }

  /// Whether the kernels can be called: built for this platform by the
  /// build hook, or already opened from a path.
  static bool get isAvailable {
    try {
      _physFootprint();
      return true;
    } on ArgumentError {
      return false;
    }
  }

  /// Android's `dlopen` keeps a library's symbols local unless asked, so the
  /// process-wide lookup that `@Native` falls back to would miss them.
  static void _openGlobal(String path) {
    const rtldNow = 2, rtldGlobal = 0x100;
    final process = DynamicLibrary.process();
    final dlopen = process
        .lookupFunction<
          Pointer<Void> Function(Pointer<Uint8>, Int32),
          Pointer<Void> Function(Pointer<Uint8>, int)
        >('dlopen');
    final malloc = process
        .lookupFunction<
          Pointer<Void> Function(IntPtr),
          Pointer<Void> Function(int)
        >('malloc');
    final bytes = utf8.encode(path);
    final cPath = malloc(bytes.length + 1).cast<Uint8>();
    cPath.asTypedList(bytes.length + 1)
      ..setAll(0, bytes)
      ..[bytes.length] = 0;
    if (dlopen(cPath, rtldNow | rtldGlobal) == nullptr) {
      throw StateError('dlopen($path) failed');
    }
  }

  final int threads;
  final bool int8Inputs;
  final bool attention;

  // Input-quantization scratch, grown on demand.
  Pointer<Int8> _xq = nullptr;
  Pointer<Float> _xsc = nullptr;
  int _capacity = 0;

  void _ensureScratch(int values) {
    if (values <= _capacity) return;
    _xq = _alloc(values).cast<Int8>();
    _xsc = _alloc(values ~/ 32 * 4 + 4).cast<Float>();
    _capacity = values;
  }

  static Pointer<T> allocate<T extends NativeType>(int bytes) {
    final p = _alloc(bytes);
    if (p == nullptr) {
      throw StateError('native allocation of $bytes bytes failed');
    }
    return p.cast<T>();
  }
}

/// What a Q8 cache file stores for one matrix (q8_cache.dart).
abstract interface class CachedMatrix {
  /// `q8` or `q4`.
  String get format;
  int get rows;
  int get cols;

  /// The raw quantized values and scales, as written to the file.
  Uint8List get qsBytes;
  Uint8List get scaleBytes;
}

/// A quantized matrix in native memory that the C kernels multiply.
abstract interface class NativeMatrix implements CachedMatrix {
  /// ys[t * rows + r] = (W xs[t])[r] for t < n, xs as [n, cols].
  /// [int8Inputs] overrides the process-wide setting (for tests).
  void apply(Float32List xs, Float32List ys, int n, {bool? int8Inputs});

  /// The weight the kernel uses at (r, c), for tests.
  double weightAt(int r, int c);
}

/// A row-major [rows, cols] matrix quantized to Q8 in native memory.
final class NativeQ8Matrix implements NativeMatrix {
  NativeQ8Matrix._(this.rows, this.cols, this._qs, this._scales);

  factory NativeQ8Matrix.quantize(Float32List w, int rows, int cols) {
    if (cols % 32 != 0) {
      throw ArgumentError('Q8 needs cols to be a multiple of 32, got $cols');
    }
    final qs = NativeKernels.allocate<Int8>(rows * cols);
    final scales = NativeKernels.allocate<Float>(rows * cols ~/ 32 * 4);
    _quantizeQ8(w.address, qs, scales, rows, cols);
    return NativeQ8Matrix._(rows, cols, qs, scales);
  }

  /// Quantizes a matrix too large to hold in float32 at once: [fill] writes
  /// [count] rows starting at row [first] into its list, [chunk] at a time.
  factory NativeQ8Matrix.quantizeRows(
    int rows,
    int cols,
    void Function(int first, int count, Float32List out) fill, {
    int chunk = 1024,
  }) {
    final qs = NativeKernels.allocate<Int8>(rows * cols);
    final scales = NativeKernels.allocate<Float>(rows * cols ~/ 32 * 4);
    final buffer = Float32List(chunk * cols);
    for (var first = 0; first < rows; first += chunk) {
      final count = rows - first < chunk ? rows - first : chunk;
      fill(first, count, buffer);
      _quantizeQ8(
        buffer.address,
        qs + first * cols,
        scales + first * cols ~/ 32,
        count,
        cols,
      );
    }
    return NativeQ8Matrix._(rows, cols, qs, scales);
  }

  /// Wraps Q8 data that already exists in native memory, such as a mapped
  /// cache file: [qs] holds rows * cols int8, [scales] rows * cols / 32.
  factory NativeQ8Matrix.fromPointers(
    int rows,
    int cols,
    Pointer<Int8> qs,
    Pointer<Float> scales,
  ) => NativeQ8Matrix._(rows, cols, qs, scales);

  @override
  String get format => 'q8';
  @override
  final int rows;
  @override
  final int cols;
  final Pointer<Int8> _qs;
  final Pointer<Float> _scales;

  int get bytes => rows * cols + rows * cols ~/ 32 * 4;

  @override
  Uint8List get qsBytes => _qs.cast<Uint8>().asTypedList(rows * cols);
  @override
  Uint8List get scaleBytes =>
      _scales.cast<Uint8>().asTypedList(rows * cols ~/ 32 * 4);

  @override
  void apply(Float32List xs, Float32List ys, int n, {bool? int8Inputs}) {
    final k = NativeKernels.instance;
    if (xs.length < n * cols || ys.length < n * rows) {
      throw ArgumentError('Q8 product: bad shapes for n=$n');
    }
    k._ensureScratch(n * cols);
    _gemmQ8(
      _qs,
      _scales,
      xs.address,
      k._xq,
      k._xsc,
      ys.address,
      rows,
      cols,
      n,
      k.threads,
      (int8Inputs ?? k.int8Inputs) ? 1 : 0,
    );
  }

  @override
  double weightAt(int r, int c) =>
      _qs[r * cols + c] * _scales[(r * cols + c) ~/ 32];

  /// Writes row [r], dequantized, into [out] at [offset]: token lookup in
  /// an embedding that exists only as the tied, quantized LM head.
  void dequantizeRow(int r, Float32List out, int offset) {
    final q = (_qs + r * cols).asTypedList(cols);
    final s = (_scales + r * cols ~/ 32).asTypedList(cols ~/ 32);
    for (var c = 0; c < cols; c++) {
      out[offset + c] = q[c] * s[c >> 5];
    }
  }
}

/// A row-major [rows, cols] matrix in Q4 (Q4_0) in native memory.
final class NativeQ4Matrix implements NativeMatrix {
  NativeQ4Matrix._(this.rows, this.cols, this._qs, this._scales);

  /// Quantizes float weights with llama.cpp's Q4_0 rule.
  factory NativeQ4Matrix.quantize(Float32List w, int rows, int cols) {
    if (cols % 32 != 0) {
      throw ArgumentError('Q4 needs cols to be a multiple of 32, got $cols');
    }
    final qs = NativeKernels.allocate<Uint8>(rows * cols ~/ 2);
    final scales = NativeKernels.allocate<Uint16>(rows * cols ~/ 32 * 2);
    _quantizeQ4(w.address, qs, scales, rows, cols);
    return NativeQ4Matrix._(rows, cols, qs, scales);
  }

  /// Wraps Q4 data in native memory: [qs] holds rows * cols / 2 bytes,
  /// [scales] rows * cols / 32 halves.
  factory NativeQ4Matrix.fromPointers(
    int rows,
    int cols,
    Pointer<Uint8> qs,
    Pointer<Uint16> scales,
  ) => NativeQ4Matrix._(rows, cols, qs, scales);

  @override
  String get format => 'q4';
  @override
  final int rows;
  @override
  final int cols;
  final Pointer<Uint8> _qs;
  final Pointer<Uint16> _scales;

  @override
  Uint8List get qsBytes => _qs.asTypedList(rows * cols ~/ 2);
  @override
  Uint8List get scaleBytes =>
      _scales.cast<Uint8>().asTypedList(rows * cols ~/ 32 * 2);

  @override
  void apply(Float32List xs, Float32List ys, int n, {bool? int8Inputs}) {
    final k = NativeKernels.instance;
    if (xs.length < n * cols || ys.length < n * rows) {
      throw ArgumentError('Q4 product: bad shapes for n=$n');
    }
    k._ensureScratch(n * cols);
    _gemmQ4(
      _qs,
      _scales,
      xs.address,
      k._xq,
      k._xsc,
      ys.address,
      rows,
      cols,
      n,
      k.threads,
      (int8Inputs ?? k.int8Inputs) ? 1 : 0,
    );
  }

  @override
  double weightAt(int r, int c) {
    final i = r * cols + c;
    final block = i ~/ 32, k = i % 32;
    final byte = _qs[block * 16 + k % 16];
    final q = k < 16 ? byte & 15 : byte >> 4;
    return halfToDouble(_scales[block]) * (q - 8);
  }
}

/// An IEEE half-precision bit pattern as a double.
double halfToDouble(int bits) {
  final sign = bits & 0x8000 != 0 ? -1.0 : 1.0;
  final exponent = (bits >> 10) & 0x1f;
  final mantissa = bits & 0x3ff;
  if (exponent == 0) return sign * mantissa * 5.960464477539063e-8; // 2^-24
  if (exponent == 31) {
    return mantissa == 0 ? sign * double.infinity : double.nan;
  }
  return sign * (1 + mantissa / 1024) * math.pow(2, exponent - 15);
}
