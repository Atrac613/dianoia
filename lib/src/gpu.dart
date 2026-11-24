/// Whole LFM3 tokens on the GPU: Metal on Apple platforms
/// (src/metal/lab_metal.m), Vulkan on Android and Windows
/// (src/vulkan/lab_vulkan.c).
/// Both export the same C API, `lab_gpu_*`, and the build hook compiles the
/// one for the target into the kernels asset, so this file drives either.
///
/// Stage 5b measured a GPU round trip at about 130 us, so the GPU only pays
/// off when whole tokens go over at once. Dart still runs the show: per
/// chunk of tokens (one when decoding, up to [GpuForward.maxChunk] for a
/// prompt) it writes the embedding rows and RoPE tables into shared memory,
/// encodes every kernel of the forward pass through dart:ffi (one call per
/// kernel) and commits one command buffer. A chunk's matrix products read
/// each weight once per 8 tokens.
///
/// The Q8 or Q4 weights are the mapped cache file, wrapped as GPU memory
/// (without a copy on Metal; on Vulkan a copy, after which the cache's
/// pages are evicted); each matrix is a pair of byte offsets into it. The small
/// float tensors (norm weights, conv kernels) are packed into one buffer.
/// Activations, conv histories and the KV caches are shared-storage
/// buffers, so Dart reads captured hidden states in place; the logits it
/// reads every token are copied out first (see [GpuForward.logits]).
@DefaultAsset('package:dianoia/kernels')
library;

import 'dart:ffi';
import 'dart:math' as math;
import 'dart:typed_data';

import 'config.dart';
import 'control_vector.dart';
import 'native.dart';
import 'ops.dart';
import 'q8_cache.dart';
import 'safetensors.dart';

@Native<Int64 Function()>(symbol: 'lab_gpu_init')
external int _init();

@Native<Int64 Function()>(symbol: 'lab_gpu_max_buffer_length', isLeaf: true)
external int _maxBufferLength();

@Native<Int64 Function()>(symbol: 'lab_gpu_max_shared_memory', isLeaf: true)
external int _maxThreadgroupMemory();

@Native<Int64 Function()>(
  symbol: 'lab_gpu_recommended_working_set',
  isLeaf: true,
)
external int _recommendedWorkingSet();

@Native<Int64 Function()>(symbol: 'lab_gpu_thermal_state', isLeaf: true)
external int _thermalState();

@Native<Int64 Function()>(symbol: 'lab_gpu_physical_memory', isLeaf: true)
external int _physicalMemory();

@Native<Void Function(Pointer<Uint8>, Int64)>(
  symbol: 'lab_gpu_device_name',
  isLeaf: true,
)
external void _deviceName(Pointer<Uint8> out, int capacity);

@Native<Int64 Function(Int64)>(symbol: 'lab_gpu_buffer', isLeaf: true)
external int _buffer(int bytes);

@Native<Int64 Function(Int64)>(symbol: 'lab_gpu_readback_buffer', isLeaf: true)
external int _readbackBuffer(int bytes);

@Native<Int64 Function(Pointer<Void>, Int64)>(
  symbol: 'lab_gpu_wrap',
  isLeaf: true,
)
external int _wrap(Pointer<Void> address, int bytes);

@Native<Int64 Function()>(symbol: 'lab_gpu_wrap_copies', isLeaf: true)
external int _wrapCopies();

@Native<Pointer<Void> Function(Int64)>(symbol: 'lab_gpu_contents', isLeaf: true)
external Pointer<Void> _contents(int handle);

@Native<Void Function(Int64)>(symbol: 'lab_gpu_release')
external void _release(int handle);

@Native<Int64 Function()>(symbol: 'lab_gpu_live_buffers', isLeaf: true)
external int _liveBuffers();

@Native<Void Function(Int64)>(symbol: 'lab_gpu_set_decode_rows', isLeaf: true)
external void _setDecodeRows(int rows);

@Native<Int64 Function(Int64)>(symbol: 'lab_gpu_set_q8_group', isLeaf: true)
external int _setQ8Group(int invocations);

@Native<Int64 Function(Int64)>(symbol: 'lab_gpu_set_q8_vector', isLeaf: true)
external int _setQ8Vector(int on);

@Native<Int64 Function(Int64, Int64, Int64, Int64, Int64)>(
  symbol: 'lab_gpu_register_q8_head',
  isLeaf: true,
)
external int _registerQ8Head(int w, int qs, int rows, int cols, int q4Body);

@Native<Void Function(Int64)>(symbol: 'lab_gpu_set_profiling', isLeaf: true)
external void _setProfiling(int on);

@Native<Int64 Function(Pointer<Double>, Pointer<Int64>, Int64)>(
  symbol: 'lab_gpu_profile',
  isLeaf: true,
)
external int _profile(Pointer<Double> ms, Pointer<Int64> counts, int kinds);

@Native<Double Function()>(symbol: 'lab_gpu_last_ms', isLeaf: true)
external double _lastGpuMs();

@Native<Void Function()>(symbol: 'lab_gpu_begin', isLeaf: true)
external void _begin();

// Not a leaf: it blocks until the GPU finishes.
@Native<Int64 Function(Int64)>(symbol: 'lab_gpu_commit')
external int _commit(int wait);

// Not a leaf: it blocks until the GPU finishes.
@Native<Int64 Function()>(symbol: 'lab_gpu_wait')
external int _waitGpu();

@Native<
  Void Function(
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
  )
>(symbol: 'lab_gpu_enc_matmul_q8', isLeaf: true)
external void _matmulQ8(
  int w,
  int qsOffset,
  int scalesOffset,
  int x,
  int xOffset,
  int y,
  int yOffset,
  int rows,
  int cols,
  int n,
);

@Native<
  Void Function(
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
  )
>(symbol: 'lab_gpu_enc_matmul_q4', isLeaf: true)
external void _matmulQ4(
  int w,
  int qsOffset,
  int scalesOffset,
  int x,
  int xOffset,
  int y,
  int yOffset,
  int rows,
  int cols,
  int n,
);

@Native<
  Void Function(
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Double,
    Int64,
  )
>(symbol: 'lab_gpu_enc_rms_norm', isLeaf: true)
external void _rmsNorm(
  int x,
  int xOffset,
  int w,
  int wOffset,
  int out,
  int outOffset,
  int rows,
  int cols,
  double eps,
  int accumulate,
);

@Native<
  Void Function(
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Double,
  )
>(symbol: 'lab_gpu_enc_head_norm_rope', isLeaf: true)
external void _headNormRope(
  int x,
  int xOffset,
  int xStride,
  int w,
  int wOffset,
  int rope,
  int ropeOffset,
  int ropeStride,
  int heads,
  int hd,
  int n,
  double eps,
);

@Native<Void Function(Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64)>(
  symbol: 'lab_gpu_enc_conv',
  isLeaf: true,
)
external void _conv(
  int bcx,
  int weight,
  int weightOffset,
  int history,
  int y,
  int width,
  int k,
  int n,
);

@Native<
  Void Function(
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
    Int64,
  )
>(symbol: 'lab_gpu_enc_attention', isLeaf: true)
external void _attention(
  int q,
  int keys,
  int values,
  int gate,
  int out,
  int position,
  int n,
  int heads,
  int kvHeads,
  int hd,
  int window,
);

@Native<Void Function(Int64, Int64, Int64)>(
  symbol: 'lab_gpu_enc_swiglu',
  isLeaf: true,
)
external void _swiglu(int g, int u, int count);

@Native<Void Function(Int64, Int64, Int64, Int64, Int64)>(
  symbol: 'lab_gpu_enc_copy',
  isLeaf: true,
)
external void _copy(int src, int srcOffset, int dst, int dstOffset, int count);

@Native<Void Function(Int64, Int64, Int64, Int64, Int64)>(
  symbol: 'lab_gpu_enc_add',
  isLeaf: true,
)
external void _add(int src, int srcOffset, int dst, int dstOffset, int count);

@Native<Void Function(Int64, Int64, Int64, Int64, Int64, Int64, Int64, Double)>(
  symbol: 'lab_gpu_enc_steer',
  isLeaf: true,
)
external void _steer(
  int x,
  int xOffset,
  int v,
  int vOffset,
  int rows,
  int cols,
  int mode,
  double scale,
);

/// A control vector on the GPU: its applied vectors, one row of hidden
/// size per layer (zero for layers it does not steer), and which layers it
/// steers.
final class _Steering {
  _Steering(this.buffer, this.layers, this.mode, this.scale);

  final _Buffer buffer;
  final List<bool> layers;
  final int mode;
  final double scale;
}

/// A Q8 or Q4 matrix: the buffer holding it and byte offsets into it.
final class _Matrix {
  const _Matrix(
    this.q4,
    this.buffer,
    this.qs,
    this.scales,
    this.rows,
    this.cols,
  );

  final bool q4;
  final int buffer, qs, scales, rows, cols;

  /// Encodes Y = X W^T for [n] tokens of [x] into [y] at [yOffset].
  void encode(int x, int y, int yOffset, int n) => (q4 ? _matmulQ4 : _matmulQ8)(
    buffer,
    qs,
    scales,
    x,
    0,
    y,
    yOffset,
    rows,
    cols,
    n,
  );
}

final class _Layer {
  _Layer({
    required this.operatorNorm,
    required this.postOperatorNorm,
    required this.ffnNorm,
    required this.postFfnNorm,
    required this.w1,
    required this.w3,
    required this.w2,
  });

  /// Byte offsets of the norm weights in the small-tensor buffer; the
  /// post norms are null without sandwich norms.
  final int operatorNorm, ffnNorm;
  final int? postOperatorNorm, postFfnNorm;
  final _Matrix w1, w3, w2;

  // Conv layers.
  _Matrix? inProj, outProj;
  int convKernel = 0; // byte offset in the small-tensor buffer
  int history = -1; // buffer handle

  // Attention layers.
  _Matrix? q, k, v, gate, o;
  int qNorm = 0, kNorm = 0; // byte offsets in the small-tensor buffer
  int keys = -1, values = -1; // buffer handles
  bool sliding = false;
}

/// A shared-storage buffer: its handle and a float view of its memory. One
/// the host reads after every forward pass is a [readback] buffer: on a
/// discrete GPU it lives in the host's memory, since the host reads the
/// GPU's own only uncached.
final class _Buffer {
  _Buffer(int floats, {bool readback = false})
    : handle = _check(
        (readback ? _readbackBuffer : _buffer)(floats * 4),
        floats * 4,
      ),
      length = floats;

  static int _check(int handle, int bytes) => handle >= 0
      ? handle
      : throw StateError('GPU buffer of $bytes bytes failed');

  final int handle;
  final int length;
  late final Float32List view = _contents(
    handle,
  ).cast<Float>().asTypedList(length);
}

final class GpuForward {
  GpuForward._(this.config, this.maxSeq, this.maxChunk, this.steers);

  /// Starts the GPU of the kernels asset, or with [libPath] of the library
  /// there, and prepares [cache]'s weights for it. The native CPU kernels
  /// must already be loaded: they map the cache.
  factory GpuForward.create({
    String? libPath,
    required String modelDir,
    required TextModelConfig config,
    required Q8Cache cache,
    required int maxSeq,
    required int maxChunk,
  }) {
    if (libPath != null) DynamicLibrary.open(libPath);
    if (_init() != 1) throw StateError('the GPU is unavailable');
    if (config.convKernel > 16) {
      throw UnsupportedError('the GPU path implements conv kernels up to 16');
    }
    // Attention keeps one score per key in threadgroup memory, next to the
    // kernel's own 128 bytes of scratch.
    if (maxSeq * 4 + 128 > _maxThreadgroupMemory()) {
      throw ArgumentError('--max-seq=$maxSeq is too long for GPU attention');
    }
    // A GPU library from elsewhere (the lab's, for parity checks) may
    // predate control vectors.
    final steers =
        libPath == null ||
        DynamicLibrary.open(libPath).providesSymbol('lab_gpu_enc_steer');
    final gpu = GpuForward._(config, maxSeq, maxChunk, steers);
    try {
      gpu._load(modelDir, cache);
    } catch (_) {
      gpu.dispose();
      rethrow;
    }
    // Vulkan copied the weights into GPU memory, so the cache's pages are a
    // second copy of the model. Give them back: the CPU's few reads of the
    // cache, such as embedding rows, fault their pages back in. A library
    // from elsewhere that predates the query wraps without a copy (Metal).
    final tells =
        libPath == null ||
        DynamicLibrary.open(libPath).providesSymbol('lab_gpu_wrap_copies');
    if (tells && _wrapCopies() != 0) cache.file.evict();
    return gpu;
  }

  /// GPU buffers in the process not yet released, for leak checks.
  static int get liveBuffers => _liveBuffers();

  /// Whether the GPU library applies control vectors ([setControlVector]).
  final bool steers;

  _Steering? _steering;

  /// Applies [vector] in every following [forward], after each layer it
  /// steers and before that layer's capture, or stops with null. Uploads its
  /// vectors to a buffer of its own, released when replaced, cleared or
  /// disposed. Callers make sure no state computed under another vector is
  /// reused (TextModel.setControlVector).
  void setControlVector(ControlVector? vector) {
    if (_disposed) throw StateError('the GPU forward pass is disposed');
    final c = config;
    if (vector != null) {
      if (!steers) {
        throw UnsupportedError('this GPU library has no lab_gpu_enc_steer');
      }
      if (vector.hiddenSize != c.hiddenSize || vector.maxLayer >= c.numLayers) {
        throw ArgumentError('the control vector does not fit this model');
      }
    }
    if (_steering case final old?) {
      _steering = null;
      _owned.remove(old.buffer.handle);
      _release(old.buffer.handle);
    }
    if (vector == null) return;
    final h = c.hiddenSize;
    final buffer = _newBuffer(c.numLayers * h);
    final layers = List.filled(c.numLayers, false);
    for (var l = 0; l < c.numLayers; l++) {
      final v = vector.appliedVector(l);
      if (v == null) continue;
      buffer.view.setAll(l * h, v);
      layers[l] = true;
    }
    _steering = _Steering(
      buffer,
      layers,
      vector.mode == ControlMode.add ? 0 : 1,
      vector.scale,
    );
  }

  /// The buffers this forward pass made, which [dispose] releases.
  final _owned = <int>[];
  bool _disposed = false;

  _Buffer _newBuffer(int floats, {bool readback = false}) {
    final buffer = _Buffer(floats, readback: readback);
    _owned.add(buffer.handle);
    return buffer;
  }

  /// Releases every GPU buffer this forward pass made. The cache it wraps
  /// stays mapped: unmap it after this, never before. Using the forward
  /// pass afterwards throws.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _owned.forEach(_release);
    _owned.clear();
    _steering = null;
  }

  /// The GPU's name, once a [GpuForward] exists.
  static String get deviceName {
    final scratch = _Buffer._check(_buffer(256), 256);
    try {
      final out = _contents(scratch).cast<Uint8>();
      _deviceName(out, 256);
      final bytes = out.asTypedList(256);
      return String.fromCharCodes(bytes.sublist(0, bytes.indexOf(0)));
    } finally {
      _release(scratch);
    }
  }

  /// Device facts for a measurement record, once a [GpuForward] exists.
  static Map<String, Object> get deviceFacts => {
    'gpu': deviceName,
    'physical_memory': _physicalMemory(),
    'max_buffer_length': _maxBufferLength(),
    'recommended_working_set': _recommendedWorkingSet(),
    'max_threadgroup_memory': _maxThreadgroupMemory(),
  };

  /// The kernel kinds [takeProfile] reports, in the C layer's order.
  static const kernelKinds = [
    'gemv_q8',
    'gemm_q8',
    'gemv_q4',
    'gemm_q4',
    'rms_norm',
    'head_norm_rope',
    'conv',
    'attention',
    'swiglu',
    'copy',
    'quantize',
    'gemm_q8_i8',
    'gemm_q4_i8',
    'gemv_q8_i8',
    'gemv_q4_i8',
    'gemv_q8_i8_vec',
    'steer',
    'add',
  ];

  /// Rows per subgroup (SIMD group on Metal) of the decode kernels: 1, 2 or
  /// 4. Zero selects the reference: the old int8 GEMM on Vulkan, one row on
  /// Metal. On Metal every setting computes the same bits. The default is 4
  /// on Vulkan and 2 on Metal ([restoreDecodeRowsDefault]).
  static set decodeRows(int rows) => _setDecodeRows(rows);

  /// Returns [decodeRows] to the backend's default.
  static void restoreDecodeRowsDefault() => _setDecodeRows(-1);

  /// Q8 decode workgroup size on Vulkan; false if unsupported (Metal too).
  static bool setQ8Workgroup(int invocations) =>
      _setQ8Group(invocations) == invocations;

  /// Overrides scoped automatic selection with scalar/vector Q8 loads.
  /// False if unavailable (including Metal).
  static bool setQ8Vector(bool on) => _setQ8Vector(on ? 1 : 0) == 1;

  /// Restores automatic selection for registered, validated LM heads.
  static bool restoreQ8VectorDefault() => _setQ8Vector(-1) == 1;

  /// GPU copy throughput in GB/s, bytes read plus written, over buffers of
  /// [megabytes]: roughly what the matrix products could reach.
  static double copyBandwidth(int megabytes) {
    final floats = megabytes << 18;
    final from = _Buffer(floats), to = _Buffer(floats);
    var best = 0.0;
    try {
      for (var i = 0; i < 3; i++) {
        _begin();
        _copy(from.handle, 0, to.handle, 0, floats);
        if (_commit(1) != 1) throw StateError('GPU command buffer failed');
        best = math.max(best, 2 * floats * 4 / (_lastGpuMs() / 1000) / 1e9);
      }
    } finally {
      _release(from.handle);
      _release(to.handle);
    }
    return best;
  }

  /// Times each kind of decode matrix across every layer that has one, so
  /// that, unlike [benchmarkDecodeMatrices], no product reads a matrix the
  /// GPU just read: a kind's per-layer matrices one after another, [rounds]
  /// times in one command buffer, the LM head [rounds] times alone. Each
  /// sample is one layout: [rowsVariants] rows per subgroup (0 is the GEMM
  /// path) at each of [workgroups] invocations per Q8 workgroup, the
  /// layouts in a rotating order. The GPU milliseconds are the products'
  /// own, from the per-dispatch profile, without the input quantization.
  /// Each sample's outputs for the last matrix it ran are compared with
  /// the default layout's (the decode rows' default at 128 invocations),
  /// bit for bit.
  List<Map<String, Object?>> benchmarkDecodeShapes({
    int rounds = 4,
    List<int> rowsVariants = const [1, 2, 4],
    List<int> workgroups = const [64, 128, 256],
    void Function(Map<String, Object?> sample)? onSample,
  }) {
    final kinds = <String, List<_Matrix>>{
      for (final (name, pick) in <(String, _Matrix? Function(_Layer))>[
        ('q_proj', (l) => l.q),
        ('k_proj', (l) => l.k),
        ('v_proj', (l) => l.v),
        ('o_proj', (l) => l.o),
        ('ffn_gate', (l) => l.w1),
        ('ffn_up', (l) => l.w3),
        ('ffn_down', (l) => l.w2),
        ('conv_in', (l) => l.inProj),
        ('conv_out', (l) => l.outProj),
      ])
        name: [for (final l in _layers) ?pick(l)],
      'lm_head': [_lmHead],
    }..removeWhere((_, matrices) => matrices.isEmpty);
    final all = kinds.values.expand((m) => m);
    final x = _newBuffer(all.map((w) => w.cols).reduce(math.max));
    final y = _newBuffer(all.map((w) => w.rows).reduce(math.max));
    for (var i = 0; i < x.length; i++) {
      x.view[i] = math.sin(i * 0.17) + 0.25 * math.cos(i * 0.07);
    }
    final layouts = [
      for (final rows in rowsVariants)
        for (final group in workgroups) (rows, group),
    ];
    void run(List<_Matrix> matrices, int times) {
      _begin();
      for (var t = 0; t < times; t++) {
        for (final w in matrices) {
          w.encode(x.handle, y.handle, 0, 1);
        }
      }
      if (_commit(1) != 1) {
        throw StateError('GPU shape benchmark failed');
      }
    }

    Uint32List output(_Matrix w) => Uint32List.fromList(
      y.view.buffer.asUint32List(y.view.offsetInBytes, w.rows),
    );

    final samples = <Map<String, Object?>>[];
    final references = <String, Uint32List>{};
    profiling = true;
    try {
      restoreDecodeRowsDefault();
      setQ8Workgroup(128);
      for (final MapEntry(key: kind, value: matrices) in kinds.entries) {
        run(matrices, 1);
        references[kind] = output(matrices.last);
      }
      for (var round = 0; round < 3; round++) {
        for (var i = 0; i < layouts.length; i++) {
          final (rows, group) = layouts[(i + round * 4) % layouts.length];
          decodeRows = rows;
          // Metal has one workgroup size: its layouts vary rows only.
          if (!setQ8Workgroup(group) && group != 128) continue;
          for (final MapEntry(key: kind, value: matrices) in kinds.entries) {
            run(matrices, 1); // warm this layout
            takeProfile();
            run(matrices, rounds);
            final profile = takeProfile();
            final gemv = profile.entries
                .where((e) => e.key.startsWith('gem'))
                .fold((
                  0.0,
                  0,
                ), (a, e) => (a.$1 + e.value.$1, a.$2 + e.value.$2));
            final w = matrices.last;
            final bits = output(w);
            final reference = references[kind]!;
            var differing = 0;
            for (var r = 0; r < w.rows; r++) {
              if (bits[r] != reference[r]) differing++;
            }
            final bytes = matrices.fold(
              0,
              (sum, m) =>
                  sum +
                  (m.q4 ? m.rows * m.cols * 9 ~/ 16 : m.rows * m.cols * 9 ~/ 8),
            );
            final ms = gemv.$1 / rounds;
            final sample = <String, Object?>{
              'kind': kind,
              'format': w.q4 ? 'q4' : 'q8',
              'matrix_rows': w.rows,
              'matrix_cols': w.cols,
              'matrices': matrices.length,
              'round': round + 1,
              'decode_rows': rows,
              'q8_workgroup': group,
              'gpu_ms_per_matrix': ms / matrices.length,
              'gpu_ms_per_token': ms,
              'gb_per_s': bytes / (ms / 1000) / 1e9,
              'dispatches': gemv.$2 / rounds,
              'differing_outputs': differing,
            };
            samples.add(sample);
            onSample?.call(sample);
          }
        }
      }
    } finally {
      restoreDecodeRowsDefault();
      setQ8Workgroup(128);
      profiling = false;
    }
    return samples;
  }

  /// Isolates representative real weight matrices from model orchestration.
  /// Each sample reuses fixed inputs for [repetitions] products in one
  /// command buffer (one input quantization). This is a cache-reuse probe,
  /// not an estimate of a complete token's memory traffic or latency.
  /// [q8WorkgroupSweep] instead isolates the LM head with four rows per
  /// subgroup and 64/128/256 invocations per workgroup, in six rounds.
  /// [q8VectorSweep] instead alternates scalar/vector loads at 128
  /// invocations, four rows per subgroup, for six rounds.
  List<Map<String, Object?>> benchmarkDecodeMatrices({
    int repetitions = 32,
    bool q8WorkgroupSweep = false,
    bool q8VectorSweep = false,
    void Function(Map<String, Object?> sample)? onSample,
  }) {
    if (repetitions < 1 || repetitions > 256) {
      throw ArgumentError.value(repetitions, 'repetitions', 'must be 1..256');
    }
    if (q8WorkgroupSweep && q8VectorSweep) {
      throw ArgumentError('choose one Q8 experiment');
    }
    final matrices = <String, _Matrix>{
      if (!q8WorkgroupSweep && !q8VectorSweep) 'ffn_up': _layers.first.w1,
      if (!q8WorkgroupSweep && !q8VectorSweep) 'ffn_down': _layers.first.w2,
      'lm_head': _lmHead,
    };
    final x = _newBuffer(matrices.values.map((w) => w.cols).reduce(math.max));
    final y = _newBuffer(matrices.values.map((w) => w.rows).reduce(math.max));
    for (var i = 0; i < x.length; i++) {
      x.view[i] = math.sin(i * 0.17) + 0.25 * math.cos(i * 0.07);
    }
    void run(_Matrix w, int count) {
      _begin();
      for (var i = 0; i < count; i++) {
        w.encode(x.handle, y.handle, 0, 1);
      }
      if (_commit(1) != 1) throw StateError('GPU matrix benchmark failed');
    }

    final samples = <Map<String, Object?>>[];
    final references = <String, Uint32List>{};
    profiling = false;
    try {
      setQ8Vector(false);
      setQ8Workgroup(128);
      decodeRows = 0;
      for (final entry in matrices.entries) {
        run(entry.value, 1);
        references[entry.key] = Uint32List.fromList(
          y.view.buffer.asUint32List(y.view.offsetInBytes, entry.value.rows),
        );
      }
      final orders = q8VectorSweep
          ? [
              [0, 1],
              [1, 0],
              [0, 1],
              [1, 0],
              [0, 1],
              [1, 0],
            ]
          : q8WorkgroupSweep
          ? [
              [64, 128, 256],
              [128, 256, 64],
              [256, 64, 128],
              [256, 128, 64],
              [128, 64, 256],
              [64, 256, 128],
            ]
          : [
              [0, 1, 2, 4],
              [1, 2, 4, 0],
              [2, 4, 0, 1],
              [4, 0, 1, 2],
            ];
      for (var round = 0; round < orders.length; round++) {
        for (final variant in orders[round]) {
          final rows = (q8WorkgroupSweep || q8VectorSweep) ? 4 : variant;
          final workgroup = q8WorkgroupSweep ? variant : 128;
          final vector = q8VectorSweep && variant == 1;
          if (q8VectorSweep && !setQ8Vector(vector)) continue;
          if (q8WorkgroupSweep && !setQ8Workgroup(workgroup)) continue;
          decodeRows = rows;
          for (final entry in matrices.entries) {
            final w = entry.value;
            run(w, 2); // warm this layout and matrix before timing
            final wall = Stopwatch()..start();
            run(w, repetitions);
            final wallMs = wall.elapsedMicroseconds / 1000 / repetitions;
            final gpuMs = _lastGpuMs() / repetitions;
            final bits = y.view.buffer.asUint32List(
              y.view.offsetInBytes,
              w.rows,
            );
            final reference = references[entry.key]!;
            var differing = 0, nonFinite = 0;
            for (var i = 0; i < w.rows; i++) {
              if (bits[i] != reference[i]) differing++;
              if (!y.view[i].isFinite) nonFinite++;
            }
            final sample = <String, Object?>{
              'matrix': entry.key,
              'format': w.q4 ? 'q4' : 'q8',
              'matrix_rows': w.rows,
              'matrix_cols': w.cols,
              'round': round + 1,
              'decode_rows': rows,
              'q8_workgroup': workgroup,
              'q8_vector': vector,
              'repetitions': repetitions,
              'gpu_ms_per_dispatch': gpuMs,
              'wall_ms_per_dispatch': wallMs,
              'differing_outputs': differing,
              'non_finite': nonFinite,
            };
            samples.add(sample);
            onSample?.call(sample);
          }
        }
      }
    } finally {
      restoreDecodeRowsDefault();
      restoreQ8VectorDefault();
      setQ8Workgroup(128);
      profiling = false;
    }
    return samples;
  }

  /// Times every dispatch on the GPU while on, for [takeProfile]. Vulkan
  /// writes timestamps around each; Metal, which has none inside a command
  /// buffer, commits and waits for every dispatch separately, so a forward
  /// pass takes far longer on the wall clock while this is on.
  static set profiling(bool on) => _setProfiling(on ? 1 : 0);

  /// GPU milliseconds and dispatch counts per kernel kind since the last
  /// call; empty if nothing was dispatched while [profiling].
  static Map<String, (double, int)> takeProfile() {
    final n = kernelKinds.length;
    final ms = NativeKernels.allocate<Double>(n * 8);
    final counts = NativeKernels.allocate<Int64>(n * 8);
    final kinds = _profile(ms, counts, n);
    return {
      for (var k = 0; k < kinds && k < n; k++)
        if (counts[k] > 0) kernelKinds[k]: (ms[k], counts[k]),
    };
  }

  /// 0 nominal, 1 fair, 2 serious, 3 critical.
  static int get thermalState => _thermalState();

  /// How many GPU buffers the weights were split into.
  int get weightSegments => _segments;
  int _segments = 0;

  final TextModelConfig config;
  final int maxSeq;

  /// Most tokens one [forward] call takes.
  final int maxChunk;

  late final _Buffer _small;
  late final List<_Layer> _layers;
  late final int _finalNorm;
  late final _Matrix _lmHead;

  /// Whether this instance's Q4-cache LM head is registered for the
  /// validated Pixel 8 vector-load default. Overrides still take precedence.
  late final bool q8VectorHeadDefaultEligible;

  late final _Buffer _x, _a, _m, _bcx, _y, _q, _gate, _attnOut, _g, _u;
  late final _Buffer _normed, _logits, _rope, _residualIn, _capture;
  late final Rope _fullRope = Rope(config.fullRope, config.headDim);
  late final Rope _slidingRope = Rope(config.slidingRope, config.headDim);

  /// The residual stream entering the next [forward], [maxChunk] rows of
  /// hidden size: write the tokens' embeddings here.
  Float32List get residual => _x.view;

  /// Logits of the last token of the last [forward] with logits, or of
  /// [logitsOf], copied out of the GPU's buffer into ordinary memory. On a
  /// phone that buffer is uncached for the CPU: an argmax reading Qwen3's
  /// 151,936 logits there one by one took 21 ms a token on a Pixel 8,
  /// against 1 ms for this bulk copy and 0.3 ms for the argmax after it.
  Float32List get logits => _hostLogits;
  late final Float32List _hostLogits = Float32List(config.vocabSize);

  void _copyLogits() => _hostLogits.setAll(0, _logits.view);

  /// Final-normalized state of the last token computed with logits.
  Float32List get lastHiddenState => _normed.view;

  /// After a [forward] of n tokens with capture: the residual entering
  /// layer 0 (slot 0) and leaving each layer (slot i + 1), n rows each.
  Float32List captured(int slot, int n) {
    final rows = maxChunk * config.hiddenSize;
    return Float32List.sublistView(
      _capture.view,
      slot * rows,
      slot * rows + n * config.hiddenSize,
    );
  }

  /// Totals since the last [clearStats]: [forward] calls and tokens.
  int calls = 0, tokens = 0, dispatches = 0, encodeMicros = 0, gpuMicros = 0;

  void clearStats() =>
      calls = tokens = dispatches = encodeMicros = gpuMicros = 0;

  void _load(String modelDir, Q8Cache cache) {
    final c = config;
    final h = c.hiddenSize;
    final hd = c.headDim;
    final segmentOf = _wrapWeights(cache);

    _Matrix matrix(String name, int rows, int cols) {
      final o = cache.offsets(name, rows, cols);
      if (o == null) throw StateError('cache lacks $name [$rows, $cols]');
      final (buffer, start) = segmentOf[name]!;
      return _Matrix(
        o.format == 'q4',
        buffer,
        o.qs - start,
        o.scales - start,
        rows,
        cols,
      );
    }

    // Small float tensors, packed into one buffer at 16-byte aligned
    // offsets: first collected, then copied in.
    final small = <Float32List>[];
    var smallFloats = 0;
    int place(Float32List values) {
      final offset = smallFloats * 4;
      small.add(values);
      smallFloats += (values.length + 3) ~/ 4 * 4;
      return offset;
    }

    final st = SafetensorsFile.open('$modelDir/model.safetensors');
    try {
      _layers = [];
      for (var i = 0; i < c.numLayers; i++) {
        final p = 'model.layers.$i.';
        final f = '${p}feed_forward.';
        final layer = _Layer(
          operatorNorm: place(st.readF32('${p}operator_norm.weight', [h])),
          postOperatorNorm: c.sandwichNorm
              ? place(st.readF32('${p}post_operator_norm.weight', [h]))
              : null,
          ffnNorm: place(st.readF32('${p}ffn_norm.weight', [h])),
          postFfnNorm: c.sandwichNorm
              ? place(st.readF32('${p}post_ffn_norm.weight', [h]))
              : null,
          w1: matrix('${f}w1', c.intermediateSize, h),
          w3: matrix('${f}w3', c.intermediateSize, h),
          w2: matrix('${f}w2', h, c.intermediateSize),
        );
        if (c.layerTypes[i] == LayerType.conv) {
          layer
            ..inProj = matrix('${p}conv.in_proj.weight', 3 * h, h)
            ..outProj = matrix('${p}conv.out_proj.weight', h, h)
            ..convKernel = place(
              st.readF32('${p}conv.conv.weight', [h, 1, c.convKernel]),
            )
            ..history = _newBuffer((c.convKernel - 1) * h).handle;
        } else {
          final a = '${p}self_attn.';
          layer
            ..q = matrix('${a}q_proj', c.numHeads * hd, h)
            ..k = matrix('${a}k_proj', c.kvDim, h)
            ..v = matrix('${a}v_proj', c.kvDim, h)
            ..gate = c.attentionOutputGate
                ? matrix('${a}gate_proj', c.numHeads, h)
                : null
            ..o = matrix('${a}o_proj', h, c.numHeads * hd)
            ..qNorm = place(st.readF32('${a}q_norm.weight', [hd]))
            ..kNorm = place(st.readF32('${a}k_norm.weight', [hd]))
            ..keys = _newBuffer(maxSeq * c.kvDim).handle
            ..values = _newBuffer(maxSeq * c.kvDim).handle
            ..sliding = c.layerTypes[i] == LayerType.slidingAttention;
        }
        _layers.add(layer);
      }
      _finalNorm = place(st.readF32('model.embedding_norm.weight', [h]));
    } finally {
      st.close();
    }
    _small = _newBuffer(smallFloats);
    var at = 0;
    for (final values in small) {
      _small.view.setAll(at, values);
      at += (values.length + 3) ~/ 4 * 4;
    }
    _lmHead = matrix('model.embed_tokens.weight', c.vocabSize, h);
    final q4Body = cache.shapes.every((shape) {
      final (name, rows, cols) = shape;
      return cache.offsets(name, rows, cols)!.format ==
          (name == 'model.embed_tokens.weight' ? 'q8' : 'q4');
    });
    q8VectorHeadDefaultEligible =
        _registerQ8Head(
          _lmHead.buffer,
          _lmHead.qs,
          _lmHead.rows,
          _lmHead.cols,
          q4Body ? 1 : 0,
        ) ==
        1;

    // Activations of a chunk: maxChunk rows each.
    final heads = c.numHeads * hd;
    final rows = maxChunk;
    _x = _newBuffer(rows * h);
    _a = _newBuffer(rows * h);
    _m = _newBuffer(rows * h);
    _bcx = _newBuffer(rows * 3 * h);
    _y = _newBuffer(rows * h);
    _q = _newBuffer(rows * heads);
    _gate = _newBuffer(rows * c.numHeads);
    _attnOut = _newBuffer(rows * heads);
    _g = _newBuffer(rows * c.intermediateSize);
    _u = _newBuffer(rows * c.intermediateSize);
    _normed = _newBuffer(h);
    _logits = _newBuffer(c.vocabSize, readback: true);
    // Per token: [full cos, full sin, sliding cos, sliding sin], hd / 2 each.
    _rope = _newBuffer(rows * 2 * hd);
    _residualIn = _newBuffer(h);
    _capture = _newBuffer((c.numLayers + 1) * rows * h, readback: true);
  }

  /// Wraps the mapped cache as GPU buffers: one buffer
  /// if the device allows one that large (a Mac does), otherwise
  /// consecutive page-aligned segments, each holding whole matrices.
  /// Returns each matrix's buffer handle and that buffer's file offset.
  Map<String, (int, int)> _wrapWeights(Q8Cache cache) {
    const page = 16384;
    final limit = _maxBufferLength();
    final matrices = [
      for (final (name, rows, cols) in cache.shapes)
        (name, cache.offsets(name, rows, cols)!),
    ]..sort((a, b) => a.$2.qs.compareTo(b.$2.qs));
    final result = <String, (int, int)>{};
    var pending = <String>[];
    var start = 0, end = 0;
    void flush() {
      if (pending.isEmpty) return;
      final handle = _wrap(
        (cache.file.address + start).cast(),
        math.min(end, cache.file.length) - start,
      );
      if (handle < 0) throw StateError('wrapping the weight cache failed');
      _owned.add(handle);
      _segments++;
      for (final name in pending) {
        result[name] = (handle, start);
      }
      pending = [];
    }

    for (final (name, offsets) in matrices) {
      final matrixEnd = offsets.end;
      if (pending.isNotEmpty && matrixEnd - start > limit) flush();
      if (pending.isEmpty) start = offsets.qs ~/ page * page;
      if (matrixEnd - start > limit) {
        throw StateError('$name alone exceeds the largest GPU buffer');
      }
      pending.add(name);
      end = matrixEnd;
    }
    flush();
    return result;
  }

  /// Clears the conv histories for a new sequence. KV cache rows are
  /// written before they are read, so they need no reset.
  void reset() {
    for (final layer in _layers) {
      if (layer.history >= 0) {
        final n = (config.convKernel - 1) * config.hiddenSize;
        _contents(
          layer.history,
        ).cast<Float>().asTypedList(n).fillRange(0, n, 0);
      }
    }
  }

  /// A copy of the conv histories (see [LayerShard.saveState]).
  List<Float32List> saveState() => [
    for (final layer in _layers)
      if (layer.history >= 0) Float32List.fromList(_history(layer)),
  ];

  /// Puts back conv histories from [saveState].
  void restoreState(List<Float32List> state) {
    var i = 0;
    for (final layer in _layers) {
      if (layer.history >= 0) _history(layer).setAll(0, state[i++]);
    }
  }

  /// Views of the state the first [position] tokens left behind, in the
  /// order of LayerShard.stateBlocks: conv histories, then each attention
  /// layer's key and value rows before [position].
  List<Float32List> stateBlocks(int position) {
    final rows = position * config.kvDim;
    Float32List view(int handle) =>
        _contents(handle).cast<Float>().asTypedList(rows);
    return [
      for (final layer in _layers)
        if (layer.history >= 0) _history(layer),
      for (final layer in _layers)
        if (layer.keys >= 0) ...[view(layer.keys), view(layer.values)],
    ];
  }

  Float32List _history(_Layer layer) => _contents(
    layer.history,
  ).cast<Float>().asTypedList((config.convKernel - 1) * config.hiddenSize);

  /// Runs [n] tokens at positions [position], ... whose embeddings are
  /// already in [residual]: every layer, and with [withLogits] the final
  /// norm and LM head of the last token into [logits]. With [capture], the
  /// residual stream is copied after every layer (see [captured]).
  ///
  /// With [meanwhile] the command buffer is committed without waiting, then
  /// [meanwhile] runs on the CPU while the GPU computes, and the pass waits
  /// for the GPU afterwards, also when [meanwhile] throws. It must not use
  /// the GPU.
  void forward(
    int n,
    int position, {
    required bool withLogits,
    required bool capture,
    void Function()? meanwhile,
  }) {
    if (_disposed) throw StateError('the GPU forward pass is disposed');
    if (n < 1 || n > maxChunk) throw ArgumentError('chunk of $n tokens');
    if (position + n > maxSeq) throw StateError('position exceeds maxSeq');
    final sw = Stopwatch()..start();
    final c = config;
    final h = c.hiddenSize;
    final hd = c.headDim;
    final eps = c.normEps;
    final pair = hd ~/ 2;
    final rope = _rope.view;
    for (var t = 0; t < n; t++) {
      _fullRope.setPosition(position + t);
      _slidingRope.setPosition(position + t);
      final at = t * 4 * pair;
      rope
        ..setAll(at, _fullRope.cos)
        ..setAll(at + pair, _fullRope.sin)
        ..setAll(at + 2 * pair, _slidingRope.cos)
        ..setAll(at + 3 * pair, _slidingRope.sin);
    }

    final steering = _steering;
    var dispatched = 0;
    void matmul(_Matrix w, _Buffer x, int y, [int yOffset = 0]) {
      w.encode(x.handle, y, yOffset, n);
      dispatched++;
    }

    void norm(_Buffer x, int w, _Buffer out, {bool accumulate = false}) {
      _rmsNorm(
        x.handle,
        0,
        _small.handle,
        w,
        out.handle,
        0,
        n,
        h,
        eps,
        accumulate ? 1 : 0,
      );
      dispatched++;
    }

    // Adds the sublayer output in _m to the residual stream, through the
    // post norm [w] where the layer has sandwich norms.
    void residual(int? w) {
      if (w != null) {
        norm(_m, w, _x, accumulate: true);
      } else {
        _add(_m.handle, 0, _x.handle, 0, n * h);
        dispatched++;
      }
    }

    void snapshot(int slot) {
      _copy(_x.handle, 0, _capture.handle, slot * maxChunk * h * 4, n * h);
      dispatched++;
    }

    void headNormRope(
      int x,
      int xOffset,
      int xStride,
      int w,
      int heads,
      int ropeOffset,
    ) {
      _headNormRope(
        x,
        xOffset,
        xStride,
        _small.handle,
        w,
        _rope.handle,
        ropeOffset,
        4 * pair,
        heads,
        hd,
        n,
        eps,
      );
      dispatched++;
    }

    _begin();
    if (capture) snapshot(0);
    for (var i = 0; i < _layers.length; i++) {
      final l = _layers[i];
      norm(_x, l.operatorNorm, _a);
      if (l.inProj != null) {
        matmul(l.inProj!, _a, _bcx.handle);
        _conv(
          _bcx.handle,
          _small.handle,
          l.convKernel,
          l.history,
          _y.handle,
          h,
          c.convKernel,
          n,
        );
        dispatched++;
        matmul(l.outProj!, _y, _m.handle);
      } else {
        // K and V go straight into the chunk's cache rows.
        final rows = position * c.kvDim * 4;
        final ropeOffset = (l.sliding ? 2 * pair : 0) * 4;
        matmul(l.q!, _a, _q.handle);
        matmul(l.k!, _a, l.keys, rows);
        matmul(l.v!, _a, l.values, rows);
        if (l.gate case final gate?) matmul(gate, _a, _gate.handle);
        headNormRope(
          _q.handle,
          0,
          c.numHeads * hd,
          l.qNorm,
          c.numHeads,
          ropeOffset,
        );
        headNormRope(l.keys, rows, c.kvDim, l.kNorm, c.numKvHeads, ropeOffset);
        _attention(
          _q.handle,
          l.keys,
          l.values,
          l.gate == null ? -1 : _gate.handle,
          _attnOut.handle,
          position,
          n,
          c.numHeads,
          c.numKvHeads,
          hd,
          l.sliding ? c.slidingWindow : 0,
        );
        dispatched++;
        matmul(l.o!, _attnOut, _m.handle);
      }
      residual(l.postOperatorNorm);

      norm(_x, l.ffnNorm, _a);
      matmul(l.w1, _a, _g.handle);
      matmul(l.w3, _a, _u.handle);
      _swiglu(_g.handle, _u.handle, n * c.intermediateSize);
      dispatched++;
      matmul(l.w2, _g, _m.handle);
      residual(l.postFfnNorm);
      // The residual leaving layer i is complete: steer it before the
      // capture, the next layer and the LM head read it.
      if (steering != null && steering.layers[i]) {
        _steer(
          _x.handle,
          0,
          steering.buffer.handle,
          i * h * 4,
          n,
          h,
          steering.mode,
          steering.scale,
        );
        dispatched++;
      }
      if (capture) snapshot(i + 1);
    }
    if (withLogits) {
      _rmsNorm(
        _x.handle,
        (n - 1) * h * 4,
        _small.handle,
        _finalNorm,
        _normed.handle,
        0,
        1,
        h,
        eps,
        0,
      );
      _lmHead.encode(_normed.handle, _logits.handle, 0, 1);
      dispatched += 2;
    }
    encodeMicros += sw.elapsedMicroseconds;
    if (meanwhile == null) {
      _finish();
    } else {
      if (_commit(0) != 1) throw StateError('GPU command buffer failed');
      try {
        meanwhile();
      } finally {
        final ok = _waitGpu() == 1;
        gpuMicros += (_lastGpuMs() * 1000).round();
        if (!ok) throw StateError('GPU command buffer failed');
      }
    }
    if (withLogits) _copyLogits();
    calls++;
    tokens += n;
    dispatches += dispatched;
  }

  /// The final norm and LM head of [residualOut] (a residual leaving the
  /// last layer) into [logits].
  Float32List logitsOf(Float32List residualOut) {
    _residualIn.view.setAll(0, residualOut);
    _begin();
    _rmsNorm(
      _residualIn.handle,
      0,
      _small.handle,
      _finalNorm,
      _normed.handle,
      0,
      1,
      config.hiddenSize,
      config.normEps,
      0,
    );
    _lmHead.encode(_normed.handle, _logits.handle, 0, 1);
    _finish();
    _copyLogits();
    return logits;
  }

  void _finish() {
    if (_commit(1) != 1) throw StateError('GPU command buffer failed');
    gpuMicros += (_lastGpuMs() * 1000).round();
  }
}

/// Products of quantized matrices on the GPU for work outside the language
/// forward pass, such as the vision encoder's linear layers: each [apply]
/// copies the inputs in, runs one matrix product over all n rows in its own
/// command buffer and copies the outputs back. The GPU computes the same
/// products as the CPU kernels with float inputs, its sums in another
/// order, so the outputs differ from theirs in the last bits.
///
/// The GPU layer encodes one command buffer at a time per process: do not
/// call [apply] while another isolate runs a [GpuForward].
final class GpuLinear {
  GpuLinear._();

  /// Starts the GPU of the kernels asset, or returns null where it does not
  /// start (no GPU, or none the kernels support).
  static GpuLinear? start() => _init() == 1 ? GpuLinear._() : null;

  final _matrices = <String, _Matrix>{};
  final _owned = <int>[];
  _Buffer? _x, _y;
  bool _disposed = false;

  /// The names of the matrices [upload] copied in.
  Iterable<String> get names => _matrices.keys;

  /// Copies [matrix] (Q4 or Q8) into a GPU buffer of its own, under [name].
  void upload(String name, CachedMatrix matrix) {
    _checkOpen();
    final qs = matrix.qsBytes, scales = matrix.scaleBytes;
    final scalesAt = (qs.length + 63) ~/ 64 * 64;
    final bytes = scalesAt + scales.length;
    final handle = _Buffer._check(_buffer(bytes), bytes);
    _owned.add(handle);
    final view = _contents(handle).cast<Uint8>().asTypedList(bytes);
    view
      ..setAll(0, qs)
      ..setAll(scalesAt, scales);
    final previous = _matrices[name];
    if (previous != null) {
      _owned.remove(previous.buffer);
      _release(previous.buffer);
    }
    _matrices[name] = _Matrix(
      switch (matrix.format) {
        'q4' => true,
        'q8' => false,
        final f => throw ArgumentError('cannot multiply $f matrices'),
      },
      handle,
      0,
      scalesAt,
      matrix.rows,
      matrix.cols,
    );
  }

  /// ys[t * rows + r] = (W xs[t])[r] for t < n, W the matrix [upload]ed
  /// as [name] and xs [n, cols].
  void apply(String name, Float32List xs, Float32List ys, int n) {
    _checkOpen();
    final w =
        _matrices[name] ?? (throw ArgumentError('no GPU matrix named $name'));
    if (n < 1 || xs.length < n * w.cols || ys.length < n * w.rows) {
      throw ArgumentError('GPU product: bad shapes for n=$n');
    }
    final x = _x = _grow(_x, n * w.cols);
    final y = _y = _grow(_y, n * w.rows);
    x.view.setRange(0, n * w.cols, xs);
    _begin();
    w.encode(x.handle, y.handle, 0, n);
    if (_commit(1) != 1) throw StateError('GPU command buffer failed');
    ys.setRange(0, n * w.rows, y.view);
  }

  /// [buffer], or a larger one replacing it, of at least [floats].
  _Buffer _grow(_Buffer? buffer, int floats) {
    if (buffer != null && buffer.length >= floats) return buffer;
    if (buffer != null) {
      _owned.remove(buffer.handle);
      _release(buffer.handle);
    }
    final grown = _Buffer(floats);
    _owned.add(grown.handle);
    return grown;
  }

  void _checkOpen() {
    if (_disposed) throw StateError('the GPU products are disposed');
  }

  /// Releases every GPU buffer. Using this afterwards throws.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _owned.forEach(_release);
    _owned.clear();
    _matrices.clear();
    _x = _y = null;
  }
}
