/// LFM3 forward pass in float32: the main-isolate side.
///
/// The structure follows `modeling_lfm3.py` (see
/// docs/lfm3_1.7b_architecture.md). Every layer is
///
///     h += post_operator_norm(mixer(operator_norm(h)))
///     h += post_ffn_norm(ffn(ffn_norm(h)))
///
/// where the mixer is a gated short convolution or attention. This class
/// owns the residual stream, the norms and token lookup; the mixers and
/// FFNs, with their weights and state, live in layer shards (shard.dart)
/// run by a [LayerExecutor]. A prompt goes through the layers in chunks of
/// up to [maxChunk] tokens; decoding is a chunk of one.
///
/// On the GPU ([load]'s `gpu`: Metal or Vulkan) the whole forward pass of a
/// chunk runs on the GPU instead (gpu.dart), and this class only looks
/// up embeddings and keeps the position.
///
/// The model remembers the tokens it has seen ([history]) and can save its
/// state at positions ([checkpoint]). [prefill] uses both: a prompt that
/// extends what the model has seen costs only its new tokens, and one that
/// diverges restarts from the last checkpoint before the divergence.
library;

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'config.dart';
import 'executor.dart';
import 'gpu.dart';
import 'native.dart';
import 'ops.dart';
import 'q8_cache.dart';
import 'quant.dart';
import 'safetensors.dart';
import 'shard.dart';

final class _LayerNorms {
  _LayerNorms(this.operator, this.postOperator, this.ffn, this.postFfn);

  final Float32List operator, ffn;
  final Float32List? postOperator, postFfn;
}

final class Lfm3Model {
  Lfm3Model._(
    this.config,
    this._embedBf16,
    this._embedQ8,
    this._norms,
    this._finalNorm,
    this._executor,
    this._gpu,
    this._stateKey,
    this.maxSeq,
    this.maxChunk,
  );

  /// Loads the model.
  ///
  /// With [workers] = 0 a single layer shard runs in this isolate. With
  /// [workers] > 0 the layers are split into that many shards, one per
  /// worker isolate (tensor parallel; see shard.dart). [weights] quantizes
  /// every matrix, the tied LM head included, at load time, except those
  /// whose tensor name contains an entry of [keepF32]. For Q8 and Q4
  /// ([WeightFormat.isNative]), [q8CachePath]
  /// names a cache file: mapped if it matches this checkpoint, written
  /// after quantizing otherwise.
  ///
  /// With the native kernels loaded, the bf16 embedding table is mapped
  /// from model.safetensors instead of read, so it costs no heap.
  ///
  /// [gpu] runs every token on the GPU from the cache (built first if
  /// missing), so it needs Q8 or Q4 weights, a cache path and the native
  /// kernels, which map the cache. [gpuLib] does too, with the GPU layer of
  /// the library at that path instead of the package's kernels asset.
  static Future<Lfm3Model> load(
    String modelDir, {
    int maxSeq = 4096,
    int maxChunk = 64,
    int workers = 0,
    WeightFormat weights = WeightFormat.f32,
    List<String> keepF32 = const [],
    String? q8CachePath,
    bool gpu = false,
    String? gpuLib,
  }) async {
    final onGpu = gpu || gpuLib != null;
    if (onGpu && (!weights.isNative || q8CachePath == null || workers > 0)) {
      throw ArgumentError(
        'the GPU needs --weights=q8 or q4, a cache and --workers=0',
      );
    }
    if (weights.isNative && workers > 0) {
      // Native kernels already split each product across native threads;
      // isolates on top would oversubscribe the cores.
      throw ArgumentError(
        '--weights=${weights.name} runs in one isolate; use --workers=0',
      );
    }
    final c = Lfm3Config.load(modelDir);
    final h = c.hiddenSize;
    final st = SafetensorsFile.open('$modelDir/model.safetensors');
    final List<_LayerNorms> norms;
    final Float32List finalNorm;
    // Token lookup reads the bf16 patterns, which widen to exactly the
    // checkpoint's floats at half the memory of a float32 table.
    final Uint16List? embedBf16;
    try {
      norms = [
        for (var i = 0; i < c.numLayers; i++)
          _LayerNorms(
            st.readF32('model.layers.$i.operator_norm.weight', [h]),
            c.sandwichNorm
                ? st.readF32('model.layers.$i.post_operator_norm.weight', [h])
                : null,
            st.readF32('model.layers.$i.ffn_norm.weight', [h]),
            c.sandwichNorm
                ? st.readF32('model.layers.$i.post_ffn_norm.weight', [h])
                : null,
          ),
      ];
      finalNorm = st.readF32('model.embedding_norm.weight', [h]);
      // A bundle imported from GGUF (`lfm3 import-gguf`) has no bf16
      // embedding: tokens are then looked up in the Q8 LM head below.
      embedBf16 = st.tensors.containsKey(_embedName)
          ? _mapEmbedding(modelDir, st, c) ??
                st.readBf16Bits(_embedName, [c.vocabSize, h])
          : null;
    } finally {
      st.close();
    }
    String? cacheKey;
    Q8Cache? cache;
    Map<String, NativeMatrix>? built;
    if (weights.isNative && q8CachePath != null) {
      cacheKey = Q8Cache.keyFor(modelDir, weights: weights.name);
      cache = Q8Cache.open(q8CachePath, cacheKey);
      if (cache == null) built = {};
    }
    // What a saved state depends on: the checkpoint, the weights, and the
    // backend, whose arithmetic differs in the last bits.
    final native = NativeKernels.isLoaded ? NativeKernels.instance : null;
    final backend = onGpu
        ? 'gpu ${Platform.operatingSystem}'
        : 'cpu inputs=${weights.isNative && native!.int8Inputs ? 'int8' : 'f32'}'
              ' attention=${native?.attention == true ? 'c' : 'dart'}';
    final stateKey =
        '${Q8Cache.keyFor(modelDir, weights: weights.name)} '
        '${weights.name} keep=${keepF32.join(',')} $backend';
    NativeQ8Matrix? embedQ8;
    if (embedBf16 == null) {
      embedQ8 = switch (cache?.matrix(_embedName, c.vocabSize, h)) {
        final NativeQ8Matrix m => m,
        _ => throw StateError(
          'model.safetensors lacks the embedding and no Q8 cache holds it',
        ),
      };
    }
    if (onGpu) {
      if (cache == null) {
        // Quantize through the CPU loader once to build the cache.
        LayerShard.load(
          modelDir,
          index: 0,
          shards: 1,
          maxSeq: 1,
          maxChunk: 1,
          format: weights,
          keepF32: keepF32,
          q8Built: built,
        );
        Q8Cache.write(q8CachePath!, cacheKey!, built!);
        cache = Q8Cache.open(q8CachePath, cacheKey)!;
      }
      final gpu = GpuForward.create(
        libPath: gpuLib,
        modelDir: modelDir,
        config: c,
        cache: cache,
        maxSeq: maxSeq,
        maxChunk: maxChunk,
      );
      return Lfm3Model._(
        c,
        embedBf16,
        embedQ8,
        norms,
        finalNorm,
        null,
        gpu,
        stateKey,
        maxSeq,
        maxChunk,
      );
    }
    final LayerExecutor executor = workers == 0
        ? LocalExecutor(
            LayerShard.load(
              modelDir,
              index: 0,
              shards: 1,
              maxSeq: maxSeq,
              maxChunk: maxChunk,
              format: weights,
              keepF32: keepF32,
              q8Cache: cache,
              q8Built: built,
            ),
          )
        : await PooledExecutor.spawn(
            modelDir,
            workers: workers,
            maxSeq: maxSeq,
            maxChunk: maxChunk,
            format: weights,
            keepF32: keepF32,
          );
    if (built != null) Q8Cache.write(q8CachePath!, cacheKey!, built);
    return Lfm3Model._(
      c,
      embedBf16,
      embedQ8,
      norms,
      finalNorm,
      executor,
      null,
      stateKey,
      maxSeq,
      maxChunk,
    );
  }

  /// A view of the bf16 embedding inside a mapping of model.safetensors,
  /// or null without the native kernels.
  static Uint16List? _mapEmbedding(
    String modelDir,
    SafetensorsFile st,
    Lfm3Config c,
  ) {
    if (!NativeKernels.isLoaded) return null;
    final info = st.tensors['model.embed_tokens.weight'];
    if (info == null || info.dtype != 'BF16') return null;
    final offset = st.dataStart + info.begin;
    if (offset.isOdd || info.numel != c.vocabSize * c.hiddenSize) return null;
    final file = MappedFile.open('$modelDir/model.safetensors');
    if (file == null) return null;
    return (file.address + offset).cast<Uint16>().asTypedList(info.numel);
  }

  final Lfm3Config config;
  final int maxSeq;

  /// Largest number of tokens [forward] pushes through the layers at once.
  final int maxChunk;

  /// [V, H] embedding as bf16 bit patterns, for token lookup only.
  final Uint16List? _embedBf16;

  /// The tied embedding as the Q8 LM head, when there is no bf16 copy.
  final NativeQ8Matrix? _embedQ8;

  static const _embedName = 'model.embed_tokens.weight';
  final List<_LayerNorms> _norms;
  final Float32List _finalNorm;
  final LayerExecutor? _executor;

  /// The GPU forward pass, when loaded with a GPU library.
  final GpuForward? _gpu;

  /// Identifies the states [saveState] writes for this model, weights and
  /// backend; [loadState] refuses others.
  final String _stateKey;
  GpuForward? get gpu => _gpu;

  /// Number of tokens processed so far, i.e. the position of the next one.
  int length = 0;

  /// The tokens processed so far, by position.
  List<int> get history => List.unmodifiable(_history);
  final _history = <int>[];

  /// Saved states by position (see [checkpoint]).
  final _checkpoints = <int, Object>{};

  /// How many checkpoints [checkpoint] keeps: the earliest (usually the
  /// end of the system prompt) and the latest others.
  static const maxCheckpoints = 16;

  // The residual stream and normalized inputs of one chunk: [maxChunk, H].
  late final Float32List _x = Float32List(maxChunk * config.hiddenSize);
  late final Float32List _a = Float32List(maxChunk * config.hiddenSize);
  late final Float32List _m = Float32List(maxChunk * config.hiddenSize);
  late final Float32List _normed = Float32List(config.hiddenSize);
  late final Float32List logits = _gpu?.logits ?? Float32List(config.vocabSize);

  /// Stops the worker isolates, if any.
  Future<void> close() async => _executor?.close();

  /// Timing of the shard calls, for profiling (empty on the GPU).
  late final ExecutorStats executorStats = _executor?.stats ?? ExecutorStats();

  /// Clears all state so the next call starts a new sequence.
  Future<void> reset() async {
    length = 0;
    _history.clear();
    _checkpoints.clear();
    _gpu?.reset();
    await _executor?.reset();
  }

  /// Feeds the prompt [ids] for a new reply, reusing what the model has
  /// already computed: the longest common prefix of [ids] and [history]
  /// if [history] does not go past it, else the state at the latest
  /// checkpoint inside it, else nothing. At least the last token is always
  /// fed, so [logits] are the prompt's. Saves checkpoints at [checkpoints]
  /// (positions in [ids], such as the end of the system prompt) and at the
  /// end of the prompt. Returns how many tokens were reused.
  ///
  /// Each token's arithmetic does not depend on how tokens are chunked, so
  /// the logits are the same as feeding [ids] from scratch.
  Future<int> prefill(
    List<int> ids, {
    Iterable<int> checkpoints = const [],
  }) async {
    if (ids.isEmpty) throw ArgumentError('empty prompt');
    var common = 0;
    final limit = math.min(_history.length, ids.length - 1);
    while (common < limit && _history[common] == ids[common]) {
      common++;
    }
    if (common < length) {
      final start = _checkpoints.keys
          .where((p) => p <= common)
          .fold(0, math.max);
      await _rewind(start);
      common = start;
    }
    final stops = {
      for (final p in checkpoints)
        if (p > common && p < ids.length) p,
      ids.length,
    }.toList()..sort();
    var at = common;
    for (final stop in stops) {
      await forward(ids.sublist(at, stop), computeLogits: stop == ids.length);
      await checkpoint();
      at = stop;
    }
    return common;
  }

  /// Saves the state at the current position, for [prefill] to return to.
  /// Costs a copy of the conv histories (640 KB for LFM3-1.7B); a no-op
  /// with worker isolates.
  Future<void> checkpoint() async {
    if (length == 0 || _checkpoints.containsKey(length)) return;
    final state = _gpu?.saveState() ?? await _executor!.saveState();
    if (state == null) return;
    _checkpoints[length] = state;
    if (_checkpoints.length > maxCheckpoints) {
      final positions = _checkpoints.keys.toList()..sort();
      _checkpoints.remove(positions[1]);
    }
  }

  /// Writes the state at the current position to [path]: the tokens, the
  /// KV rows before it and the conv histories, about 45 KB per token for
  /// LFM3-1.7B (34 MB for a 761-token system prompt with tools). A later
  /// process on the same model, weights and backend can [loadState] it
  /// instead of computing those tokens. Written through a temporary file
  /// and a rename, so a reader never sees half a state.
  ///
  /// Layout: magic `LFM3ST01`, a little-endian u64 header length, a JSON
  /// header (version, key, tokens, block lengths), zero padding to a
  /// multiple of 64, then the float blocks of [_stateBlocks] in order.
  Future<void> saveState(String path) async {
    final blocks =
        _stateBlocks(length) ??
        (throw UnsupportedError('worker isolates do not save states'));
    final header = utf8.encode(
      jsonEncode({
        'version': _stateVersion,
        'key': _stateKey,
        'tokens': _history,
        'blocks': [for (final b in blocks) b.length],
      }),
    );
    final temp = File('$path.partial');
    final out = temp.openSync(mode: FileMode.write);
    try {
      final prefix = ByteData(16);
      for (var i = 0; i < 8; i++) {
        prefix.setUint8(i, _stateMagic.codeUnitAt(i));
      }
      prefix.setUint64(8, header.length, Endian.little);
      out
        ..writeFromSync(prefix.buffer.asUint8List())
        ..writeFromSync(header)
        ..writeFromSync(Uint8List(_padTo64(16 + header.length)));
      for (final b in blocks) {
        out.writeFromSync(
          b.buffer.asUint8List(b.offsetInBytes, b.lengthInBytes),
        );
      }
    } finally {
      out.closeSync();
    }
    temp.renameSync(path);
  }

  /// Restores a state written by [saveState] for this model, weights and
  /// backend, and checkpoints it. Returns false, with the model reset, if
  /// [path] is missing, malformed or written for anything else.
  Future<bool> loadState(String path) async {
    await reset();
    final file = File(path);
    if (!file.existsSync()) return false;
    final input = file.openSync();
    try {
      final prefix = input.readSync(16);
      if (prefix.length < 16 ||
          String.fromCharCodes(prefix.sublist(0, 8)) != _stateMagic) {
        return false;
      }
      final headerLength = ByteData.sublistView(
        prefix,
      ).getUint64(8, Endian.little);
      final header =
          jsonDecode(utf8.decode(input.readSync(headerLength)))
              as Map<String, Object?>;
      final tokens = (header['tokens'] as List).cast<int>();
      if (header['version'] != _stateVersion ||
          header['key'] != _stateKey ||
          tokens.isEmpty ||
          tokens.length > maxSeq) {
        return false;
      }
      final blocks = _stateBlocks(tokens.length);
      final lengths = (header['blocks'] as List).cast<int>();
      if (blocks == null ||
          blocks.length != lengths.length ||
          Iterable.generate(
            blocks.length,
          ).any((i) => blocks[i].length != lengths[i])) {
        return false;
      }
      input.setPositionSync(16 + headerLength + _padTo64(16 + headerLength));
      for (final b in blocks) {
        final bytes = b.buffer.asUint8List(b.offsetInBytes, b.lengthInBytes);
        var done = 0;
        while (done < bytes.length) {
          final n = input.readIntoSync(bytes, done);
          if (n <= 0) {
            await reset();
            return false;
          }
          done += n;
        }
      }
      length = tokens.length;
      _history.addAll(tokens);
      await checkpoint();
      return true;
    } on FormatException {
      await reset();
      return false;
    } finally {
      input.closeSync();
    }
  }

  static const _stateMagic = 'LFM3ST01';
  static const _stateVersion = 1;

  static int _padTo64(int offset) => (64 - offset % 64) % 64;

  List<Float32List>? _stateBlocks(int position) =>
      _gpu?.stateBlocks(position) ?? _executor!.stateBlocks(position);

  /// Returns to the state at checkpoint [position] (0: the empty state).
  /// KV cache rows past it are overwritten before they are read again.
  Future<void> _rewind(int position) async {
    if (position == 0) {
      await reset();
      return;
    }
    final state = _checkpoints[position]!;
    if (_gpu case final gpu?) {
      gpu.restoreState(state as List<Float32List>);
    } else {
      await _executor!.restoreState(state);
    }
    length = position;
    _history.length = position;
    _checkpoints.removeWhere((p, _) => p > position);
  }

  /// Feeds one token at position [length] and advances.
  Future<void> step(int token, {bool computeLogits = true}) =>
      forward([token], computeLogits: computeLogits);

  /// Feeds [tokens] at positions [length], [length] + 1, ... and advances.
  ///
  /// [capture] receives, per chunk, the residual stream entering layer 0
  /// (slot 0) and leaving each layer (slot i + 1), as `count` rows of hidden
  /// size starting at position `first`. With [computeLogits] the last
  /// token's logits are left in [logits].
  Future<void> forward(
    List<int> tokens, {
    void Function(int slot, int first, Float32List rows, int count)? capture,
    bool computeLogits = true,
  }) async {
    if (length + tokens.length > maxSeq) {
      throw StateError('sequence exceeds maxSeq=$maxSeq');
    }
    final gpu = _gpu;
    if (gpu != null) {
      final h = config.hiddenSize;
      for (var start = 0; start < tokens.length; start += maxChunk) {
        final n = math.min(maxChunk, tokens.length - start);
        for (var t = 0; t < n; t++) {
          _embed(tokens[start + t], gpu.residual, t * h);
        }
        gpu.forward(
          n,
          length,
          withLogits: computeLogits && start + n == tokens.length,
          capture: capture != null,
        );
        if (capture != null) {
          for (var slot = 0; slot <= config.numLayers; slot++) {
            capture(slot, length, gpu.captured(slot, n), n);
          }
        }
        length += n;
        _history.addAll(tokens.sublist(start, start + n));
      }
      return;
    }
    for (var start = 0; start < tokens.length; start += maxChunk) {
      final end = math.min(start + maxChunk, tokens.length);
      await _forwardChunk(tokens.sublist(start, end), capture);
    }
    if (computeLogits && tokens.isNotEmpty) {
      final h = config.hiddenSize;
      final last = (tokens.length - 1) % maxChunk;
      await logitsOf(Float32List.sublistView(_x, last * h, (last + 1) * h));
    }
  }

  /// Applies the final norm and the tied LM head to one residual vector,
  /// leaving the result in [logits].
  Future<Float32List> logitsOf(Float32List residual) async {
    if (_gpu != null) return _gpu.logitsOf(residual);
    rmsNorm(
      residual,
      _finalNorm,
      _normed,
      0,
      0,
      config.hiddenSize,
      config.normEps,
    );
    await _executor!.lmHead(_normed, logits);
    return logits;
  }

  Future<void> _forwardChunk(
    List<int> tokens,
    void Function(int slot, int first, Float32List rows, int count)? capture,
  ) async {
    final n = tokens.length;
    final h = config.hiddenSize;
    final executor = _executor!;
    for (var t = 0; t < n; t++) {
      _embed(tokens[t], _x, t * h);
    }
    capture?.call(0, length, _x, n);

    for (var i = 0; i < _norms.length; i++) {
      final norms = _norms[i];
      _normRows(_x, norms.operator, _a, n);
      _m.setAll(0, await executor.mixer(i, _a, n, length));
      if (norms.postOperator != null) _normRows(_m, norms.postOperator!, _m, n);
      _addInto(_x, _m, n * h);

      _normRows(_x, norms.ffn, _a, n);
      _m.setAll(0, await executor.ffn(i, _a, n));
      if (norms.postFfn != null) _normRows(_m, norms.postFfn!, _m, n);
      _addInto(_x, _m, n * h);
      capture?.call(i + 1, length, _x, n);
    }
    length += n;
    _history.addAll(tokens);
  }

  /// Writes token [token]'s embedding row into [out] at [offset].
  void _embed(int token, Float32List out, int offset) {
    final bf16 = _embedBf16;
    if (bf16 == null) {
      _embedQ8!.dequantizeRow(token, out, offset);
      return;
    }
    final h = config.hiddenSize;
    final bits = Uint32List.sublistView(out);
    final src = token * h;
    for (var j = 0; j < h; j++) {
      bits[offset + j] = bf16[src + j] << 16;
    }
  }

  void _normRows(Float32List x, Float32List weight, Float32List out, int n) {
    final h = config.hiddenSize;
    for (var t = 0; t < n; t++) {
      rmsNorm(x, weight, out, t * h, t * h, h, config.normEps);
    }
  }

  static void _addInto(Float32List x, Float32List m, int count) {
    for (var j = 0; j < count; j++) {
      x[j] = x[j] + m[j];
    }
  }
}
