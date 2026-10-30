/// Runs the layer shards: one in this isolate, or one per worker isolate.
///
/// With workers, every mixer and every FFN is one round trip: the main
/// isolate sends the normalized input to all shards and adds their partial
/// outputs in shard order, so the sum is deterministic. A decoded token
/// takes 2 round trips per layer plus one for the LM head (49 for LFM3),
/// where splitting each matrix product separately took 97.
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'config.dart';
import 'quant.dart';
import 'shard.dart';

/// Where the time of the executor's calls went, for profiling.
final class ExecutorStats {
  int calls = 0;

  /// Main isolate: from sending a request to having every reply.
  int wallMicros = 0;

  /// The slowest shard's compute time per call, summed.
  int slowestComputeMicros = 0;

  /// All shards' compute time, summed (divide by shards for the mean).
  int totalComputeMicros = 0;

  /// Main isolate: adding the partial outputs.
  int sumMicros = 0;

  /// Request delivery (main send to worker start) and reply delivery
  /// (worker send to main handler), summed over all shards, and the worst.
  int requestLatencyMicros = 0, replyLatencyMicros = 0;
  int worstRequestLatency = 0, worstReplyLatency = 0;

  void clear() {
    calls = wallMicros = slowestComputeMicros = totalComputeMicros = sumMicros =
        0;
    requestLatencyMicros = replyLatencyMicros = 0;
    worstRequestLatency = worstReplyLatency = 0;
  }
}

abstract interface class LayerExecutor {
  /// Timing of mixer, FFN and LM-head calls since the last clear.
  ExecutorStats get stats;

  /// Layer [layer]'s mixer for [n] normalized inputs [a] ([n, H]) at
  /// positions [position], [position] + 1, ...; returns [n, H].
  Future<Float32List> mixer(int layer, Float32List a, int n, int position);

  /// Layer [layer]'s FFN for [n] normalized inputs; returns [n, H].
  Future<Float32List> ffn(int layer, Float32List a, int n);

  /// The LM head for one normalized vector; writes [logits].
  Future<void> lmHead(Float32List normed, Float32List logits);

  /// Starts a new sequence.
  Future<void> reset();

  /// The recurrent state at the current position (see
  /// [LayerShard.saveState]), or null if this executor cannot save it.
  Future<Object?> saveState();

  /// Returns to a state from [saveState].
  Future<void> restoreState(Object state);

  /// Views of the state the first [position] tokens left behind (see
  /// [LayerShard.stateBlocks]), or null if it lives elsewhere.
  List<Float32List>? stateBlocks(int position);

  Future<void> close();
}

/// One shard owning everything, in this isolate.
final class LocalExecutor implements LayerExecutor {
  LocalExecutor(this._shard);

  final LayerShard _shard;

  @override
  final ExecutorStats stats = ExecutorStats();

  @override
  Future<Float32List> mixer(int layer, Float32List a, int n, int position) =>
      Future.value(_shard.mixer(layer, a, n, position));

  @override
  Future<Float32List> ffn(int layer, Float32List a, int n) =>
      Future.value(_shard.ffn(layer, a, n));

  @override
  Future<void> lmHead(Float32List normed, Float32List logits) {
    logits.setAll(0, _shard.lmHead(normed));
    return Future.value();
  }

  @override
  Future<void> reset() {
    _shard.reset();
    return Future.value();
  }

  @override
  Future<Object?> saveState() => Future.value(_shard.saveState());

  @override
  Future<void> restoreState(Object state) {
    _shard.restoreState(state as List<Float32List>);
    return Future.value();
  }

  @override
  List<Float32List> stateBlocks(int position) => _shard.stateBlocks(position);

  @override
  Future<void> close() => Future.value();
}

enum _Op { mixer, ffn, lmHead, reset }

typedef _WorkerArgs = (
  SendPort main,
  String modelDir,
  int index,
  int shards,
  int maxSeq,
  int maxChunk,
  WeightFormat format,
  List<String> keepF32,
);

/// One shard per worker isolate.
final class PooledExecutor implements LayerExecutor {
  PooledExecutor._(this._config, this._shards, this._replies, this._maxChunk);

  static Future<PooledExecutor> spawn(
    String modelDir, {
    required int workers,
    required int maxSeq,
    required int maxChunk,
    WeightFormat format = WeightFormat.f32,
    List<String> keepF32 = const [],
  }) async {
    final replies = ReceivePort();
    final pool = PooledExecutor._(
      Lfm3Config.load(modelDir),
      workers,
      replies,
      maxChunk,
    );
    final ready = Completer<void>();
    var readyCount = 0;
    replies.listen((message) {
      switch (message) {
        case (int worker, SendPort port, int vocabStart):
          pool._ports[worker] = port;
          pool._vocabStarts[worker] = vocabStart;
          if (++readyCount == workers) ready.complete();
        case (
          int worker,
          int seq,
          Float32List? result,
          int micros,
          int requestLatency,
          int sentAt,
        ):
          final replyLatency = DateTime.now().microsecondsSinceEpoch - sentAt;
          final st = pool.stats;
          st
            ..requestLatencyMicros += requestLatency
            ..replyLatencyMicros += replyLatency;
          if (requestLatency > st.worstRequestLatency) {
            st.worstRequestLatency = requestLatency;
          }
          if (replyLatency > st.worstReplyLatency) {
            st.worstReplyLatency = replyLatency;
          }
          pool._receive(worker, seq, result, micros);
        case (int worker, String error):
          final failure = StateError('worker $worker: $error');
          if (!ready.isCompleted) ready.completeError(failure);
          for (final pending in pool._pending.values) {
            pending.completer.completeError(failure);
          }
          pool._pending.clear();
      }
    });
    for (var w = 0; w < workers; w++) {
      await Isolate.spawn(_workerMain, (
        replies.sendPort,
        modelDir,
        w,
        workers,
        maxSeq,
        maxChunk,
        format,
        keepF32,
      ));
    }
    await ready.future;
    return pool;
  }

  final Lfm3Config _config;
  final int _shards;
  final ReceivePort _replies;
  final int _maxChunk;
  late final List<SendPort?> _ports = List.filled(_shards, null);
  late final List<int> _vocabStarts = List.filled(_shards, 0);
  final Map<int, _Pending> _pending = {};
  int _seq = 0;
  final Stopwatch _clock = Stopwatch()..start();

  @override
  final ExecutorStats stats = ExecutorStats();
  late final Float32List _sum = Float32List(_maxChunk * _config.hiddenSize);

  Future<List<Float32List?>> _broadcast(
    _Op op,
    int layer,
    Float32List? input,
    int n,
    int position,
  ) {
    final seq = _seq++;
    final pending = _Pending(_shards, _clock.elapsedMicroseconds);
    _pending[seq] = pending;
    for (final port in _ports) {
      port!.send((
        seq,
        op.index,
        layer,
        input,
        n,
        position,
        DateTime.now().microsecondsSinceEpoch,
      ));
    }
    return pending.completer.future;
  }

  void _receive(int worker, int seq, Float32List? result, int micros) {
    final pending = _pending[seq]!;
    pending.results[worker] = result;
    pending.slowest = micros > pending.slowest ? micros : pending.slowest;
    pending.total += micros;
    if (++pending.replies == _shards) {
      _pending.remove(seq);
      stats
        ..calls += 1
        ..wallMicros += _clock.elapsedMicroseconds - pending.started
        ..slowestComputeMicros += pending.slowest
        ..totalComputeMicros += pending.total;
      pending.completer.complete(pending.results);
    }
  }

  /// Adds the partials in shard order, so the result does not depend on
  /// which worker answers first.
  Float32List _sumPartials(List<Float32List?> partials, int count) {
    final started = _clock.elapsedMicroseconds;
    _sum.setRange(0, count, partials[0]!);
    for (var w = 1; w < partials.length; w++) {
      final p = partials[w]!;
      for (var j = 0; j < count; j++) {
        _sum[j] = _sum[j] + p[j];
      }
    }
    stats.sumMicros += _clock.elapsedMicroseconds - started;
    return Float32List.sublistView(_sum, 0, count);
  }

  @override
  Future<Float32List> mixer(
    int layer,
    Float32List a,
    int n,
    int position,
  ) async {
    final count = n * _config.hiddenSize;
    final partials = await _broadcast(
      _Op.mixer,
      layer,
      Float32List.sublistView(a, 0, count),
      n,
      position,
    );
    return _sumPartials(partials, count);
  }

  @override
  Future<Float32List> ffn(int layer, Float32List a, int n) async {
    final count = n * _config.hiddenSize;
    final partials = await _broadcast(
      _Op.ffn,
      layer,
      Float32List.sublistView(a, 0, count),
      n,
      0,
    );
    return _sumPartials(partials, count);
  }

  @override
  Future<void> lmHead(Float32List normed, Float32List logits) async {
    final slices = await _broadcast(_Op.lmHead, 0, normed, 1, 0);
    for (var w = 0; w < _shards; w++) {
      logits.setAll(_vocabStarts[w], slices[w]!);
    }
  }

  @override
  Future<void> reset() => _broadcast(_Op.reset, 0, null, 0, 0);

  // The state lives in the workers; reusing prefixes is not worth a
  // message protocol for this research path.
  @override
  Future<Object?> saveState() => Future.value();

  @override
  Future<void> restoreState(Object state) =>
      throw UnsupportedError('worker isolates do not restore states');

  @override
  List<Float32List>? stateBlocks(int position) => null;

  @override
  Future<void> close() async {
    for (final port in _ports) {
      port?.send(null);
    }
    _replies.close();
  }
}

final class _Pending {
  _Pending(int shards, this.started) : results = List.filled(shards, null);

  final List<Float32List?> results;
  final int started;
  final Completer<List<Float32List?>> completer = Completer();
  int replies = 0;
  int slowest = 0;
  int total = 0;
}

void _workerMain(_WorkerArgs args) {
  final (main, modelDir, index, shards, maxSeq, maxChunk, format, keepF32) =
      args;
  final LayerShard shard;
  try {
    shard = LayerShard.load(
      modelDir,
      index: index,
      shards: shards,
      maxSeq: maxSeq,
      maxChunk: maxChunk,
      format: format,
      keepF32: keepF32,
    );
  } on Object catch (e) {
    main.send((index, '$e'));
    return;
  }
  final requests = ReceivePort();
  final clock = Stopwatch()..start();
  main.send((index, requests.sendPort, shard.vocabStart));
  requests.listen((message) {
    if (message == null) {
      requests.close();
      return;
    }
    final (seq, op, layer, input, n, position, sentAt) =
        message as (int, int, int, Float32List?, int, int, int);
    final requestLatency = DateTime.now().microsecondsSinceEpoch - sentAt;
    try {
      final started = clock.elapsedMicroseconds;
      final Float32List? result = switch (_Op.values[op]) {
        _Op.mixer => shard.mixer(layer, input!, n, position),
        _Op.ffn => shard.ffn(layer, input!, n),
        _Op.lmHead => shard.lmHead(input!),
        _Op.reset => () {
          shard.reset();
          return null;
        }(),
      };
      main.send((
        index,
        seq,
        result,
        clock.elapsedMicroseconds - started,
        requestLatency,
        DateTime.now().microsecondsSinceEpoch,
      ));
    } on Object catch (e) {
      main.send((index, '$e'));
    }
  });
}
