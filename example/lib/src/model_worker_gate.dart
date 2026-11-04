import 'dart:async';

/// Serializes all model-worker lifetimes, including different example screens.
/// The native GPU backend is process-wide, so the next worker must wait for
/// the previous isolate's exit, not just its route's disposal.
final class ModelWorkerLease {
  ModelWorkerLease._(this.ready);
  static Future<void> _tail = Future<void>.value();
  final Future<void> ready;
  final _exited = Completer<void>();
  bool _released = false;

  static ModelWorkerLease reserve() {
    final lease = ModelWorkerLease._(_tail);
    _tail = lease._exited.future;
    return lease;
  }

  /// Safe before [ready]: a cancelled reservation still waits for its predecessor.
  void release() {
    if (_released) return;
    _released = true;
    unawaited(ready.then((_) => _exited.complete()));
  }
}
