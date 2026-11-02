import 'package:dianoia_example/src/model_worker_gate.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('workers wait for the previous isolate lifetime', () async {
    final a = ModelWorkerLease.reserve();
    final b = ModelWorkerLease.reserve();
    await a.ready;
    var granted = false;
    b.ready.then((_) => granted = true);
    await Future<void>.delayed(Duration.zero);
    expect(granted, isFalse);
    a.release();
    a.release();
    await b.ready;
    expect(granted, isTrue);
    b.release();
  });

  test(
    'cancelling a queued reservation cannot bypass an active worker',
    () async {
      final a = ModelWorkerLease.reserve();
      final skipped = ModelWorkerLease.reserve();
      final next = ModelWorkerLease.reserve();
      await a.ready;
      skipped.release();
      var granted = false;
      next.ready.then((_) => granted = true);
      await Future<void>.delayed(Duration.zero);
      expect(granted, isFalse);
      a.release();
      await next.ready;
      expect(granted, isTrue);
      next.release();
    },
  );
}
