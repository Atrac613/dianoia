/// Loading and closing the model over and over must give back what each
/// load takes: the device benchmark's `lifecycle_cycles` mode.
library;

import 'package:dianoia/lab.dart';

/// Loads the model in [modelDir] on the GPU, answers one short message and
/// closes it, [cycles] times after a first cycle that settles what the
/// process keeps for any model. Records, per cycle, the GPU buffers alive
/// with the model open and after closing it, and the process footprint.
Future<void> lifecycleCheck(
  Map<String, Object?> record,
  String modelDir,
  int cycles,
  void Function(String) log,
  void Function({bool finished}) save,
) async {
  final rows = <Map<String, Object?>>[];
  record['lifecycle'] = rows;
  for (var cycle = 0; cycle <= cycles; cycle++) {
    final sw = Stopwatch()..start();
    final model = await LocalModel.load(
      modelDir,
      backend: InferenceBackend.gpu,
    );
    final turn = await model.chat(maxNewTokens: 8).send({
      'role': 'user',
      'content': 'Say hi.',
    });
    final open = GpuForward.liveBuffers;
    await model.close();
    final row = {
      'cycle': cycle,
      'buffers_open': open,
      'buffers_closed': GpuForward.liveBuffers,
      'footprint_mb': physFootprint() / (1 << 20),
      'reply_tokens': turn.generatedTokens,
      'seconds': sw.elapsedMilliseconds / 1000,
    };
    rows.add(row);
    log(
      'cycle $cycle: $open GPU buffers open, ${row['buffers_closed']} '
      'after close, footprint ${(row['footprint_mb'] as double).round()} MB',
    );
    save();
  }

  final closed = await LocalModel.load(modelDir, backend: InferenceBackend.gpu);
  final chat = closed.chat(maxNewTokens: 4);
  await closed.close();
  try {
    await chat.send({'role': 'user', 'content': 'Hi.'});
    record['closed_model_refused'] = false;
  } on StateError {
    record['closed_model_refused'] = true;
  }
}
