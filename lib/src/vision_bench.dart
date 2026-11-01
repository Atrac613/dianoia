/// Public-API vision timing and process-memory evidence for desktop and phones.
library;

import 'dart:async';
import 'dart:io';

import 'package:crypto/crypto.dart';

import 'local_model.dart';
import 'native.dart';
import 'gpu.dart';
import 'vision_session.dart';

/// Measures verified bundle loading, image decoding, fresh image turns and a
/// text-only follow-up retaining visual features. Run in an inference isolate.
/// Each fresh turn uses a new conversation, so it re-encodes the image.
///
/// [onRecord] receives phase checkpoints, including before model loading, so
/// an interrupted device run leaves inspectable progress. Write/copy the record
/// synchronously if retaining it. Exceptions propagate after closing resources.
/// RSS high-water marks are process-lifetime values. Periodic footprint samples
/// can miss peaks within synchronous operations; they are not an OS peak meter.
Future<Map<String, Object?>> benchmarkVisionBundle({
  required String bundle,
  required String imagePath,
  required String prompt,
  InferenceBackend backend = InferenceBackend.gpu,
  int contextLength = 512,
  int threads = 4,
  int maxNewTokens = 32,
  int runs = 3,
  String? expectedText,
  void Function(Map<String, Object?> record)? onRecord,
  void Function(String message)? log,
}) async {
  if (runs <= 0 || contextLength <= 0 || threads <= 0 || maxNewTokens <= 0) {
    throw ArgumentError(
      'runs, context length, threads and token limit must be positive',
    );
  }
  final clock = Stopwatch()..start();
  final samples = <Map<String, Object?>>[];
  final turns = <Map<String, Object?>>[];
  final record = <String, Object?>{
    'version': 1,
    'started': DateTime.now().toUtc().toIso8601String(),
    'finished': false,
    'passed': false,
    'os': Platform.operatingSystem,
    'os_version': Platform.operatingSystemVersion,
    'dart_version': Platform.version,
    'cores': Platform.numberOfProcessors,
    'requested_backend': backend.name,
    'context_length': contextLength,
    'threads': threads,
    'max_new_tokens': maxNewTokens,
    'fresh_runs': runs,
    'prompt': prompt,
    'expected_text': expectedText,
    'load_includes_sha256_verification': true,
    'memory_sample_interval_ms': 100,
    'memory_scope':
        'process lifetime RSS high-water; sampled physical footprint; no forced GC',
    'memory': samples,
    'turns': turns,
  };
  var phase = 'starting';
  void sample() {
    samples.add({
      'elapsed_s': clock.elapsedMicroseconds / 1e6,
      'phase': phase,
      'rss_bytes': ProcessInfo.currentRss,
      'process_peak_rss_bytes': ProcessInfo.maxRss,
      'physical_footprint_bytes': physFootprint(),
      'gpu_live_buffers': GpuForward.liveBuffers,
      if (Platform.isIOS || Platform.isMacOS)
        'thermal_state': GpuForward.thermalState,
    });
  }

  void checkpoint(String next) {
    phase = next;
    record['phase'] = phase;
    sample();
    onRecord?.call(record);
    log?.call('vision: $phase (${clock.elapsedMilliseconds} ms)');
  }

  LocalVisionModel? model;
  Timer? timer;
  try {
    checkpoint('loading');
    final manifest = File('$bundle/vl_bundle.json').readAsBytesSync();
    record['bundle_manifest_sha256'] = sha256.convert(manifest).toString();
    timer = Timer.periodic(const Duration(milliseconds: 100), (_) => sample());
    final load = Stopwatch()..start();
    model = await LocalVisionModel.loadBundle(
      bundle,
      backend: backend,
      contextLength: contextLength,
      threads: threads,
    );
    record['load_s'] = load.elapsedMicroseconds / 1e6;
    record['backend'] = model.backend.name;
    checkpoint('loaded');
    final decode = Stopwatch()..start();
    final image = await ChatImage.fromFile(imagePath);
    record['image_decode_s'] = decode.elapsedMicroseconds / 1e6;
    record['image_width'] = image.width;
    record['image_height'] = image.height;
    record['image_sha256'] =
        (await sha256.bind(File(imagePath).openRead()).first).toString();
    checkpoint('image_decoded');

    Future<void> ask(
      VisionChatSession chat,
      String label, {
      bool attach = true,
    }) async {
      checkpoint(label);
      final elapsed = Stopwatch()..start();
      double? firstText;
      final turn = await chat.send(
        prompt,
        images: attach ? [image] : [],
        onText: (_) {
          if (firstText == null) {
            firstText = elapsed.elapsedMicroseconds / 1e6;
            sample();
          }
          return true;
        },
      );
      final wall = elapsed.elapsedMicroseconds / 1e6;
      turns.add({
        'case': label,
        'new_image': attach,
        'text': turn.text,
        'expected_text_matches': expectedText == null
            ? null
            : turn.text == expectedText,
        'prompt_tokens': turn.promptTokens,
        'reused_tokens': turn.reusedTokens,
        'generated_tokens': turn.generatedTokens,
        'prefill_s': turn.prefillSeconds,
        'first_text_s': firstText,
        'decode_s': turn.decodeSeconds,
        'decode_tokens_per_s': turn.decodeSeconds > 0
            ? turn.generatedTokens / turn.decodeSeconds
            : null,
        'wall_s': wall,
      });
      checkpoint('${label}_done');
    }

    VisionChatSession? last;
    for (var i = 0; i < runs; i++) {
      last?.clear();
      last = model.chat(maxNewTokens: maxNewTokens);
      await ask(last, 'fresh_${i + 1}');
    }
    await ask(last!, 'follow_up', attach: false);
    last.clear();
    record['fresh_answers_identical'] =
        turns.take(runs).map((t) => t['text']).toSet().length == 1;
    record['passed'] =
        record['fresh_answers_identical'] == true &&
        turns.every((t) => t['expected_text_matches'] != false);
  } catch (error) {
    record['error'] = '$error';
    rethrow;
  } finally {
    timer?.cancel();
    if (model != null) {
      try {
        checkpoint('closing');
      } finally {
        await model.close();
      }
    }
    record['finished'] = true;
    record['elapsed_s'] = clock.elapsedMicroseconds / 1e6;
    checkpoint(record.containsKey('error') ? 'failed' : 'done');
  }
  return record;
}
