/// Compare VL Q4/Q8 through the public API using existing validation images.
/// Build with dart build cli --target=tool/validate_vl_quantization.dart.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:dianoia/dianoia.dart';
import 'package:dianoia/lab.dart'
    show Lfm3Model, NativeKernels, WeightFormat, physFootprint, GpuForward;

Map<String, Object?> memory() => {
  'rss_bytes': ProcessInfo.currentRss,
  'process_peak_rss_bytes': ProcessInfo.maxRss,
  'physical_footprint_bytes': physFootprint(),
  'gpu_live_buffers': GpuForward.liveBuffers,
  'thermal_state': GpuForward.thermalState,
};

Future<void> main(List<String> args) async {
  final flags = {
    for (final a in args)
      a.substring(2, a.indexOf('=')): a.substring(a.indexOf('=') + 1),
  };
  String require(String key) =>
      flags[key] ?? (throw ArgumentError('--$key is required'));
  if (flags['prepare'] == 'q4') {
    NativeKernels.load(null, threads: 4);
    final backbone = require('backbone');
    final watch = Stopwatch()..start();
    final model = await Lfm3Model.load(
      backbone,
      maxSeq: 1,
      maxChunk: 1,
      weights: WeightFormat.q4,
      q8CachePath: '$backbone/lfm3-q4.cache',
    );
    await model.close();
    stdout.writeln(
      jsonEncode({
        'prepared_q4_s': watch.elapsedMicroseconds / 1e6,
        'cache_bytes': File('$backbone/lfm3-q4.cache').lengthSync(),
        ...memory(),
      }),
    );
    return;
  }
  final output = File(require('out'));
  if (output.existsSync()) {
    throw StateError('refusing to replace ${output.path}');
  }
  output.parent.createSync(recursive: true);
  final bundle = flags['bundle'];
  final source = flags['source'];
  if ((bundle == null) == (source == null)) {
    throw ArgumentError(
      'provide either --bundle or --source with --backbone and --audit',
    );
  }
  final images = require('images');
  final runs = int.parse(flags['runs'] ?? '3');
  final backend = InferenceBackend.values.byName(require('backend'));
  final cases = <Map<String, String>>[
    {
      'name': 'red',
      'image': '$images/red.png',
      'prompt': 'What is the main color? Answer with one color word.',
      'expected': 'Red',
    },
    {
      'name': 'blue',
      'image': '$images/blue.png',
      'prompt': 'What is the main color? Answer with one color word.',
      'expected': 'Blue',
    },
    {
      'name': 'split',
      'image': '$images/split.png',
      'prompt': 'What color is on the left? Answer with one color word.',
      'expected': 'Red',
    },
    {
      'name': 'photo-en',
      'image': '$images/extended/astronaut.png',
      'prompt': 'What is the person wearing? Answer in a short phrase.',
      'contains': 'spacesuit',
    },
    {
      'name': 'photo-ja',
      'image': '$images/extended/astronaut.png',
      'prompt': 'この人物は何を着ていますか。日本語で短く答えてください。',
      'contains': '宇宙服',
    },
    {
      'name': 'ocr-latin',
      'image': '$images/extended/ocr-latin.png',
      'prompt': 'Read the code after CODE. Return only the code.',
      'expected': 'AB42',
    },
    {
      'name': 'ocr-ja',
      'image': '$images/extended/ocr-ja.png',
      'prompt': '画像に書かれた文字を、そのまま書き起こしてください。',
      'expected': '東京 123',
    },
  ];
  final samples = <Map<String, Object?>>[];
  final results = <Map<String, Object?>>[];
  final watch = Stopwatch()..start();
  final record = <String, Object?>{
    'version': 1,
    'executable_sha256': sha256
        .convert(File(Platform.resolvedExecutable).readAsBytesSync())
        .toString(),
    'git_head': (Process.runSync('git', ['rev-parse', 'HEAD']).stdout as String)
        .trim(),
    'source_sha256': {
      for (final p in [
        'tool/validate_vl_quantization.dart',
        'lib/src/model.dart',
        'lib/src/shard.dart',
        'lib/src/native.dart',
        'lib/src/vision_session.dart',
        'lib/src/gpu.dart',
      ])
        p: sha256.convert(File(p).readAsBytesSync()).toString(),
    },
    'started': DateTime.now().toUtc().toIso8601String(),
    'os': Platform.operatingSystemVersion,
    'dart': Platform.version,
    'backend_requested': backend.name,
    'context_length': 512,
    'threads': 4,
    'max_new_tokens': 32,
    'fresh_runs': runs,
    'finished': false,
    'memory_scope':
        'fresh AOT process; RSS high-water and 100ms physical-footprint samples, which can miss synchronous peaks; no forced GC',
    'cases': results,
  };
  void save() => output.writeAsStringSync(
    '${const JsonEncoder.withIndent('  ').convert(record)}\n',
  );
  void sample(String phase) => samples.add({
    'phase': phase,
    'elapsed_s': watch.elapsedMicroseconds / 1e6,
    ...memory(),
  });
  sample('before_load');
  save();
  var phase = 'loading';
  final timer = Timer.periodic(
    const Duration(milliseconds: 100),
    (_) => sample(phase),
  );
  LocalVisionModel? model;
  try {
    if (bundle != null) {
      final manifestBytes = File('$bundle/vl_bundle.json').readAsBytesSync();
      record['manifest_sha256'] = sha256.convert(manifestBytes).toString();
      record['weights'] =
          (jsonDecode(utf8.decode(manifestBytes)) as Map)['weights'];
      record['layout'] = 'standalone-bundle';
      record['bundle_bytes'] = Directory(bundle)
          .listSync(recursive: true)
          .whereType<File>()
          .fold<int>(0, (n, f) => n + f.lengthSync());
      model = await LocalVisionModel.loadBundle(
        bundle,
        backend: backend,
        contextLength: 512,
        threads: 4,
      );
    } else {
      final auditFile = File(require('audit'));
      final auditBytes = auditFile.readAsBytesSync();
      final audit = jsonDecode(utf8.decode(auditBytes)) as Map;
      record['audit_sha256'] = sha256.convert(auditBytes).toString();
      record['weights'] = 'qad-q4';
      record['layout'] = 'official-GGUF-import';
      var bytes = 0;
      for (final entry in (audit['files'] as Map).entries) {
        final f = File(entry.key as String);
        final expected = entry.value as Map;
        if (f.lengthSync() != expected['bytes'] ||
            (await sha256.bind(f.openRead()).first).toString() !=
                expected['sha256']) {
          throw StateError('QAD imported file changed: ${f.path}');
        }
        bytes += f.lengthSync();
      }
      record['runtime_bytes'] = bytes;
      record['embedding_policy'] = audit['embedding_policy'];
      model = await LocalVisionModel.load(
        source!,
        backboneDir: require('backbone'),
        backend: backend,
        contextLength: 512,
        threads: 4,
      );
    }
    record['backend'] = model.backend.name;
    record['load_with_integrity_s'] = watch.elapsedMicroseconds / 1e6;
    sample('loaded');
    for (final c in cases) {
      phase = c['name']!;
      final imageFile = File(c['image']!);
      final image = await ChatImage.fromFile(imageFile.path);
      final turns = <Map<String, Object?>>[];
      final entry = <String, Object?>{
        ...c,
        'image_sha256': sha256.convert(imageFile.readAsBytesSync()).toString(),
        'turns': turns,
      };
      results.add(entry);
      VisionChatSession? chat;
      for (var i = 0; i <= runs; i++) {
        final fresh = i < runs;
        if (fresh) {
          chat?.clear();
          chat = model.chat(maxNewTokens: 32);
        }
        final turnWatch = Stopwatch()..start();
        double? firstText;
        final turn = await chat!.send(
          c['prompt']!,
          images: fresh ? [image] : [],
          onText: (_) {
            firstText ??= turnWatch.elapsedMicroseconds / 1e6;
            return true;
          },
        );
        final passed = c.containsKey('expected')
            ? turn.text == c['expected']
            : turn.text.toLowerCase().contains(c['contains']!.toLowerCase());
        turns.add({
          'fresh': fresh,
          'run': i + 1,
          'text': turn.text,
          'passed': passed,
          'prompt_tokens': turn.promptTokens,
          'generated_tokens': turn.generatedTokens,
          'prefill_s': turn.prefillSeconds,
          'first_text_s': firstText,
          'decode_s': turn.decodeSeconds,
          'tokens_per_s': turn.decodeSeconds > 0
              ? turn.generatedTokens / turn.decodeSeconds
              : null,
          'wall_s': turnWatch.elapsedMicroseconds / 1e6,
        });
        sample('${phase}_${i + 1}');
        save();
      }
      chat?.clear();
      stdout.writeln('$phase: ${turns.first['text']}');
    }
  } catch (e) {
    record['error'] = '$e';
    rethrow;
  } finally {
    timer.cancel();
    sample('before_close');
    if (model != null) await model.close();
    sample('closed');
    record['fresh_passed'] = results
        .expand((c) => c['turns'] as List<Map<String, Object?>>)
        .where((t) => t['fresh'] == true)
        .every((t) => t['passed'] == true);
    record['memory_samples'] = samples;
    record['elapsed_s'] = watch.elapsedMicroseconds / 1e6;
    record['finished'] = true;
    save();
  }
}
