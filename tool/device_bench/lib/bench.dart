/// The on-device benchmark, run in a worker isolate.
///
/// Expects the app's documents directory ([documentsPath]: Documents on
/// iOS, the app's external files directory on Android, where adb can push
/// and pull) to hold
///
/// * a model bundle: `LFM3-1.7B-device/` from `dianoia pack` (Q8) by default,
///   or the one `bench_config.json` names (`{"bundle": ..., "weights":
///   "q8" | "q4"}`, e.g. the QAD bundle from `dianoia import-gguf`). A file,
///   because environment variables given to `devicectl device process
///   launch` did not reach the app;
/// * optionally `ref/<case>/ids.i32` and `gen.i32` from the transformers
///   reference runs, for the accuracy dumps;
/// * optionally `tool_cases.json` (example/), for the time to first
///   token with a system prompt and a tool catalog (chat_bench.dart).
///
/// It measures the tool-calling prompt on the GPU (Metal on iOS, Vulkan on
/// Android) and on the CPU (native kernels) and writes `bench/<time>.json`.
/// For each reference
/// case it also writes a teacher-forced run in the layout of `lfm3 run`
/// (`out/<weights>/<backend>_<case>/`), so dart-llm-lab's
/// tools/lfm3_compare.py and tools/lfm3_quality.py on the Mac can check the
/// phone's numbers.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'gpu_sustained.dart';
import 'lifecycle_check.dart';

const _androidPackage = 'com.noguwo.dartllmlab.lfm3_bench';

const _prompt = "What's the weather like in Paris right now?";
const _tools = [
  {
    'type': 'function',
    'function': {
      'name': 'get_weather',
      'description': 'Get the current weather for a city.',
      'parameters': {
        'type': 'object',
        'properties': {
          'city': {'type': 'string', 'description': 'City name'},
        },
        'required': ['city'],
      },
    },
  },
];

/// Runs everything, reporting progress lines to [log], and returns the
/// record it also writes to Documents/bench/.
Future<Map<String, Object?>> runBench(void Function(String) log) async {
  final docs = documentsPath();
  final configFile = File('$docs/bench_config.json');
  final config = configFile.existsSync()
      ? (jsonDecode(configFile.readAsStringSync()) as Map)
            .cast<String, Object?>()
      : const <String, Object?>{};
  final bundle = config['bundle'] as String? ?? 'LFM3-1.7B-device';
  final weights = WeightFormat.values.byName(
    config['weights'] as String? ?? 'q8',
  );
  final modelDir = Platform.isAndroid
      ? '/data/data/$_androidPackage/files/$bundle'
      : '$docs/$bundle';
  final cachePath = '$modelDir/lfm3-${weights.name}.cache';
  final out = '$docs/out/${weights.name}';
  final started = DateTime.now();
  final record = <String, Object?>{
    'started': started.toIso8601String(),
    'os': Platform.operatingSystemVersion,
    'bundle': bundle,
    'weights': weights.name,
  };
  void save({bool finished = false}) {
    record['finished'] = finished;
    final dir = Directory('$docs/bench')..createSync(recursive: true);
    final json = const JsonEncoder.withIndent('  ').convert(record);
    final stamp = started.toIso8601String().replaceAll(':', '-');
    File('${dir.path}/$stamp.json').writeAsStringSync(json);
    File('${dir.path}/latest.json').writeAsStringSync(json);
  }

  save();
  if (Platform.isAndroid) {
    // Files pushed by adb land in external storage; map an internal copy.
    final seconds = _copyBundle('$docs/$bundle', modelDir);
    if (seconds != null) {
      log('Copied $bundle to internal storage in $seconds s');
    }
  }
  if (!File(cachePath).existsSync()) {
    log('Model bundle missing: copy $bundle into $docs.');
    record['error'] = 'model bundle missing';
    save(finished: true);
    return record;
  }

  NativeKernels.load(null, threads: 2);
  final tokenizer = Lfm3Tokenizer.load(modelDir);
  final ids = tokenizer.encode(
    renderChat(
      [
        {'role': 'user', 'content': _prompt},
      ],
      tools: [for (final t in _tools) t],
      enableThinking: false,
    ),
  );

  final sw = Stopwatch();
  final cycles = (config['lifecycle_cycles'] as num?)?.toInt() ?? 0;
  if (cycles > 0) {
    await lifecycleCheck(record, modelDir, cycles, log, save);
    record['seconds'] =
        DateTime.now().difference(started).inMilliseconds / 1000;
    save(finished: true);
    log('Done (lifecycle) in ${record['seconds']} s.');
    return record;
  }
  final toolCases = File('$docs/tool_cases.json');
  if (!Platform.isIOS) {
    record['device'] = _linuxDeviceFacts();
    log('${Platform.operatingSystemVersion}\n$bundle (${weights.name})');
  }
  try {
    await _gpuBench(
      record,
      modelDir,
      weights,
      cachePath,
      tokenizer,
      ids,
      toolCases,
      docs,
      out,
      log,
      save,
      vectorAcceptance: config['q8_vector_acceptance'] == true,
      vectorDefaultAcceptance: config['q8_vector_default_acceptance'] == true,
      sustainedSeconds: (config['gpu_sustained_seconds'] as num?)?.toInt() ?? 0,
    );
  } on StateError catch (e) {
    // No usable GPU: the CPU part still runs.
    log('GPU: $e');
    record['gpu_error'] = '$e';
  }

  if (config['skip_cpu'] == true) {
    record['seconds'] =
        DateTime.now().difference(started).inMilliseconds / 1000;
    save(finished: true);
    log('Done (GPU only) in ${record['seconds']} s.');
    return record;
  }

  // CPU: on iOS the two performance cores and all six; elsewhere 2, 4, the
  // cores faster than the slowest kind (phones mix core types) and every
  // core.
  final threadCounts = Platform.isIOS
      ? [2, 6]
      : ({
          2,
          4,
          _fastCores(record['device'] as Map<String, Object?>),
          Platform.numberOfProcessors,
        }.where((t) => t <= Platform.numberOfProcessors).toList()..sort());
  final cpuRecord = <String, Object?>{};
  record['cpu'] = cpuRecord;
  for (final threads in threadCounts) {
    NativeKernels.load(null, threads: threads);
    sw
      ..reset()
      ..start();
    final cpu = await Lfm3Model.load(
      modelDir,
      maxSeq: 2048,
      weights: weights,
      q8CachePath: cachePath,
    );
    final loadS = sw.elapsedMilliseconds / 1000;
    final runs = [
      for (var i = 1; i <= 2; i++)
        await _ask(cpu, tokenizer, ids, log, 'CPU x$threads ask $i'),
    ];
    cpuRecord['threads_$threads'] = {'load_s': loadS, 'ask': runs};
    await cpu.close();
    save();
  }

  // The forced dump, and the chat benchmark if the GPU did not run it, at
  // the thread count that decoded fastest.
  double decode(int t) => [
    for (final a
        in ((cpuRecord['threads_$t'] as Map)['ask'] as List)
            .cast<Map<String, Object?>>())
      a['decode_tok_per_s'] as double,
  ].reduce((a, b) => a > b ? a : b);
  final best = threadCounts.reduce((a, b) => decode(a) >= decode(b) ? a : b);
  cpuRecord['best_threads'] = best;
  NativeKernels.load(null, threads: best);
  final cpu = await Lfm3Model.load(
    modelDir,
    maxSeq: 2048,
    weights: weights,
    q8CachePath: cachePath,
  );
  if (File('$docs/ref/tools/ids.i32').existsSync()) {
    cpuRecord['forced_tools'] = await _forced(
      cpu,
      weights.name,
      '$docs/ref/tools',
      '$out/cpu_tools',
      'tools',
      log,
      'CPU x$best forced tools',
    );
    save();
  }
  if ((record['gpu'] as Map?)?['chat'] == null && toolCases.existsSync()) {
    cpuRecord['chat'] = await chatBench(
      cpu,
      tokenizer,
      jsonDecode(toolCases.readAsStringSync()) as Map<String, Object?>,
      (line) => log('CPU x$best chat: $line'),
      statePath: '$modelDir/chat.state',
    );
  }
  await cpu.close();
  record['seconds'] = DateTime.now().difference(started).inMilliseconds / 1000;
  save(finished: true);
  log('Done in ${record['seconds']} s; results in $docs/bench/.');
  return record;
}

/// The GPU part (Metal): three asks, the chat benchmark and forced dumps.
Future<void> _gpuBench(
  Map<String, Object?> record,
  String modelDir,
  WeightFormat weights,
  String cachePath,
  Lfm3Tokenizer tokenizer,
  List<int> ids,
  File toolCases,
  String docs,
  String out,
  void Function(String) log,
  void Function() save, {
  bool vectorAcceptance = false,
  bool vectorDefaultAcceptance = false,
  int sustainedSeconds = 0,
}) async {
  final bundle = record['bundle'];
  final sw = Stopwatch()..start();
  final gpu = await Lfm3Model.load(
    modelDir,
    maxSeq: 2048,
    weights: weights,
    q8CachePath: cachePath,
    gpu: true,
  );
  final gpuLoadS = sw.elapsedMilliseconds / 1000;
  record['device'] = {
    ...?record['device'] as Map<String, Object?>?,
    ...GpuForward.deviceFacts,
    'weight_segments': gpu.gpu!.weightSegments,
  };
  log(
    '${GpuForward.deviceName}, ${Platform.operatingSystemVersion}\n'
    '$bundle (${weights.name})\n'
    'GPU model loaded in ${gpuLoadS.toStringAsFixed(2)} s '
    '(${gpu.gpu!.weightSegments} weight buffer(s)), '
    'footprint ${_mb(physFootprint())} MB',
  );
  final gpuRecord = <String, Object?>{'load_s': gpuLoadS};
  record['gpu'] = gpuRecord;
  gpuRecord['q8_vector_head_default_eligible'] =
      gpu.gpu!.q8VectorHeadDefaultEligible;
  if (sustainedSeconds > 0) {
    if (!gpu.gpu!.q8VectorHeadDefaultEligible) {
      throw StateError(
        'sustained experiment requires the validated Q4 device/cache',
      );
    }
    await runGpuSustained(gpu, ids, sustainedSeconds, log, (result) {
      gpuRecord['sustained'] = result;
      save();
    });
    final controls = <String, Object?>{};
    gpuRecord['sustained_controls'] = controls;
    for (final c in ['tools', 'long']) {
      final ref = '$docs/ref/$c';
      GpuForward.decodeRows = 0;
      GpuForward.setQ8Vector(false);
      await _forced(
        gpu,
        weights.name,
        ref,
        '$out/reference_$c',
        c,
        log,
        'Reference GEMM $c',
      );
      GpuForward.decodeRows = 4;
      GpuForward.restoreQ8VectorDefault();
      gpuRecord['forced_$c'] = await _forced(
        gpu,
        weights.name,
        ref,
        '$out/gpu_$c',
        c,
        log,
        'Adopted GEMV $c',
      );
      controls[c] = {
        for (final name in [
          'hidden.f32',
          'logits.f32',
          'gen.i32',
          'gen_logits.f32',
        ])
          name: _sameFiles('$out/reference_$c/$name', '$out/gpu_$c/$name'),
      };
      save();
    }
    return;
  }
  if (vectorDefaultAcceptance) {
    if (!gpu.gpu!.q8VectorHeadDefaultEligible) {
      throw StateError(
        'device/cache is not eligible for scoped vector default',
      );
    }
    // Observe the factory's untouched default before any explicit override.
    GpuForward.profiling = true;
    GpuForward.takeProfile();
    await gpu.reset();
    await gpu.forward(ids);
    GpuForward.takeProfile();
    await gpu.step(_argmax(gpu.logits));
    final profile = GpuForward.takeProfile();
    GpuForward.profiling = false;
    final vectorCount = profile['gemv_q8_i8_vec']?.$2 ?? 0;
    final scalarCount = profile['gemv_q8_i8']?.$2 ?? 0;
    gpuRecord['q8_vector_default_probe'] = {
      'vector_dispatches': vectorCount,
      'scalar_dispatches': scalarCount,
    };
    if (vectorCount != 1 || scalarCount != 0) {
      throw StateError(
        'factory default did not select the registered vector head',
      );
    }
    log(
      'Scoped default probe: $vectorCount vector / $scalarCount scalar Q8 dispatches',
    );
  }
  if (vectorAcceptance &&
      !_selectVectorCandidate(true, vectorDefaultAcceptance)) {
    throw StateError('Q8 vector acceptance requires a supported Vulkan GPU');
  }
  gpuRecord['q8_vector_acceptance'] = vectorAcceptance;
  gpuRecord['q8_vector_default_acceptance'] = vectorDefaultAcceptance;
  final gpuAsks = <Map<String, Object?>>[];
  gpuRecord['ask'] = gpuAsks;
  for (var i = 1; i <= 3; i++) {
    gpuAsks.add(await _ask(gpu, tokenizer, ids, log, 'GPU ask $i'));
    save();
  }
  gpuRecord['profile'] = await _profileGpu(
    gpu,
    ids,
    log,
    vectorOnly: vectorAcceptance,
  );
  save();
  if (Platform.isAndroid) {
    // Pixel 8's driver exposes cur_freq; other devices or app SELinux
    // policies may deny it, in which case record null rather than guess.
    String? gpuClock() {
      try {
        return File(
          '/sys/devices/platform/1f000000.mali/cur_freq',
        ).readAsStringSync().trim();
      } on FileSystemException {
        return null;
      }
    }

    void matrixSample(Map<String, Object?> sample) {
      sample['gpu_cur_freq_raw_after'] = gpuClock();
      log(
        'Matrix sweep ${sample['matrix']} ${sample['format']}, '
        'round ${sample['round']}, rows ${sample['decode_rows']}, Q8 group ${sample['q8_workgroup']}, vector ${sample['q8_vector']}: '
        '${(sample['gpu_ms_per_dispatch'] as double).toStringAsFixed(3)} GPU ms/dispatch, '
        '${sample['differing_outputs']} differing outputs, '
        'clock ${sample['gpu_cur_freq_raw_after']}',
      );
    }

    if (!vectorAcceptance) {
      gpuRecord['matrix_sweep'] = gpu.gpu!.benchmarkDecodeMatrices(
        onSample: matrixSample,
      );
      gpuRecord['q8_workgroup_sweep'] = gpu.gpu!.benchmarkDecodeMatrices(
        q8WorkgroupSweep: true,
        onSample: matrixSample,
      );
    }
    gpuRecord['q8_vector_sweep'] = gpu.gpu!.benchmarkDecodeMatrices(
      q8VectorSweep: true,
      onSample: matrixSample,
    );
    save();
  }
  final acceptance = <String, Object?>{};
  if (vectorAcceptance) {
    gpuRecord['vector_acceptance'] = acceptance;
    if (toolCases.existsSync()) {
      acceptance['chat'] = await _validateVectorChat(
        gpu,
        tokenizer,
        jsonDecode(toolCases.readAsStringSync()) as Map<String, Object?>,
        '$modelDir/vector_acceptance',
        log,
        automaticCandidate: vectorDefaultAcceptance,
      );
      save();
    }
    if (!_selectVectorCandidate(true, vectorDefaultAcceptance)) {
      throw StateError('Q8 vector selection failed');
    }
  }
  if (toolCases.existsSync()) {
    gpuRecord['chat'] = await chatBench(
      gpu,
      tokenizer,
      jsonDecode(toolCases.readAsStringSync()) as Map<String, Object?>,
      (line) => log('GPU chat: $line'),
      statePath: '$modelDir/chat.state',
    );
    save();
  }
  for (final c in ['tools', 'long']) {
    final ref = '$docs/ref/$c';
    if (!File('$ref/ids.i32').existsSync()) continue;
    if (vectorAcceptance) {
      GpuForward.setQ8Vector(false);
      await _forced(
        gpu,
        weights.name,
        ref,
        '$out/scalar_$c',
        c,
        log,
        'Scalar forced $c',
      );
      if (!_selectVectorCandidate(true, vectorDefaultAcceptance)) {
        throw StateError('Q8 vector selection failed');
      }
    }
    gpuRecord['forced_$c'] = await _forced(
      gpu,
      weights.name,
      ref,
      '$out/gpu_$c',
      c,
      log,
      'GPU forced $c',
    );
    if (vectorAcceptance) {
      final files = <String, bool>{
        for (final name in [
          'hidden.f32',
          'logits.f32',
          'gen.i32',
          'gen_logits.f32',
        ])
          name: _sameFiles('$out/scalar_$c/$name', '$out/gpu_$c/$name'),
      };
      acceptance[c] = files;
      log('Vector acceptance $c: $files');
    }
    save();
  }
}

/// Checks every prompt/decode logit in each conversation. Observer work
/// affects timings, so this paired check reports correctness only.
bool _selectVectorCandidate(bool vector, bool automatic) => vector && automatic
    ? GpuForward.restoreQ8VectorDefault()
    : GpuForward.setQ8Vector(vector);

Future<Map<String, Object?>> _validateVectorChat(
  Lfm3Model model,
  Lfm3Tokenizer tokenizer,
  Map<String, Object?> spec,
  String stateDir,
  void Function(String) log, {
  bool automaticCandidate = false,
}) async {
  Directory(stateDir).createSync(recursive: true);
  final session = ChatSession(
    model,
    tokenizer,
    system: benchSystemPrompt,
    tools: catalogTools(spec),
    maxNew: 96,
  );
  final cases = (spec['cases'] as List).cast<Map<String, Object?>>();
  // Compute each mode's prefix independently, then reload that mode's
  // checkpoint per case so a scalar prefix cannot mask a vector defect.
  for (final vector in [false, true]) {
    if (!_selectVectorCandidate(vector, automaticCandidate)) {
      throw StateError('Q8 vector selection failed');
    }
    await model.reset();
    await session.warmUp();
    await model.saveState('$stateDir/$vector.state');
  }
  final results = <Map<String, Object?>>[];
  for (var i = 0; i < cases.length; i++) {
    final reference = <Float32List>[];
    final turns = <ChatTurn>[];
    var checked = 0, differing = 0, nonFinite = 0;
    for (final vector in [false, true]) {
      if (!_selectVectorCandidate(vector, automaticCandidate)) {
        throw StateError('Q8 vector selection failed');
      }
      await model.reset();
      session.clear();
      if (!await session.warmUp(statePath: '$stateDir/$vector.state')) {
        throw StateError('acceptance prefix checkpoint did not load');
      }
      turns.add(
        await session.send(
          {'role': 'user', 'content': cases[i]['prompt']},
          onLogits: (actual) {
            for (final value in actual) {
              if (!value.isFinite) nonFinite++;
            }
            if (!vector) {
              reference.add(Float32List.fromList(actual));
              return;
            }
            final step = checked++;
            if (step >= reference.length ||
                actual.length != reference[step].length) {
              differing++;
              return;
            }
            final expected = reference[step];
            final a = actual.buffer.asUint32List(
              actual.offsetInBytes,
              actual.length,
            );
            final b = expected.buffer.asUint32List(
              expected.offsetInBytes,
              expected.length,
            );
            for (var j = 0; j < a.length; j++) {
              if (a[j] != b[j]) differing++;
            }
          },
        ),
      );
    }
    final expect = cases[i]['expect'] as Map<String, Object?>?;
    final same =
        differing == 0 &&
        nonFinite == 0 &&
        checked == reference.length &&
        reference.length == turns[0].generatedTokens + 1 &&
        checked == turns[1].generatedTokens + 1 &&
        turns[0].text == turns[1].text;
    final result = <String, Object?>{
      'case': i + 1,
      'scalar_passed': toolCallMatches(turns[0].reply.toolCalls, expect),
      'vector_passed': toolCallMatches(turns[1].reply.toolCalls, expect),
      'identical': same,
      'scalar_logit_vectors': reference.length,
      'vector_logit_vectors': checked,
      'differing_logits': differing,
      'non_finite': nonFinite,
      'scalar_reply': turns[0].text,
      'vector_reply': turns[1].text,
    };
    results.add(result);
    log(
      'Vector chat acceptance ${i + 1}/${cases.length}: '
      'identical=$same, scalar=${result['scalar_passed']}, vector=${result['vector_passed']}, '
      '$checked logit vectors, $differing differing logits',
    );
  }
  return {
    'cases': results.length,
    'identical_cases': results.where((r) => r['identical'] == true).length,
    'scalar_passed': results.where((r) => r['scalar_passed'] == true).length,
    'vector_passed': results.where((r) => r['vector_passed'] == true).length,
    'runs': results,
  };
}

bool _sameFiles(String left, String right) {
  if (File(left).lengthSync() != File(right).lengthSync()) return false;
  final a = File(left).openSync(), b = File(right).openSync();
  final ab = Uint8List(65536), bb = Uint8List(65536);
  int readBlock(RandomAccessFile file, Uint8List buffer) {
    var n = 0;
    while (n < buffer.length) {
      final count = file.readIntoSync(buffer, n);
      if (count == 0) break;
      n += count;
    }
    return n;
  }

  try {
    while (true) {
      final n = readBlock(a, ab);
      if (readBlock(b, bb) != n) return false;
      if (n == 0) return true;
      for (var i = 0; i < n; i++) {
        if (ab[i] != bb[i]) return false;
      }
    }
  } finally {
    a.closeSync();
    b.closeSync();
  }
}

/// Copies the files of [from] to [to] unless [to] already has them all
/// with the same sizes. Returns the seconds it took, or null if nothing was
/// copied or [from] is missing.
String? _copyBundle(String from, String to) {
  final source = Directory(from);
  if (!source.existsSync()) return null;
  final files = source.listSync().whereType<File>().toList();
  bool same(File f) {
    final copy = File('$to/${f.uri.pathSegments.last}');
    return copy.existsSync() && copy.lengthSync() == f.lengthSync();
  }

  if (files.every(same)) return null;
  final sw = Stopwatch()..start();
  Directory(to).createSync(recursive: true);
  for (final f in files) {
    if (!same(f)) f.copySync('$to/${f.uri.pathSegments.last}');
  }
  return (sw.elapsedMilliseconds / 1000).toStringAsFixed(1);
}

/// Cores whose top frequency beats the slowest kind's: the big cores of a
/// phone (5 of the Pixel 8's 9), or every core if they are all alike.
int _fastCores(Map<String, Object?> facts) {
  final khz = (facts['max_khz'] as List).whereType<int>().toList();
  if (khz.isEmpty) return Platform.numberOfProcessors;
  final slowest = khz.reduce((a, b) => a < b ? a : b);
  final fast = khz.where((k) => k > slowest).length;
  return fast == 0 ? khz.length : fast;
}

/// What /proc says about the CPU on Linux and Android: core count, each
/// core's top frequency, and the CPU parts (big and little cores differ).
Map<String, Object?> _linuxDeviceFacts() {
  String? read(String path) {
    try {
      return File(path).readAsStringSync().trim();
    } on FileSystemException {
      return null;
    }
  }

  final cpuinfo = read('/proc/cpuinfo') ?? '';
  return {
    'cores': Platform.numberOfProcessors,
    'max_khz': [
      for (var i = 0; i < Platform.numberOfProcessors; i++)
        int.tryParse(
          read('/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq') ?? '',
        ),
    ],
    'cpu_parts': {
      for (final m in RegExp(r'CPU part\s*:\s*(\S+)').allMatches(cpuinfo))
        m[1]!,
    }.toList(),
    'mem_total_kb': int.tryParse(
      RegExp(
            r'MemTotal:\s*(\d+)',
          ).firstMatch(read('/proc/meminfo') ?? '')?[1] ??
          '',
    ),
  };
}

/// Where the bench reads its inputs and writes its results: Documents on
/// iOS (HOME is the app container; its tmp directory sits beside Documents
/// if HOME is unset), and on Android the app's external files directory,
/// which adb can write and read.
String documentsPath() => Platform.isAndroid
    ? '/storage/emulated/0/Android/data/$_androidPackage/files'
    : '${Platform.environment['HOME'] ?? Directory.systemTemp.parent.path}'
          '/Documents';

/// NSProcessInfo's thermal state on iOS (Metal layer); unknown elsewhere.
int? _thermal() => Platform.isIOS ? GpuForward.thermalState : null;

int _mb(int bytes) => bytes < 0 ? -1 : bytes >> 20;

int _argmax(Float32List v) {
  var best = 0;
  for (var i = 1; i < v.length; i++) {
    if (v[i] > v[best]) best = i;
  }
  return best;
}

/// GPU milliseconds per token by kernel kind (Vulkan times every dispatch;
/// Metal reports nothing): over the tool prompt in chunks, and over 32
/// decoded tokens.
Future<Map<String, Object?>> _profileGpu(
  Lfm3Model model,
  List<int> ids,
  void Function(String) log, {
  bool vectorOnly = false,
}) async {
  Map<String, Object?> perToken(Map<String, (double, int)> profile, int n) => {
    for (final MapEntry(key: kind, value: (ms, count)) in profile.entries)
      kind: {'ms_per_token': ms / n, 'dispatches_per_token': count / n},
  };
  void show(String label, Map<String, (double, int)> profile, int n) {
    if (profile.isEmpty) return;
    final total = profile.values.fold(0.0, (a, p) => a + p.$1) / n;
    final parts = profile.entries.toList()
      ..sort((a, b) => b.value.$1.compareTo(a.value.$1));
    log(
      'GPU profile, $label: ${total.toStringAsFixed(1)} ms/token: '
      '${parts.map((e) => '${e.key} ${(e.value.$1 / n).toStringAsFixed(2)}').join(', ')}',
    );
  }

  GpuForward.profiling = true;
  GpuForward.decodeRows = 0;
  await model.reset();
  GpuForward.takeProfile();
  await model.forward(ids);
  final prompt = GpuForward.takeProfile();
  final tokens = <int>[];
  final reference = <Float32List>[];
  var steps = 0;
  while (steps < 32) {
    final token = _argmax(model.logits);
    tokens.add(token);
    await model.step(token);
    reference.add(Float32List.fromList(model.logits));
    steps++;
  }
  final decode = GpuForward.takeProfile();
  show('prompt (${ids.length} tokens)', prompt, ids.length);
  show('decode reference (old int8 GEMM)', decode, steps);
  final record = <String, Object?>{
    'prompt': perToken(prompt, ids.length),
    'decode': perToken(decode, steps),
  };
  // Force the same tokens for every layout. Rotate the order across three
  // rounds to expose warm-up or clock drift instead of picking a lucky run.
  final sweep = <Map<String, Object?>>[];
  final q8Sweep = <Map<String, Object?>>[];
  final vectorSweep = <Map<String, Object?>>[];
  // Values 0/1/2/4 select rows at the default 128-invocation workgroup;
  // 64/128/256 select Q8 workgroups while retaining four rows per subgroup.
  // 512/513 compare scalar/vector loads at 128 invocations in ABBA order.
  final orders = vectorOnly
      ? [
          [512, 513, 513, 512],
          [512, 513, 513, 512],
          [512, 513, 513, 512],
        ]
      : [
          [0, 1, 2, 4, 64, 128, 256, 512, 513, 513, 512],
          [4, 2, 1, 0, 128, 256, 64, 512, 513, 513, 512],
          [2, 0, 4, 1, 256, 64, 128, 512, 513, 513, 512],
        ];
  for (var round = 0; round < orders.length; round++) {
    for (final variant in orders[round]) {
      final vectorExperiment = variant >= 512;
      final vector = variant == 513;
      final vectorSupported = GpuForward.setQ8Vector(vector);
      if (vectorExperiment && (!Platform.isAndroid || !vectorSupported)) {
        continue;
      }
      final workgroup = variant >= 64 && !vectorExperiment ? variant : 128;
      final supported = GpuForward.setQ8Workgroup(workgroup);
      if (variant >= 64 && (!Platform.isAndroid || !supported)) continue;
      final rows = variant >= 64 ? 4 : variant;
      GpuForward.decodeRows = rows;
      await model.reset();
      await model.forward(ids);
      GpuForward.takeProfile();
      var differing = 0, nonFinite = 0;
      var maxDiff = 0.0;
      final wall = Stopwatch()..start();
      var wallMicros = 0;
      for (var i = 0; i < 16; i++) {
        wall.reset();
        await model.step(tokens[i]);
        wallMicros += wall.elapsedMicroseconds;
        final actual = model.logits;
        final expected = reference[i];
        final bits = actual.buffer.asUint32List(
          actual.offsetInBytes,
          actual.length,
        );
        final refBits = expected.buffer.asUint32List(
          expected.offsetInBytes,
          expected.length,
        );
        for (var j = 0; j < actual.length; j++) {
          if (bits[j] != refBits[j]) differing++;
          if (!actual[j].isFinite || !expected[j].isFinite) {
            nonFinite++;
          } else {
            final diff = (actual[j] - expected[j]).abs();
            if (diff > maxDiff) maxDiff = diff;
          }
        }
      }
      final profile = GpuForward.takeProfile();
      final gpuMs = profile.values.fold(0.0, (sum, p) => sum + p.$1) / 16;
      log(
        'Decode sweep round ${round + 1}, rows $rows, Q8 group $workgroup, vector $vector: '
        '${gpuMs.toStringAsFixed(2)} GPU ms/token, '
        '${(wallMicros / 16000).toStringAsFixed(2)} wall ms/token, '
        '$differing differing logits, max |d| $maxDiff, non-finite $nonFinite',
      );
      (vectorExperiment
              ? vectorSweep
              : variant >= 64
              ? q8Sweep
              : sweep)
          .add({
            'round': round + 1,
            'rows': rows,
            'q8_workgroup': workgroup,
            'q8_vector': vector,
            'q8_ms_per_token':
                ((profile['gemv_q8_i8']?.$1 ?? 0.0) +
                    (profile['gemv_q8_i8_vec']?.$1 ?? 0.0)) /
                16,
            'gpu_ms_per_token': gpuMs,
            'wall_ms_per_token': wallMicros / 16000,
            'differing_logits': differing,
            'max_abs_diff': maxDiff,
            'non_finite': nonFinite,
          });
      if (round == 0 && variant < 64) {
        record['decode_rows_$rows'] = perToken(profile, 16);
      }
    }
  }
  record['decode_sweep'] = sweep;
  record['q8_group_sweep'] = q8Sweep;
  record['q8_vector_sweep'] = vectorSweep;
  GpuForward.decodeRows = 4;
  GpuForward.restoreQ8VectorDefault();
  GpuForward.setQ8Workgroup(128);
  GpuForward.profiling = false;
  final bandwidth = GpuForward.copyBandwidth(128);
  log('GPU copy bandwidth: ${bandwidth.toStringAsFixed(1)} GB/s');
  record['copy_gb_per_s'] = bandwidth;
  return record;
}

/// The tool-calling question, greedy, as `lfm3 ask` runs it.
Future<Map<String, Object?>> _ask(
  Lfm3Model model,
  Lfm3Tokenizer tokenizer,
  List<int> ids,
  void Function(String) log,
  String label,
) async {
  await model.reset();
  final thermalBefore = _thermal();
  final gpu = model.gpu;
  gpu?.clearStats();
  final total = Stopwatch()..start();
  final sw = Stopwatch()..start();
  await model.forward(ids);
  final promptS = sw.elapsedMicroseconds / 1e6;
  final promptGpuMs = gpu == null ? null : gpu.gpuMicros / 1000;
  gpu?.clearStats();
  final gen = <int>[];
  sw
    ..reset()
    ..start();
  while (gen.length < 64) {
    final next = _argmax(model.logits);
    if (next == model.config.eosTokenId) break;
    gen.add(next);
    await model.step(next);
  }
  final decodeS = sw.elapsedMicroseconds / 1e6;
  final totalS = total.elapsedMicroseconds / 1e6;
  // Where a decode step's time goes: GPU execution, and Dart encoding.
  final decodeGpuMs = gpu == null || gpu.calls == 0
      ? null
      : gpu.gpuMicros / 1000 / gpu.calls;
  final decodeEncodeMs = gpu == null || gpu.calls == 0
      ? null
      : gpu.encodeMicros / 1000 / gpu.calls;
  final text = tokenizer.decode(gen);
  final calls = parseAssistantReply(text).toolCalls;
  final result = {
    'prompt_tokens': ids.length,
    'prompt_s': promptS,
    'prompt_tok_per_s': ids.length / promptS,
    'gen_tokens': gen.length,
    'decode_s': decodeS,
    'decode_tok_per_s': gen.length / decodeS,
    'prompt_gpu_ms': promptGpuMs,
    'decode_gpu_ms_per_token': decodeGpuMs,
    'decode_encode_ms_per_token': decodeEncodeMs,
    'total_s': totalS,
    'text': text,
    'tool_calls': [for (final c in calls) c.toJson()],
    'thermal_before': thermalBefore,
    'thermal_after': _thermal(),
    'footprint_mb': _mb(physFootprint()),
  };
  log(
    '$label: prompt ${(ids.length / promptS).toStringAsFixed(1)} tok/s, '
    'decode ${(gen.length / decodeS).toStringAsFixed(1)} tok/s, '
    'total ${totalS.toStringAsFixed(2)} s, '
    '${decodeGpuMs == null ? '' : 'decode GPU ${decodeGpuMs.toStringAsFixed(1)} ms/token '
              '(wall ${(decodeS * 1000 / gen.length).toStringAsFixed(1)}, encoding '
              '${decodeEncodeMs!.toStringAsFixed(2)}), '}'
    'calls ${[for (final c in calls) '${c.name}${c.arguments}']}, '
    'thermal $thermalBefore→${_thermal()}, '
    'footprint ${_mb(physFootprint())} MB',
  );
  return result;
}

/// A teacher-forced run of a reference case, written in the layout of
/// `lfm3 run --force` (see bin/lfm3.dart). Prompt logits are kept for every
/// position of prompts up to 128 tokens, else for the last one only.
Future<Map<String, Object?>> _forced(
  Lfm3Model model,
  String weights,
  String refDir,
  String outDir,
  String name,
  void Function(String) log,
  String label,
) async {
  final ids = Int32List.sublistView(File('$refDir/ids.i32').readAsBytesSync());
  final forced = Int32List.sublistView(
    File('$refDir/gen.i32').readAsBytesSync(),
  );
  await model.reset();
  final c = model.config;
  final h = c.hiddenSize;
  final slots = c.numLayers + 1;
  final t = ids.length;
  final positions = t <= 128 ? List<int>.generate(t, (i) => i) : [t - 1];
  final hidden = Float32List(t * slots * h);
  final sw = Stopwatch()..start();
  await model.forward(
    ids,
    capture: (slot, first, rows, count) {
      for (var r = 0; r < count; r++) {
        final pos = first + r;
        hidden.setRange(
          (pos * slots + slot) * h,
          (pos * slots + slot + 1) * h,
          rows,
          r * h,
        );
      }
    },
  );
  final promptS = sw.elapsedMicroseconds / 1e6;
  final lastLogits = Float32List.fromList(model.logits);
  final promptLogits = Float32List(positions.length * c.vocabSize);
  for (var i = 0; i < positions.length; i++) {
    final base = (positions[i] * slots + c.numLayers) * h;
    promptLogits.setRange(
      i * c.vocabSize,
      (i + 1) * c.vocabSize,
      await model.logitsOf(Float32List.sublistView(hidden, base, base + h)),
    );
  }
  model.logits.setAll(0, lastLogits);

  final gen = <int>[];
  final genLogits = <Float32List>[];
  var decodeSteps = 0;
  sw
    ..reset()
    ..start();
  while (gen.length < forced.length) {
    final next = forced[gen.length];
    gen.add(next);
    genLogits.add(Float32List.fromList(model.logits));
    if (next == c.eosTokenId || gen.length == forced.length) break;
    await model.step(next);
    decodeSteps++;
  }
  final decodeS = sw.elapsedMicroseconds / 1e6;

  Directory(outDir).createSync(recursive: true);
  void write(String file, TypedData data) =>
      File('$outDir/$file').writeAsBytesSync(
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
      );
  write('hidden.f32', hidden);
  write('logits.f32', promptLogits);
  write('gen.i32', Int32List.fromList(gen));
  final flat = Float32List(genLogits.length * c.vocabSize);
  for (var i = 0; i < genLogits.length; i++) {
    flat.setRange(i * c.vocabSize, (i + 1) * c.vocabSize, genLogits[i]);
  }
  write('gen_logits.f32', flat);
  final meta = {
    'case': name,
    'seq_len': t,
    'hidden_size': h,
    'num_layers': c.numLayers,
    'vocab_size': c.vocabSize,
    'logit_positions': positions,
    'gen_len': gen.length,
    'generated': '',
    'weights': weights,
    'forced': true,
    'metal': model.gpu != null,
    'prompt_s': promptS,
    'prompt_tok_per_s': t / promptS,
    'decode_s': decodeS,
    'decode_tok_per_s': decodeSteps > 0 ? decodeSteps / decodeS : null,
  };
  File(
    '$outDir/meta.json',
  ).writeAsStringSync(const JsonEncoder.withIndent('  ').convert(meta));
  log(
    '$label: $t prompt tokens at ${(t / promptS).toStringAsFixed(1)} tok/s, '
    '$decodeSteps forced steps at '
    '${decodeSteps > 0 ? (decodeSteps / decodeS).toStringAsFixed(1) : '-'} tok/s',
  );
  return meta;
}
