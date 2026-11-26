/// Opening a language bundle on the backend an app asked for, shared by
/// the app-facing loaders. Internal: neither `dianoia.dart` nor `lab.dart`
/// exports it.
library;

import 'dart:ffi';
import 'dart:io';

import 'local_model.dart' show InferenceBackend;
import 'model.dart';
import 'native.dart';
import 'quant.dart';
import 'config.dart';
import 'text_bundle.dart';

/// Loads the native kernels with [threads] native threads unless they are
/// loaded already, failing clearly where none were built.
void ensureNativeKernels(int threads) {
  if (NativeKernels.isLoaded) return;
  if (!NativeKernels.isAvailable) {
    throw UnsupportedError(
      'dianoia has no native kernels for ${Abi.current()}; it runs on '
      'arm64 macOS, iOS and Android and on x86-64 Windows with AVX2',
    );
  }
  NativeKernels.load(null, threads: threads);
}

/// The context a model family gets when the app names none: tokens of
/// prompt and replies together. Qwen3 thinks at length (1.7B took 1,699
/// tokens on one question), and its half KV cache holds 4,096 tokens in the
/// 470 MB that 2,048 float32 tokens took; LFM3 keeps 2,048.
int defaultContextLength(TextModelConfig config) =>
    config.architecture == ModelArchitecture.qwen3 ? 4096 : 2048;

/// Opens the language bundle in [dir], holding [weights] (see
/// [bundleWeights]), on [backend], with [contextLength] tokens or the
/// family's [defaultContextLength]. With [InferenceBackend.auto] a GPU that
/// cannot run the model, or one slower than the CPU for it
/// ([autoPrefersCpu]), leaves it on the CPU kernels; [TextModel.gpu] says
/// which ran.
Future<TextModel> openLanguageBundle(
  String dir, {
  required WeightFormat weights,
  required InferenceBackend backend,
  int? contextLength,
}) async {
  final config = TextModelConfig.load(dir);
  final maxSeq = contextLength ?? defaultContextLength(config);
  Future<TextModel> open({required bool gpu}) => TextModel.load(
    dir,
    maxSeq: maxSeq,
    weights: weights,
    q8CachePath: '$dir/${quantizedCacheName(config, weights)}',
    gpu: gpu,
  );

  if (backend == InferenceBackend.auto &&
      autoPrefersCpu(config, android: Platform.isAndroid)) {
    return open(gpu: false);
  }
  try {
    return await open(gpu: backend != InferenceBackend.cpu);
  } on Error catch (e) {
    // StateError: no usable GPU; UnsupportedError: the model needs what
    // the GPU path lacks.
    if (backend != InferenceBackend.auto ||
        (e is! StateError && e is! UnsupportedError)) {
      rethrow;
    }
    return open(gpu: false);
  }
}

/// Whether [InferenceBackend.auto] runs a model of [config] on the CPU
/// kernels even where the GPU could run it. On Android it does for Qwen3:
/// a Pixel 8's Mali GPU decodes Qwen3 a little slower than its CPU, with
/// matrix-vector products bound by memory bandwidth whatever their layout,
/// and needs more memory (docs/qwen3_support.md). It prefills about twice
/// as fast, so an app with long prompts may still ask for
/// [InferenceBackend.gpu].
bool autoPrefersCpu(TextModelConfig config, {required bool android}) =>
    android && config.architecture == ModelArchitecture.qwen3;

/// The weights the bundle in [dir] holds: Q4 if it has a Q4 cache, Q8 if
/// it has a Q8 one (`lfm3-q4.cache`, `qwen3-q8.cache` and so on).
WeightFormat bundleWeights(String dir) {
  if (!File('$dir/config.json').existsSync()) {
    throw ArgumentError.value(
      dir,
      'dir',
      'holds no model bundle (config.json); make one with '
          '`dianoia import-gguf` or `dianoia pack`',
    );
  }
  final config = TextModelConfig.load(dir);
  for (final weights in [WeightFormat.q4, WeightFormat.q8]) {
    if (File('$dir/${quantizedCacheName(config, weights)}').existsSync()) {
      return weights;
    }
  }
  throw ArgumentError.value(
    dir,
    'dir',
    'holds no quantized weights (${quantizedCacheName(config, WeightFormat.q4)} '
        'or ${quantizedCacheName(config, WeightFormat.q8)}); make them with '
        '`dianoia import-gguf` or `dianoia pack`',
  );
}
