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

/// Opens the language bundle in [dir], holding [weights] (see
/// [bundleWeights]), on [backend]. With [InferenceBackend.auto] a GPU that
/// cannot run the model leaves it on the CPU kernels; [Lfm3Model.gpu] says
/// which ran.
Future<Lfm3Model> openLanguageBundle(
  String dir, {
  required WeightFormat weights,
  required InferenceBackend backend,
  required int contextLength,
}) async {
  Future<Lfm3Model> open({required bool gpu}) => Lfm3Model.load(
    dir,
    maxSeq: contextLength,
    weights: weights,
    q8CachePath: '$dir/lfm3-${weights.name}.cache',
    gpu: gpu,
  );

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

/// The weights the bundle in [dir] holds: Q4 if it has a Q4 cache, Q8 if
/// it has a Q8 one.
WeightFormat bundleWeights(String dir) {
  for (final weights in [WeightFormat.q4, WeightFormat.q8]) {
    if (File('$dir/lfm3-${weights.name}.cache').existsSync()) {
      return weights;
    }
  }
  throw ArgumentError.value(
    dir,
    'dir',
    'holds no quantized weights (lfm3-q4.cache or lfm3-q8.cache); make '
        'them with `dianoia import-gguf` or `dianoia pack`',
  );
}
