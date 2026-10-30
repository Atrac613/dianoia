/// The entry point for apps: a model bundle on this device, loaded onto the
/// GPU where there is one, ready for conversations.
library;

import 'dart:ffi';
import 'dart:io';

import 'chat_session.dart';
import 'model.dart';
import 'native.dart';
import 'quant.dart';
import 'tokenizer.dart';

/// Where the forward pass runs.
enum InferenceBackend {
  /// The GPU where it can run the model, the CPU kernels otherwise.
  auto,

  /// Metal on Apple platforms, Vulkan on Android.
  gpu,

  /// The native CPU kernels.
  cpu,
}

/// A language model on this device: its weights mapped, the native kernels
/// and, where possible, the GPU ready. Conversations come from [chat].
///
/// Conversations share the model's state: switching between two costs the
/// prompt tokens they do not have in common, and only one may generate at
/// a time.
final class LocalModel {
  LocalModel._(this._model, this._tokenizer, this.backend);

  final Lfm3Model _model;
  final Lfm3Tokenizer _tokenizer;

  /// Where the model runs: [InferenceBackend.gpu] or [InferenceBackend.cpu].
  final InferenceBackend backend;

  /// The most tokens a conversation can hold, prompt and replies together.
  int get contextLength => _model.maxSeq;

  /// Whether [close] has run.
  bool get isClosed => _model.isClosed;

  /// Frees the model: its GPU memory and its mappings of the bundle's
  /// files. Its conversations stop working: a reply being generated fails
  /// at its next token, and [ChatSession.send] throws [StateError]. Closing
  /// twice does nothing.
  Future<void> close() => _model.close();

  /// Loads the model bundle in [dir]: `config.json`, `tokenizer.json`,
  /// `model.safetensors` and the quantized weights, `lfm3-q4.cache` (from
  /// `dianoia import-gguf`) or `lfm3-q8.cache` (from `dianoia pack`).
  ///
  /// With [InferenceBackend.auto] a GPU that cannot run the model (none, or
  /// without the features the kernels need) leaves it on the CPU kernels.
  /// GPU attention keeps a score per key in on-chip memory, so a long
  /// [contextLength] may not fit on the GPU; 2048 does on the phones
  /// measured. [threads] native threads share each CPU matrix product.
  static Future<LocalModel> load(
    String dir, {
    InferenceBackend backend = InferenceBackend.auto,
    int contextLength = 2048,
    int threads = 4,
  }) async {
    final weights = _bundleWeights(dir);
    if (!NativeKernels.isLoaded) {
      if (!NativeKernels.isAvailable) {
        throw UnsupportedError(
          'dianoia has no native kernels for ${Abi.current()}; it runs on '
          'arm64 macOS, iOS and Android',
        );
      }
      NativeKernels.load(null, threads: threads);
    }
    Future<Lfm3Model> open({required bool gpu}) => Lfm3Model.load(
      dir,
      maxSeq: contextLength,
      weights: weights,
      q8CachePath: '$dir/lfm3-${weights.name}.cache',
      gpu: gpu,
    );

    var used = backend == InferenceBackend.cpu
        ? InferenceBackend.cpu
        : InferenceBackend.gpu;
    Lfm3Model model;
    try {
      model = await open(gpu: used == InferenceBackend.gpu);
    } on Error catch (e) {
      // StateError: no usable GPU; UnsupportedError: the model needs what
      // the GPU path lacks.
      if (backend != InferenceBackend.auto ||
          (e is! StateError && e is! UnsupportedError)) {
        rethrow;
      }
      used = InferenceBackend.cpu;
      model = await open(gpu: false);
    }
    return LocalModel._(model, Lfm3Tokenizer.load(dir), used);
  }

  /// A new conversation with [system] as its system prompt and [tools]
  /// (OpenAI-style function definitions) offered to the model. Replies
  /// stop after [maxNewTokens].
  ChatSession chat({
    String? system,
    List<Map<String, Object?>>? tools,
    int maxNewTokens = 512,
  }) => ChatSession(
    _model,
    _tokenizer,
    system: system,
    tools: tools,
    maxNew: maxNewTokens,
  );

  /// The weights the bundle in [dir] holds: Q4 if it has a Q4 cache, Q8
  /// if it has a Q8 one.
  static WeightFormat _bundleWeights(String dir) {
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
}
