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
import 'vision_backbone.dart';
import 'vision_bundle.dart';
import 'vision_config.dart';
import 'vision_encoder.dart';
import 'vision_image.dart';
import 'vision_prompt.dart';
import 'vision_session.dart';

/// Where the forward pass runs.
enum InferenceBackend {
  /// The GPU where it can run the model, the CPU kernels otherwise.
  auto,

  /// Metal on Apple platforms, Vulkan on Android and Windows.
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

  /// Whether models can run here: the native kernels were built for this
  /// platform and the CPU has what they need (the int8 dot product on
  /// arm64, AVX2 on x86-64). An app should check it before offering
  /// on-device models at all.
  static bool get isSupported => NativeKernels.isAvailable;

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
    final tokenizer = Lfm3Tokenizer.load(dir);
    if (!NativeKernels.isLoaded) {
      if (!NativeKernels.isAvailable) {
        throw UnsupportedError(
          'dianoia has no native kernels for ${Abi.current()}; it runs on '
          'arm64 macOS, iOS and Android and on x86-64 Windows with AVX2',
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
    return LocalModel._(model, tokenizer, used);
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

/// An experimental visual model using the CPU reference vision encoder and
/// a quantized CPU/Metal/Vulkan language backbone. Load and run it in an
/// inference isolate: each vision layer performs synchronous CPU work.
/// Prepare the checkpoint and quantized backbone with the import/vision-run
/// tools before loading, or install a prebuilt VL bundle with ModelStore.
final class LocalVisionModel {
  LocalVisionModel._(this._runtime, this.backend);
  final VisionChatRuntime _runtime;

  /// Where the language backbone runs. Image encoding currently uses the CPU.
  final InferenceBackend backend;
  static bool get isSupported => LocalModel.isSupported;
  int get contextLength => _runtime.model.maxSeq;
  bool get isClosed => _runtime.isClosed;

  /// Loads original VL vision weights and a matching prepared language bundle.
  /// Checks import fingerprints and special-token IDs before allocating model
  /// resources. The bundle must contain a Q8 or Q4 cache as LocalModel expects.
  static Future<LocalVisionModel> load(
    String modelDir, {
    required String backboneDir,
    InferenceBackend backend = InferenceBackend.auto,
    int contextLength = 2048,
    int threads = 4,
  }) async {
    validateVlBackbone(modelDir, backboneDir);
    final config = Lfm3VlConfig.load(modelDir);
    final processor = Lfm3ImageProcessor.load(modelDir);
    if (processor.patchSize != config.vision.patchSize ||
        processor.mergeSize != config.vision.spatialMergeSize ||
        processor.temporalPatchSize != config.vision.temporalPatchSize ||
        config.vision.inChannels != 3) {
      throw const FormatException('processor and vision configuration differ');
    }
    final tokenizer = Lfm3Tokenizer.load(modelDir);
    // Verify special tokens using the same path as real visual prompts.
    prepareVisionPrompt(
      [],
      tokenizer: tokenizer,
      config: config,
      imageTokenCounts: const [],
    );
    final language = await LocalModel.load(
      backboneDir,
      backend: backend,
      contextLength: contextLength,
      threads: threads,
    );
    try {
      final encoder = Lfm3VisionEncoder.load(modelDir);
      return LocalVisionModel._(
        VisionChatRuntime(language._model, tokenizer, processor, encoder),
        language.backend,
      );
    } catch (_) {
      await language.close();
      rethrow;
    }
  }

  /// Loads a portable bundle produced by `dianoia pack-vl`. Verifies every
  /// recorded file's size and SHA-256 before opening native model resources.
  /// The original checkpoint and full imported backbone are not needed.
  static Future<LocalVisionModel> loadBundle(
    String directory, {
    InferenceBackend backend = InferenceBackend.auto,
    int contextLength = 2048,
    int threads = 4,
  }) async {
    await validateVlBundle(directory);
    return load(
      directory,
      backboneDir: '$directory/language',
      backend: backend,
      contextLength: contextLength,
      threads: threads,
    );
  }

  VisionChatSession chat({
    String? system,
    List<Map<String, Object?>>? tools,
    int maxNewTokens = 512,
  }) => _runtime.chat(system: system, tools: tools, maxNewTokens: maxNewTokens);

  /// Cancels active work, waits for it to settle, then releases image weights
  /// and the language model. New requests are rejected once closing starts.
  Future<void> close() => _runtime.close();
}
