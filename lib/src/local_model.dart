/// The entry point for apps: a model bundle on this device, loaded onto the
/// GPU where there is one, ready for conversations.
library;

import 'chat_session.dart';
import 'control_vector.dart';
import 'model.dart';
import 'model_loading.dart';
import 'model_use.dart';
import 'native.dart';
import 'sampler.dart';
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
  /// The GPU where it can run the model, the CPU kernels otherwise, and
  /// the CPU kernels where the GPU is known to be slower: Qwen3 on Android.
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

  final TextModel _model;
  final BpeTokenizer _tokenizer;

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
  /// `model.safetensors` and the quantized weights, named by architecture
  /// and format (`lfm3-q4.cache` from `dianoia import-gguf`,
  /// `qwen3-q8.cache` from `dianoia pack`, and so on). [ModelStore]
  /// installs such bundles.
  ///
  /// With [InferenceBackend.auto] a GPU that cannot run the model (none, or
  /// without the features the kernels need) leaves it on the CPU kernels,
  /// and so does Android for Qwen3, whose decoding is faster there on the
  /// CPU.
  ///
  /// [contextLength] is the most tokens a conversation holds, prompt and
  /// replies together: by default 4096 for Qwen3, whose thinking runs
  /// long, except on Vulkan, where its KV cache stays float32 and twice
  /// the size, and 2048 for LFM3. GPU attention keeps a score per key in
  /// on-chip memory, so a much longer context may not fit on the GPU; 4096
  /// does on the phones measured. [threads] native threads share each CPU
  /// matrix product.
  static Future<LocalModel> load(
    String dir, {
    InferenceBackend backend = InferenceBackend.auto,
    int? contextLength,
    int threads = 4,
  }) async {
    final weights = bundleWeights(dir); // fails before anything loads
    final tokenizer = BpeTokenizer.load(dir);
    ensureNativeKernels(threads);
    final model = await openLanguageBundle(
      dir,
      weights: weights,
      backend: backend,
      contextLength: contextLength,
    );
    return LocalModel._(
      model,
      tokenizer,
      model.gpu == null ? InferenceBackend.cpu : InferenceBackend.gpu,
    );
  }

  /// A new conversation with [system] as its system prompt and [tools]
  /// (OpenAI-style function definitions) offered to the model. Replies
  /// stop after [maxNewTokens] and pick their tokens under [sampling]
  /// (greedy by default). With [enableThinking] the model reasons before
  /// answering; the reasoning arrives in [ChatTurn.reply]. Reasoning models
  /// tend to repeat themselves under greedy decoding, so pair it with
  /// sampling, such as [SamplingParams.qwen3Thinking].
  ///
  /// [maxNewTokens] defaults to 512, or with [enableThinking] to the whole
  /// [contextLength]: reasoning often runs past a thousand tokens (1,699 for
  /// Qwen3-1.7B on one question), and a reply cut inside it has no answer.
  /// Replies always stop where the context ends.
  ChatSession chat({
    String? system,
    List<Map<String, Object?>>? tools,
    int? maxNewTokens,
    SamplingParams sampling = SamplingParams.greedy,
    bool enableThinking = false,
  }) => ChatSession(
    _model,
    _tokenizer,
    system: system,
    tools: tools,
    maxNew: maxNewTokens ?? (enableThinking ? contextLength : 512),
    sampling: sampling,
    enableThinking: enableThinking,
  );
}

/// Experimental, lab-only: steering a [LocalModel] with a control vector
/// (docs/control_vectors.md). `package:dianoia/lab.dart` exports this
/// extension; `package:dianoia/dianoia.dart` hides it, so apps see no new
/// member on [LocalModel].
extension LocalModelControlVector on LocalModel {
  /// The control vector steering this model's replies, or null.
  ControlVector? get controlVector => _model.controlVector;

  /// Steers every following reply of every conversation of this model with
  /// [vector], or stops steering with null.
  ///
  /// The model's computed state holds only under the vector it was computed
  /// with, so a different vector resets it: each [ChatSession] keeps its
  /// messages, and its next [ChatSession.send] prefills them again under the
  /// new vector, giving what a model steered from the start would. Saved
  /// states (`warmUp(statePath:)`) record the vector and are not loaded
  /// under another. A vector that steers exactly as the current one changes
  /// nothing.
  ///
  /// Throws [StateError] while a reply or a warm-up is running (call it
  /// between replies), [ArgumentError] for a vector of another model.
  Future<void> setControlVector(ControlVector? vector) =>
      _steer(_model, vector);
}

/// Sets [vector] on [model], resetting its sequence if the vector changes
/// the arithmetic, unless a conversation is using it.
Future<void> _steer(TextModel model, ControlVector? vector) async {
  if (modelInUse(model)) {
    throw StateError(
      'a reply is being generated; set the control vector between replies',
    );
  }
  if (vector?.fingerprint == model.controlVector?.fingerprint) {
    model.setControlVector(vector);
    return;
  }
  await model.reset();
  model.setControlVector(vector);
}

/// Experimental, lab-only: steering a [LocalVisionModel]'s language
/// backbone with a control vector, as [LocalModelControlVector] does for a
/// [LocalModel]. Every visual turn prefills from scratch, so a new vector
/// applies from the next turn on.
extension LocalVisionModelControlVector on LocalVisionModel {
  /// The control vector steering this model's replies, or null.
  ControlVector? get controlVector => _runtime.model.controlVector;

  /// Steers every following turn with [vector], or stops with null. Throws
  /// [StateError] while a turn is running, [ArgumentError] for a vector of
  /// another model.
  Future<void> setControlVector(ControlVector? vector) =>
      _steer(_runtime.model, vector);
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

  /// The default [maxImageTokens]: a 4032x3024 photo becomes 972 tokens,
  /// which leaves room for a conversation in a 2048-token context.
  static const defaultMaxImageTokens = 1024;

  /// The most tokens one image is resized to: the load's request, or the
  /// checkpoint's own budget if that is smaller.
  int get maxImageTokens => _runtime.processor.maxImageTokens;

  /// Loads original VL vision weights and a matching prepared language bundle.
  /// Checks import fingerprints and special-token IDs before allocating model
  /// resources. The bundle must contain a Q8 or Q4 cache as LocalModel expects.
  ///
  /// [maxImageTokens] lowers the checkpoint's pixel budget (16,384 tokens for
  /// LFM3-VL, beyond any on-device context) so larger images are downscaled
  /// with the upstream resize rule. Images within it are unaffected.
  static Future<LocalVisionModel> load(
    String modelDir, {
    required String backboneDir,
    InferenceBackend backend = InferenceBackend.auto,
    int contextLength = 2048,
    int threads = 4,
    int maxImageTokens = defaultMaxImageTokens,
  }) async {
    validateVlBackbone(modelDir, backboneDir);
    final config = Lfm3VlConfig.load(modelDir);
    final processor = Lfm3ImageProcessor.load(
      modelDir,
    ).withMaxImageTokens(maxImageTokens);
    if (processor.patchSize != config.vision.patchSize ||
        processor.mergeSize != config.vision.spatialMergeSize ||
        processor.temporalPatchSize != config.vision.temporalPatchSize ||
        config.vision.inChannels != 3) {
      throw const FormatException('processor and vision configuration differ');
    }
    final tokenizer = BpeTokenizer.load(modelDir);
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
      // Q8 linears run in the native kernels, or on the GPU with the
      // language model, instead of in Dart.
      final encoder = Lfm3VisionEncoder.load(modelDir, quantize: true);
      try {
        if (language.backend == InferenceBackend.gpu) encoder.useGpu();
      } catch (_) {
        encoder.close();
        rethrow;
      }
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
    int maxImageTokens = defaultMaxImageTokens,
  }) async {
    await validateVlBundle(directory);
    return load(
      directory,
      backboneDir: '$directory/language',
      backend: backend,
      contextLength: contextLength,
      threads: threads,
      maxImageTokens: maxImageTokens,
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
