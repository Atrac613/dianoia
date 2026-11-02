/// On-device LLM inference orchestrated in Dart, on native CPU kernels and
/// a Metal or Vulkan GPU layer compiled by the package's build hook.
///
/// Install a model with a [ModelStore], load its bundle with
/// [LocalModel.load] and talk to it through a [ChatSession] from
/// [LocalModel.chat]:
///
/// ```dart
/// final store = ModelStore(dir);
/// await store.install(ModelSpec.lfm3Q4).drain<void>();
/// final model = await LocalModel.load(store.bundlePath(ModelSpec.lfm3Q4));
/// final chat = model.chat(system: 'Answer briefly.');
/// final turn = await chat.send({'role': 'user', 'content': 'Hi!'});
/// print(turn.reply.content);
/// ```
///
/// Decoded RGB image conversations use [LocalVisionModel], [ChatImage] and
/// [VisionChatSession]. Install prebuilt VL bundles with [VisionModelSpec] and
/// [ModelStore.install], or build the official checkpoint on device with
/// [VisionCheckpointSpec.lfm3VlQ8], or import paired official QAD GGUFs with
/// [VisionGgufSpec.lfm3VlQadQ4]. Load it with [LocalVisionModel.loadBundle]. Image encoding
/// currently runs on the CPU. Use [ChatImage.encoded] or [ChatImage.fromFile]
/// for PNG/JPEG input, or [ChatImage.rgb] for caller-decoded pixels.
/// [AudioGgufSpec.lfm3AudioQ4] installs the paired audio model. Audio inference
/// remains experimental under `package:dianoia/lab.dart` and in the example.
/// `package:dianoia/lab.dart` exposes the engine's internals.
library;

export 'src/chat_session.dart'
    show ChatSession, ChatTurn, ContextLengthExceeded;
export 'src/local_model.dart';
export 'src/model_store.dart';
export 'src/tool_calls.dart';
export 'src/vision_session.dart'
    show
        ChatImage,
        VisionChatSession,
        VisionRequestCancelled,
        VisionProgress,
        VisionPhase;
