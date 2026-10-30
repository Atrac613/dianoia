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
/// `package:dianoia/lab.dart` exposes the engine's internals.
library;

export 'src/chat_session.dart'
    show ChatSession, ChatTurn, ContextLengthExceeded;
export 'src/local_model.dart';
export 'src/model_store.dart';
export 'src/tool_calls.dart';
