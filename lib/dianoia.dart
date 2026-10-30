/// On-device LLM inference orchestrated in Dart, on native CPU kernels and
/// a Metal or Vulkan GPU layer compiled by the package's build hook.
///
/// Load a model bundle with [LocalModel.load] and talk to it through a
/// [ChatSession] from [LocalModel.chat]:
///
/// ```dart
/// final model = await LocalModel.load(dir);
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
export 'src/tool_calls.dart';
