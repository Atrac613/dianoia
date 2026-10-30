/// On-device LLM inference orchestrated in Dart: tokenizer, chat template,
/// tool-call parsing and the LFM3 forward pass, on native CPU kernels and a
/// Metal or Vulkan GPU layer compiled by the package's build hook.
library;

export 'src/chat_bench.dart';
export 'src/chat_session.dart';
export 'src/chat_template.dart';
export 'src/config.dart';
export 'src/executor.dart';
export 'src/linear.dart';
export 'src/gguf.dart';
export 'src/matmul.dart';
export 'src/gpu.dart';
export 'src/model.dart';
export 'src/native.dart';
export 'src/ops.dart';
export 'src/q8_cache.dart';
export 'src/quant.dart';
export 'src/safetensors.dart';
export 'src/shard.dart';
export 'src/tokenizer.dart';
export 'src/tool_calls.dart';
