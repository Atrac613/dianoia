/// Everything inside dianoia, for the command-line tool, benchmarks and
/// research: the forward pass and its executors, the kernels and their
/// tuning, quantization, file formats and the measurement helpers.
///
/// Apps should import `package:dianoia/dianoia.dart` instead. What is here
/// changes without notice.
library;

export 'dianoia.dart';
export 'src/chat_bench.dart';
export 'src/chat_template.dart';
export 'src/config.dart';
export 'src/executor.dart';
export 'src/gguf.dart';
export 'src/gpu.dart';
export 'src/linear.dart';
export 'src/matmul.dart';
export 'src/model.dart';
export 'src/native.dart';
export 'src/ops.dart';
export 'src/q8_cache.dart';
export 'src/quant.dart';
export 'src/safetensors.dart';
export 'src/shard.dart';
export 'src/tokenizer.dart';
