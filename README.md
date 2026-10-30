# dianoia

On-device LLM inference orchestrated in Dart. Dart owns the model: the
tokenizer, the chat template, tool-call parsing, the graph of the forward
pass and the state between turns. Native code does only the arithmetic: CPU
kernels for quantized matrix products and attention, and a GPU layer (Metal
on Apple platforms, Vulkan on Android) that records one dispatch per Dart
call.

> **Status: 0.1.0-dev.** The engine was imported from
> [dart-llm-lab](#origin) and runs one model family, LFM3, for text chat and
> tool calling. The public API is the lab's and will change. Planned: LFM3
> audio and general GGUF models.

## What runs where

| Target | CPU kernels | GPU | Measured (LFM3-1.7B, QAD Q4_0, GPU) |
|---|---|---|---|
| macOS, Apple silicon | NEON `sdot`, Grand Central Dispatch | Metal | M2 Max: ~280 tok/s prompt, ~90 tok/s decode |
| iOS, A14 or later | NEON `sdot`, Grand Central Dispatch | Metal | iPhone: ~80 tok/s prompt, 29.5 tok/s decode, 0.56 s to first token |
| Android, arm64 with dotprod | NEON `sdot`, OpenMP | Vulkan 1.2 (int8 dot products where present) | Pixel 8: ~82 tok/s prompt, ~18 tok/s sustained decode |
| Anything else | none: Dart only | none | slow, but correct |

Reusing state is exact: a conversation that continues from a saved
checkpoint, in memory or in a state file, produces the logits of a run from
scratch bit for bit. With the system prompt and tool catalog warmed up once,
a new request waits only for its own tokens (0.3 s to the first token on a
Pixel 8, 0.56 s on an iPhone).

## Requirements

- Dart SDK 3.10 or newer (build hooks). Flutter 3.38 or newer for apps.
- A C compiler for the target: Xcode for Apple platforms, the Android NDK
  for Android (Flutter supplies both through its toolchain). No Metal
  compiler and no `glslc` are needed: Metal shaders compile when the GPU
  starts, and the Vulkan shaders ship as SPIR-V.
- Model weights, which are not part of this package. LFM3-1.7B is
  [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) under
  the LFM Open License v1.0; read its terms before you ship it.

## Setup

```bash
dart pub get
dart test        # runs the build hook, then checks the kernels it built
```

The first `dart run`, `dart test` or `flutter run` compiles the native
kernels through `hook/build.dart` into one code asset,
`package:dianoia/kernels`, and caches it.

### Getting a model

Apps download a model with a `ModelStore`. `ModelSpec.lfm3Q4` pins Liquid's
quantization-aware Q4_0 GGUF and the matching config, tokenizer and license
on Hugging Face by revision, size and SHA-256 (1.07 GB); the store fetches
them, resuming interrupted downloads, builds the bundle on the device in a
background isolate and deletes the downloads. dianoia redistributes no
weights. `spec.fromMirror(url)` fetches the same pinned files from a mirror
of your own.

```dart
final store = ModelStore('${supportDir.path}/models');
const spec = ModelSpec.lfm3Q4;
if (!store.isInstalled(spec)) {
  // Show spec.name, spec.downloadBytes and spec.license first.
  await for (final progress in store.install(spec)) {
    print('${progress.stage.name} ${(progress.fraction * 100).round()}%');
  }
}
final model = await LocalModel.load(store.bundlePath(spec));
```

On the command line, a bundle can also be made from local files. A model
directory holds the checkpoint's `config.json`, `tokenizer.json` and
`model.safetensors`. Q8 and Q4 weights live in a cache file next to
them, built on first use. For the quantization-aware distilled Q4 weights,
import Liquid's GGUF into a smaller bundle (about 1.1 GB):

```bash
dart run dianoia import-gguf --gguf=LFM3-1.7B-QAD-Q4_0.gguf \
  --model=path/to/LFM3-1.7B --out=path/to/LFM3-1.7B-QAD-Q4
```

## Usage

### Command line

```bash
dart run dianoia ask --model=path/to/LFM3-1.7B-QAD-Q4 --weights=q4 --gpu \
  --tools=example/weather_tools.json --prompt="Is it raining in Kyoto?"
```

`dart run dianoia` lists the commands: `ask`, `run`, `selftest`, `pack`,
`import-gguf`, `toolbench` and `chatbench`. Their flags are documented at
the top of `bin/dianoia.dart`.

### Library

```dart
import 'package:dianoia/dianoia.dart';

// The GPU where it can run the model, the CPU kernels otherwise.
final model = await LocalModel.load(modelDir);
final chat = model.chat(
  system: 'Answer briefly.',
  tools: tools, // OpenAI-style function definitions
);
await chat.warmUp(); // the system prompt and tools, once

final turn = await chat.send(
  {'role': 'user', 'content': 'Hi!'},
  onText: (text) => true, // the reply so far; return false to stop
);
print(turn.reply.content);
print(turn.reply.toolCalls);
```

`send` throws `ContextLengthExceeded`, leaving the conversation as it was,
once a conversation outgrows `model.contextLength` (2048 tokens by
default); `chat.clear()` starts over. Conversations share the model's
state, and one generates at a time.

`await model.close()` gives back the model's GPU memory and its mappings
of the bundle's files; its conversations then throw `StateError`. The GPU
layer is process-wide, so keep every GPU model of a process in one isolate:
two isolates encoding GPU work at once corrupt each other's command
buffers, and Metal aborts the process.

`package:dianoia/dianoia.dart` is the API for apps. The engine's internals
(the forward pass, kernels and their tuning, quantization, file formats)
are in `package:dianoia/lab.dart`, which the command-line tool and the
benchmarks use and which may change without notice.

`example/example.dart` is a complete tool round trip, and `example/` is a
Flutter chat app that streams replies on an iPhone or Android phone (see
`example/README.md`).

## Development

```bash
dart format .
dart analyze
dart test
tool/compile_shaders.sh   # after editing src/vulkan/shaders (needs the NDK)
tool/lab_parity.sh        # bit-for-bit against dart-llm-lab (macOS)
```

| Path | Contents |
|---|---|
| `lib/src/` | The engine in Dart: model, chat session, tokenizer, template, tool calls, FFI bindings |
| `src/c/` | CPU kernels: Q8 and Q4 matrix products, attention, file mapping |
| `src/metal/` | The Metal layer and its shader source |
| `src/vulkan/` | The Vulkan layer, its GLSL shaders and their committed SPIR-V |
| `hook/build.dart` | Compiles the kernels for the target into the code asset |
| `bin/dianoia.dart` | The command-line tool |
| `tool/` | Shader compilation and the lab parity check |
| `tool/device_bench/` | The on-device benchmark app and its iPhone and Android round trips |
| `example/` | The Flutter chat app and a command-line example |

`tool/lab_parity.sh` is the safety net while the lab's code becomes a
library: it runs the lab's CLI and this one on the lab's reference prompts,
on the CPU and on the GPU with Q8 and Q4 weights, and requires every hidden
state, logit and token to match byte for byte.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `dianoia has no native kernels for ...` | The target is not arm64 macOS, iOS or Android. The quantized weights need the native kernels. |
| `LocalModel.backend` is `cpu` | No Metal device, or no Vulkan 1.2 device with buffer device addresses; `InferenceBackend.auto` fell back to the CPU kernels. `InferenceBackend.gpu` throws instead. |
| `--max-seq=N is too long for GPU attention` | GPU attention keeps one score per key in threadgroup memory. Lower `contextLength`. |
| Edited C or Metal code has no effect | Hook outputs are cached under `.dart_tool/`; delete `.dart_tool/hooks_runner` and run again. |
| `test/shaders_test.dart` fails | A shader changed without its SPIR-V. Run `tool/compile_shaders.sh`. |

## Origin

dianoia is the engine of dart-llm-lab, a research project that built an
LLM inference engine in Dart one measured step at a time (GEMV, SIMD,
quantization, the forward pass checked layer by layer against
`transformers`, native kernels, Metal, Vulkan, iPhone and Pixel). The
import is commit `e04e7c3` of that repository; its history, measurements
and reference tools stay there.

## License

MIT for this package's code. Model weights carry their own licenses.
