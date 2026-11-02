# dianoia

[![CI](https://github.com/Atrac613/dianoia/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Atrac613/dianoia/actions/workflows/ci.yml)

On-device text, tool calling and image chat in Dart. dianoia runs
**LFM3-1.7B** for text and tools, and **LFM3-VL-2B** for experimental image
conversations with streaming replies and visual follow-up questions.

Dart owns the tokenizer, chat template, graph and conversation state. Native
code provides quantized CPU kernels and the Metal/Vulkan language backend.
The vision encoder and projector currently run in the CPU reference
implementation, even when language inference uses the GPU.

`package:dianoia/dianoia.dart` is the app API and may change before 1.0.
`package:dianoia/lab.dart` exposes internals without a stability promise.
Experimental [LFM3-Audio Q4_0 transcription and speech synthesis](docs/lfm3_audio_support.md) is
available through research tools and `lab.dart`, with a CPU audio encoder and
CPU/Metal language inference. The Flutter example includes an experimental
audio preview using a locally prepared model, with microphone transcription
and text-to-speech playback. Speech synthesis also produces 24 kHz WAV files
through the research CLI. Audio is not yet part of the app API. General-purpose
GGUF support is planned.

## Try the Flutter example

After installing the [toolchain requirements](#requirements), run from a checkout:

```bash
cd example
flutter pub get
flutter devices
flutter run --profile -d <device-id>
```

The root screen offers these pinned models. Select one and tap **Download**;
weights are downloaded only after that action and run on the device afterwards.

| Model | Preset | Download | After setup |
|---|---|---:|---:|
| LFM3-1.7B, text and tools | `ModelSpec.lfm3Q4` | 1.07 GB | About 1.14 GB |
| LFM3-VL-2B, text and images | `VisionGgufSpec.lfm3VlQadQ4` | 1.45 GB | About 1.54 GB |

Sizes use decimal GB. Allow at least **5 GB free** for the VL download and
setup, including intermediate files. An interrupted download resumes when
**Download** is selected again. No weights are bundled with the package.

For VL, attach one PNG/JPEG and send it with a question or by itself. The app
shows image-analysis stages before the first reply text. Later text questions
retain visual context; **New chat** clears it. History keeps small thumbnails
rather than the full attachment bytes.

The example uses a **256-image-token budget** to reduce CPU encoder work
for camera photos. The library default is **1024**. Smaller budgets retain less
fine detail, including small text. Profile mode uses AOT compilation for physical
iPhone checks and avoids Debug JIT execution. See the
[example guide](example/README.md) for mirrors, sideloading and device details.

## What runs where

The figures below are recorded **LFM3-1.7B text** measurements; image decoding
and vision encoding are additional work and are not included.

| Target | CPU kernels | GPU | Measured (LFM3-1.7B, QAD Q4_0, GPU) |
|---|---|---|---|
| macOS, Apple silicon | NEON `sdot`, Grand Central Dispatch | Metal | M2 Max: ~280 tok/s prompt, ~90 tok/s decode |
| iOS, A14 or later | NEON `sdot`, Grand Central Dispatch | Metal | iPhone: ~80 tok/s prompt, 29.5 tok/s decode, 0.56 s to first token |
| Android, arm64 with dotprod | NEON `sdot`, OpenMP | Vulkan 1.2 (int8 dot products where present) | Pixel 8: ~82 tok/s prompt, ~18 tok/s sustained decode |
| Windows, x86-64 with AVX2 | AVX2, the Windows thread pool | Vulkan 1.2 (Resizable BAR on a discrete GPU) | Radeon RX 6600 XT: ~400-800 tok/s prompt, ~130 tok/s decode |
| Other targets (lab/CLI) | Dart reference paths | none | Public model loaders require native kernels |

Apps should check `LocalModel.isSupported` before offering model downloads.

The x86-64 kernels do NEON's float operations in NEON's order, so their
matrix products equal arm64's bit for bit. On its CPU alone (Core
i3-12100F, 4 threads) the Windows PC above prompts at ~26 tok/s and
decodes at ~22 tok/s.

Text state reuse is exact: a conversation that continues from a saved
checkpoint, in memory or in a state file, reproduces the logits of a run from
scratch bit for bit. Warming the system prompt and tool catalog avoids feeding
their tokens again. Recorded text workloads reached 0.3 s to the first token on
a Pixel 8 and 0.56 s on an iPhone; latency depends on the prompt and hardware.

VL conversations retain projected image features, but re-prefill the language
prompt on each turn. macOS CPU/Metal image inference and physical iPhone image
chat have been exercised; Android/Vulkan vision conversations remain unvalidated.
See the [image API guide](docs/vision_chat_api.md) and
[device benchmark report](docs/lfm3_vl_iphone_validation_2026-11-01.md).

## Requirements

- Dart SDK 3.10 or newer (build hooks). Flutter 3.38 or newer for apps.
- A C compiler for the target: Xcode for Apple platforms, the Android NDK
  for Android (Flutter supplies both through its toolchain), Visual Studio
  2022 or its Build Tools with the C++ workload (MSVC) for Windows. No Metal
  compiler, no `glslc` and no Vulkan SDK are needed: Metal shaders compile
  when the GPU starts, the Vulkan shaders ship as SPIR-V, and on Windows
  the Vulkan layer opens the driver's `vulkan-1.dll` at run time against
  headers in `src/vulkan/include`.
- Model weights, supplied separately. The built-in presets use
  [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) and
  [LiquidAI/LFM3-VL-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF)
  under the LFM Open License v1.0. Review the model license before distribution.
- Run model setup, loading and inference in a worker isolate in Flutter apps.
  Vision stages perform synchronous CPU work and yield between stages.

## Setup

```bash
dart pub get
dart test        # runs the build hook, then checks the kernels it built
```

The first `dart run`, `dart test` or `flutter run` compiles the native
kernels through `hook/build.dart` into one code asset,
`package:dianoia/kernels`, and caches it.

### Installing models

`ModelStore` downloads pinned files, verifies their byte lengths and SHA-256,
resumes partial downloads and prepares a runtime bundle in a background isolate.
Show the model name, download size and license before the user starts setup.
`spec.fromMirror(url)` changes the host/path while preserving the source pins.

For the example's VL preset, the installer downloads the official **QAD Q4_0
language GGUF** and its paired **BF16 vision/projector GGUF**, plus metadata.
It preserves the language Q4 blocks/scales and adapts the tied Q6_K
embedding/output head to Q8. All conversion runs in Dart/native code on the
device; Python and a full BF16 language checkpoint are not required.

The completed VL bundle is published atomically and source downloads are
removed. `isInstalled` checks its recorded source identity against the requested
QAD pins; `loadBundle` verifies every runtime payload before loading. Existing
QAD installs without the source record require explicit removal and reinstallation.

Other VL installation paths use the same store API:

| Input | API | Preparation |
|---|---|---|
| Official QAD Q4 GGUF pair | `VisionGgufSpec.lfm3VlQadQ4` | Import and pack; example default |
| Original BF16 checkpoint | `VisionCheckpointSpec.lfm3VlQ4` / `lfm3VlQ8` | Download 3.85 GB and quantize; allow about 13 GB free |
| Hosted portable bundle | `VisionModelSpec.fromManifest` | Verify files against an app-supplied trusted manifest |

QAD Q4 and BF16-derived PTQ Q4 have distinct model IDs and weights. See the
[installation guide](docs/vision_chat_api.md#install-the-official-qad-q4-ggufs),
[QAD reference comparison](docs/lfm3_vl_qad_q4_validation_2026-11-01.md) and
[PTQ Q4 comparison](docs/lfm3_vl_q4_validation_2026-11-01.md) for conversion details
and the limits of the small-fixture validation.

## Usage

### Image chat

Provide an application-owned model directory and a readable local PNG/JPEG.
For Flutter, run this work in the model's worker isolate after the user has
approved the download:

```dart
import 'package:dianoia/dianoia.dart';

Future<void> describeImage(String modelsDirectory, String imagePath) async {
  final store = ModelStore(modelsDirectory);
  final spec = VisionGgufSpec.lfm3VlQadQ4;
  if (!store.isInstalled(spec)) {
    await store.install(spec).drain<void>();
  }
  final model = await LocalVisionModel.loadBundle(
    store.bundlePath(spec),
    maxImageTokens: 256, // Same CPU-work budget as the Flutter example.
  );
  try {
    final chat = model.chat(maxNewTokens: 128);
    final image = await ChatImage.fromFile(imagePath);
    final turn = await chat.send(
      'Describe this image.',
      images: [image],
      onProgress: (progress) {
        print(progress.phase.name);
      },
      onText: (text) {
        print(text); // Cumulative reply, not a token delta.
        return true;
      },
    );
    print(turn.text);
    print((await chat.send('What details stand out?')).text);
    chat.clear(); // Releases visual features and conversation history.
  } finally {
    await model.close();
  }
}
```

`ChatImage.encoded(bytes)` accepts encoded PNG/JPEG bytes; `ChatImage.rgb` accepts
caller-decoded RGB pixels. Encoded input defaults to a 32 MiB limit and a
16,000,000-pixel decode limit, applies JPEG EXIF orientation and composites alpha
over white. HEIC, animation and video are not supported. `maxImageTokens` controls
later model resizing, independently of the decode limits.

`onProgress` reports new-image encoding stages, prompt prefill and generation.
Text-only follow-ups reuse earlier image features. Return `false` from `onText`
to keep a partial reply; `chat.cancel()` aborts a visual turn at a yield boundary
and rolls back its draft. Context overflow also leaves prior history intact.
See the [full image API guide](docs/vision_chat_api.md) for multiple images,
interleaved messages, cancellation and image-budget details.

### Text and tool calling

Install `ModelSpec.lfm3Q4` with the same store, then use `LocalModel` and
OpenAI-style function definitions:

```dart
import 'package:dianoia/dianoia.dart';

Future<void> textChat(
  String modelsDirectory,
  List<Map<String, Object?>> tools,
) async {
  final store = ModelStore(modelsDirectory);
  const spec = ModelSpec.lfm3Q4;
  if (!store.isInstalled(spec)) {
    await store.install(spec).drain<void>();
  }
  final model = await LocalModel.load(store.bundlePath(spec));
  try {
    final chat = model.chat(system: 'Answer briefly.', tools: tools);
    await chat.warmUp();
    final turn = await chat.send(
      {'role': 'user', 'content': 'Hi!'},
      onText: (text) => true,
    );
    print(turn.reply.content);
    print(turn.reply.toolCalls);
  } finally {
    await model.close();
  }
}
```

`send` throws `ContextLengthExceeded` once the prompt outgrows the model context
(2048 tokens by default), preserving the conversation. `chat.clear()` starts over.
The library parses tool calls; the app executes the tools and supplies results.

A large tool catalog need not fill the context. Mark tools with
`"defer_loading": true` and give the model a search tool of your own: the
deferred tools stay out of the system prompt, and the search tool's reply
brings in the ones it found, as `{"type": "tool_reference", "name": ...}`
items in the tool message's content. Keep every tool, deferred or not, in
`tools`, since a reference renders the definition from there.

`await model.close()` gives back the model's GPU memory and its mappings
of the bundle's files; its conversations then throw `StateError`. The GPU
layer is process-wide, so keep every GPU model of a process in one isolate:
two isolates encoding GPU work at once corrupt each other's command
buffers, and Metal aborts the process.

`package:dianoia/dianoia.dart` is the API for apps. The engine's internals
(the forward pass, kernels and their tuning, quantization, file formats)
are in `package:dianoia/lab.dart`, which the command-line tool and the
benchmarks use and which may change without notice.

`example/example.dart` demonstrates a complete tool round trip. The
[Flutter example](example/README.md) uses a dedicated worker and waits for its
shutdown before opening another model.

### Command line

For a locally imported text bundle:

```bash
dart run dianoia ask --model=path/to/LFM3-1.7B-QAD-Q4 --weights=q4 --gpu \
  --tools=example/weather_tools.json --prompt="Is it raining in Kyoto?"
```

Create that bundle from the official GGUF and matching local metadata:

```bash
dart run dianoia import-gguf --gguf=LFM3-1.7B-QAD-Q4_0.gguf \
  --model=path/to/LFM3-1.7B --out=path/to/LFM3-1.7B-QAD-Q4
```

For an installed portable VL bundle:

```bash
dart run dianoia vision-bench --bundle=path/to/vl-bundle --image=photo.jpg \
  --prompt="Describe this image." --backend=auto --max-seq=2048 \
  --max-image-tokens=256 --runs=1 --out=vision-result.json
```

| Commands | Purpose |
|---|---|
| `ask`, `toolbench`, `chatbench` | Text/tool chat and latency checks |
| `import-gguf`, `pack` | Import or package a text runtime bundle |
| `import-vl-backbone`, `pack-vl` | Prepare a BF16 VL backbone and portable Q8/Q4 bundle |
| `vision-run` | Raw-RGB vision inference and reference dumps |
| `vision-bench` | Verified load, image decoding, fresh/follow-up latency and process memory |
| `run`, `selftest` | Token-ID inference and tokenizer/template verification |

Command flags are documented at the top of [bin/dianoia.dart](bin/dianoia.dart).
The [VL integration notes](docs/lfm3_vl_support.md) cover local checkpoint
preparation; the [benchmark guide](docs/lfm3_vl_device_benchmark.md) covers desktop
and dedicated-device measurements.

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
| `lib/src/` | Dart model graph, text/image sessions, vision processing, tokenizer, tool calls and FFI bindings |
| `src/c/` | CPU kernels (NEON, and AVX2 on x86-64): Q8 and Q4 matrix products, attention, file mapping |
| `src/metal/` | The Metal layer and its shader source |
| `src/vulkan/` | The Vulkan layer, its GLSL shaders, their committed SPIR-V and the Vulkan headers for Windows |
| `hook/build.dart` | Compiles the kernels for the target into the code asset |
| `bin/dianoia.dart` | The command-line tool |
| `tool/` | Shader compilation, the lab parity check and the chat-template oracle |
| `tool/device_bench/` | The on-device benchmark app and its iPhone and Android round trips |
| `example/` | The Flutter chat app and a command-line example |

`tool/lab_parity.sh` is the safety net while the lab's code becomes a
library: it runs the lab's CLI and this one on the lab's reference prompts,
on the CPU and on the GPU with Q8 and Q4 weights, and requires every hidden
state, logit and token to match byte for byte.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `dianoia has no native kernels for ...` | The target is not arm64 macOS, iOS or Android, or x86-64 Windows on a CPU with AVX2, FMA and F16C (Haswell, Zen or later). The quantized weights need the native kernels. |
| `LocalModel.backend` is `cpu` | No Metal device, or no Vulkan 1.2 device with buffer device addresses; on Windows also no `vulkan-1.dll`, or a discrete GPU without Resizable BAR (the buffers live in the part of its memory the CPU can map). `InferenceBackend.auto` fell back to the CPU kernels. `InferenceBackend.gpu` throws instead. |
| `--max-seq=N is too long for GPU attention` | GPU attention keeps one score per key in threadgroup memory. Lower `contextLength`. |
| Edited C or Metal code has no effect | Hook outputs are cached under `.dart_tool/`; delete `.dart_tool/hooks_runner` and run again. |
| A camera photo takes a long time before the first reply | Vision encoding uses the CPU reference implementation. Use `onProgress` and a lower `maxImageTokens` budget (the example uses 256); finer image detail costs more work. |
| QAD model exists but `isInstalled` is false | Check that the requested model ID and source pins match. A legacy bundle without `qad_source.json` must be explicitly removed and reinstalled. |
| Physical iPhone Debug build exits with `SIGBUS` in JIT memory | Reproduce with `flutter run --profile` to separate AOT app behavior from Debug JIT/debugger failures; keep the app installed to preserve partial downloads. |
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
