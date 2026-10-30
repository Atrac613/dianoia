## 0.1.1

- The chat template renders LFM3's tool search: tools with
  `defer_loading` stay out of the `<tools>` block, and tool messages bring
  them in through `tool_reference` items, `result.tools` or the
  `additional_tools` field. Consecutive tool messages with one
  `tool_call_id` merge into one response, and tool content may be a
  mapping or a list, as in the model's `chat_template.jinja`.
- `tool/template_oracle.py` records Hugging Face's rendering of these
  cases, and `test/chat_template_test.dart` checks `renderChat` against it
  and against the lab's text oracle.

## 0.1.0

- Import the LFM3 engine from dart-llm-lab (commit e04e7c3): tokenizer,
  chat template, tool-call parser, the forward pass in Dart, Q8 and Q4
  native CPU kernels, and the Metal (Apple) and Vulkan (Android) GPU layers.
- Build the native kernels with a build hook (`hook/build.dart`) into one
  code asset, so `dart run`, `dart test` and Flutter apps get them without
  platform build files. Vulkan shaders ship as precompiled SPIR-V.
- Stream replies: `ChatSession.send(onText:)` sees the text after every
  token and can stop the reply; replies also stop at the model's context.
- Add a Flutter chat example (`example/`), tried on an iPhone 15 Pro Max
  and a Pixel 8; it also builds for macOS.
- Add the on-device benchmark (`tool/device_bench/`). On an iPhone 15 Pro
  Max and a Pixel 8 its results equal dart-llm-lab's.
- Link libm into the Android kernels.
- Give apps `LocalModel`: it finds the bundle's weights, loads the native
  kernels and falls back from the GPU to the CPU kernels on its own.
  `package:dianoia/dianoia.dart` now exports only `LocalModel`,
  `InferenceBackend`, `ChatSession`, `ChatTurn`, `ContextLengthExceeded`
  and the tool-call types; everything else moved to
  `package:dianoia/lab.dart`.
- `ChatSession.send` throws `ContextLengthExceeded` when the prompt leaves
  no room for a reply, and leaves the conversation unchanged when it fails.
- `LocalModel.close()` releases the model's GPU buffers (new
  `lab_gpu_release` in the Metal and Vulkan layers, which reuse freed
  handles) and unmaps the weight cache and `model.safetensors`, whose
  mappings are now reference-counted. A closed model throws `StateError`.
- Metal starts once per process: loading a second model no longer drops
  the first one's buffers.
- Download models: `ModelStore.install` fetches a `ModelSpec`'s files from
  where the model's authors publish them, pinned by revision, size and
  SHA-256, resumes interrupted downloads and builds the bundle on the
  device. `ModelSpec.lfm3Q4` is LFM3-1.7B's QAD Q4_0 build;
  `ModelSpec.fromMirror` fetches the same files from a mirror.
- The GGUF import builds the cache one matrix at a time (peak resident
  memory 2.2 GB to 1.38 GB, the same bytes).
- The chat example offers the download, with the model's size and
  license, and shows its progress.
- Run on Windows x86-64. The CPU kernels have AVX2 twins of the NEON ones
  that give the same matrix products bit for bit, threaded with the
  Windows thread pool (no OpenMP runtime to ship). The Vulkan layer builds
  with MSVC against the Khronos headers in `src/vulkan/include` and opens
  the driver's `vulkan-1.dll` at run time, so no Vulkan SDK is needed and
  the CPU kernels load without a driver. On a Core i3-12100F with a Radeon
  RX 6600 XT: ~22 tok/s decode on the CPU, ~130 tok/s on the GPU.
- On a discrete GPU the logits and captured hidden states live in the
  host's cached memory (`lab_gpu_readback_buffer`): reading the logits from
  the GPU's own memory across PCIe took 150 ms per token.
- Shader sources check out with LF line endings everywhere
  (`.gitattributes`), so `test/shaders_test.dart` passes on Windows.
