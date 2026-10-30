## 0.1.0-dev.1

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
