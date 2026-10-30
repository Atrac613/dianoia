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
