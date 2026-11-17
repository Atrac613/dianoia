## Unreleased

- `LocalVisionModel` quantizes the LFM3-VL vision encoder's linear weights
  (BF16 in the checkpoint) to Q8 at load (`Lfm3VisionEncoder.load`'s
  `quantize`) and runs their products on the GPU with a GPU language model,
  otherwise in the native CPU kernels, instead of in Dart over float32.
  This changes the encoder's outputs: on the official QAD Q4 bundle three
  cards (Japanese and Latin OCR, a shapes photo) give the same answers as
  before on the CPU and the GPU. On an M2 Max a 640x480 image (320 prompt
  tokens) reaches its first token in 3.2-3.4 s instead of 19.5 s on the
  GPU and 10.6 s instead of 23.7 s on the CPU, and peak memory drops from
  1.1 to 0.6 GB (GPU) and 2.2 to 1.5 GB (CPU). Closing a vision model now
  frees its encoder's native and GPU memory.
- `Lfm3VisionEncoder.useGpu` moves the products of an encoder's native Q4
  or Q8 matrices (an mmproj GGUF's, as LFM3-Omni loads them) to the GPU
  through the new `GpuLinear`, keeping a second copy of those matrices in
  GPU memory; the rest of the encoder stays on the CPU. Outputs differ
  from the CPU kernels' in the last bits (relative L2 2.5e-6). The Omni
  tool's `--gpu` and the example's Omni chat use it when the language model
  is on the GPU. On an M2 Max a 300-token image encodes in 2.0 s instead of
  5.8 s, and an image question takes 3.9 s instead of 7.6 s.
- The vision encoder's attention runs in C on every native thread
  (`lab_attention_full`, `nativeFullAttention`) when the native kernels
  are loaded with their attention on, instead of in Dart. Its sums run in
  another order, so encoder outputs differ from the Dart path's in the
  last bits (relative L2 5e-6 on LFM3-Omni-2B). On an M2 Max with 4 threads
  a 640x480 image (300 image tokens) encodes in 5.8 s instead of 9.0 s.
- Faster decoding on Metal with the same bits. A one-token matrix product
  now gives each SIMD group two rows instead of one (`gemv_q8_r2`,
  `gemv_q4_r2`; `GpuForward.decodeRows` selects 1, 2 or 4 and
  `restoreDecodeRowsDefault` the backend's default), and the attention
  kernel loads eight keys per step in both of its loops over the keys.
  Every output equals the old kernels' bit for bit (`tool/lab_parity.sh`).
  On an M2 Max under other load, old and new builds run alternately on a
  431-token prompt: Q4 decode 36-46 to 56-65 tok/s, Q8 31-33 to 37-47
  tok/s; prompt processing is unchanged. Not yet measured on an iPhone.
- `GpuForward.profiling` now works on Metal: each dispatch runs in a
  command buffer of its own, so `takeProfile` reports GPU time per kernel
  kind (the wall clock is far slower meanwhile). `dianoia run --gpu
  --kernel-profile` prints the breakdown for the prompt and for decoding.
- Experimental control vectors (activation steering) under
  `package:dianoia/lab.dart`: `ControlVector` adds a direction to, or
  projects it out of, the residual stream leaving chosen layers, and
  `Lfm3Model.setControlVector` applies one on the CPU path and on the GPU
  (a new `steer` kernel for Metal and Vulkan; Vulkan is so far checked on
  its own kernel only, not on a device). Vectors load from safetensors or
  llama.cpp GGUF control-vector files and save as safetensors.
  `tool/control_vector_extract.dart` builds one from two contrasting prompt
  sets, reports how well each layer separates them (a leave-one-out d',
  `LayerSeparation`) and leaves out layers below `--min-separation`;
  `tool/control_vector_probe.dart` compares replies with and without it.
  `benchmarkControlVector` and `tool/control_vector_bench.dart` check a
  vector on a device's CPU and GPU, and the device bench app runs them on
  an iPhone (`tool/device_bench/run_ios_control_vector.sh`).
  `LocalModel` and `LocalVisionModel` take a vector between replies
  through lab-only extensions (`setControlVector`, `controlVector`);
  `package:dianoia/dianoia.dart` is unchanged.
  `tool/control_vector_eval.dart` (`ControlVectorEval`) measures one
  configuration on refused, benign and reference-answer sets: refusal
  openings, first-token KL, reply agreement and reference likelihood,
  reporting counts and likelihoods only.
  The example's text chat has an experimental control vector menu, fed
  from a `control_vectors` folder in the app's documents.
  See docs/control_vectors.md.

## 0.3.0

- Experimental LFM3-Omni-2B Q4_0 under `package:dianoia/lab.dart`. One turn
  can mix text, images, audio clips and a video with its soundtrack
  (`prepareOmniPrompt`, `Lfm3VideoProcessor`), answered in text, speech or
  interleaved text and speech (`decodeSpeechReply`). The LFM3-Audio
  transcription, speech and conversation APIs also run Omni.
  `OmniGgufSpec.lfm3OmniQ4` installs the official pair with `ModelStore`;
  `tool/prepare_omni_q4.dart` and `tool/omni_q4_chat.dart` are the research
  tools. LiveASR stays LFM3-Audio only.
- The Flutter example adds an Omni preview with an **Ask about a photo** mode,
  by voice or text, with an optional spoken reply, and follow-up questions
  that keep the photo and earlier answers in context. Its photo picker, shared
  with the VL chat, also accepts HEIC and HEIF photos and converts them to
  upright PNG. Speech playback now creates a missing temporary directory.
  Its **Text chat** button opens the same bundle in the text chat, with
  images on any message.
- The audio preview adds **Live translation (English)**: each sentence LiveASR
  finishes is translated as text (Japanese, Chinese, Korean, French, Spanish
  or German) by a text-only copy of the bundle's language backbone, running
  on the CPU in its own isolate beside the GPU recognition.
- `Lfm3Tokenizer.addedTokenIn` finds control tokens spelled out in text that
  must stay plain.
- `Lfm3SpeechVariant` carries each checkpoint's prompt conventions, and
  `Lfm3SpeechSampling` reads the speech sampling defaults from the mmproj.
  The generator's `sampleFrame` uses those defaults when an option is
  omitted, including Omni's 64-frame repetition window.
- `loadVisionEncoderGguf` loads the vision tower straight from an mmproj,
  keeping Q4_0 matrices native, and vision attention runs as batched matrix
  products: an Omni 640×480 image encodes in about 9.5 s instead of 72 s.
  The VL tower uses the same attention. `Lfm3VlConfig.fromOmniJson` parses
  Omni's configuration.

## 0.2.0

- Chat about images with LFM3-VL-2B. `LocalVisionModel` loads a VL
  bundle, and `VisionChatSession` streams replies, reports
  `VisionProgress` and can be cancelled (`VisionRequestCancelled`).
  `ChatImage.encoded` and `ChatImage.fromFile` decode PNG and JPEG (new
  dependency: `image`); `ChatImage.rgb` takes decoded pixels. Images are
  capped at `LocalVisionModel.defaultMaxImageTokens` (1024) unless
  `maxImageTokens` says otherwise. The vision encoder and projector run
  on the CPU; language inference still uses the GPU when it can.
- `ModelStore.install` takes any `InstallableModel`: `ModelSpec` as
  before, prebuilt VL bundles (`VisionModelSpec`), the official checkpoint
  built on device (`VisionCheckpointSpec.lfm3VlQ8`, `.lfm3VlQ4`), the
  paired official QAD GGUFs (`VisionGgufSpec.lfm3VlQadQ4`) and the
  LFM3-Audio Q4_0 bundle (`AudioGgufSpec.lfm3AudioQ4`). Downloads are
  verified before reuse, cancellation is polled while hashing, and a
  killed build no longer leaves staging behind.
- Experimental LFM3-Audio-2B Q4_0 under `package:dianoia/lab.dart`:
  clip and streaming transcription, English LiveASR that keeps recording
  past 30 seconds, speech synthesis with a streaming codec, and
  interleaved voice conversation. Audio is not yet part of the app API.
- LFM3 forward passes accept projected embeddings, and `lab.dart` exports
  the vision and audio internals, the GGUF readers and the vision
  benchmark. `GgufFile.close` is idempotent.
- The Flutter example selects a model, chats about images, and previews
  transcription, speech synthesis and voice conversation; it downloads
  the vision and audio models itself.
- Benchmark vision latency and memory on devices.
- CI builds and tests the native kernels on macOS and Windows, and the
  publish workflow runs CI before it publishes.

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
