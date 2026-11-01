# dianoia examples

## Chat app (Flutter)

A simple chat with LFM3-1.7B running on the device: replies stream in
token by token, a stop button ends a reply early, and "New chat" starts
over while keeping the system prompt's state. The model runs on the GPU
(Metal on iOS and macOS, Vulkan on Android and Windows), or on the CPU
kernels when there is no usable GPU, in a worker isolate so the UI never
waits on it.

The root screen lists the supported models with their download sizes and
licenses (LFM3-1.7B QAD Q4_0 and LFM3-VL-2B QAD Q4_0). Selecting a model opens its chat and
loads it if already installed. Back returns to model selection and releases
the chat worker; opening a model again starts a fresh conversation.

The text model is not bundled. Without one the app offers to download it,
naming its size (1.07 GB) and license; `ModelStore.install` then fetches
Liquid's published files from Hugging Face, checks each one's SHA-256,
builds the bundle on the device and loads it. A cancelled download resumes
where it stopped. On Windows the app keeps the model under
`%LOCALAPPDATA%\dev.atrac613\dianoia_example`, not in the roaming
profile.

For development there are two shortcuts around the 1 GB download:

```bash
cd example
# A local mirror: the same files, served from this machine. On Android,
# adb reverse makes it reachable from the phone.
dart run tool/serve_model.dart --port=8642 path/to/gguf-dir path/to/hf-dir
adb reverse tcp:8642 tcp:8642
flutter build apk --dart-define=DIANOIA_MODEL_MIRROR=http://127.0.0.1:8642

# Or a ready bundle (from `dianoia import-gguf`) copied into the app's data,
# which the app loads first.
tool/install_model.sh ios path/to/LFM3-1.7B-QAD-Q4  # or android, windows
```

`serve_model.dart --mbps=50` slows the mirror enough to try cancelling and
resuming by hand. Android release builds need the INTERNET permission,
which the app's main manifest declares.

| Code | Role |
|---|---|
| `lib/src/chat_worker.dart` | The worker isolate: installs and loads the model, and answers through `ChatSession.send(onText:)` |
| `lib/src/chat_protocol.dart` | The messages between the UI and the worker |
| `lib/src/chat_controller.dart` | The UI-side state, a `ChangeNotifier` |
| `lib/src/model_selection_page.dart` | The root model list and ownership of the selected chat worker |
| `lib/src/model_setup.dart` | The download offer and its progress |
| `lib/src/chat_page.dart`, `message_bubble.dart` | The screen |

Measured on the GPU while the reply streams onto the screen:

| Device | First token | Decode |
|---|---|---|
| iPhone 15 Pro Max | 309 ms | 25.9 tok/s (a 9-token reply) |
| Pixel 8 | 310 ms | about 12 tok/s |
| Radeon RX 6600 XT, Windows (debug build) | 49 ms | 91.7 tok/s (a 78-token reply) |

On a Pixel 8 the app also installed from Hugging Face itself: download, build and load finished within 11 minutes of the tap (not timed more closely), and the model answered. On the Windows PC the same install took 70 seconds. The macOS build has not been run yet. Replies are shown as plain text, so
the model's Markdown (`**bold**`) appears as written.

## LFM3-VL in the chat app

Select **LFM3-VL-2B (QAD Q4_0)** and tap **Download**.
`VisionGgufSpec.lfm3VlQadQ4` downloads the official language QAD Q4_0 GGUF
and paired BF16 vision/projector GGUF from LiquidAI/LFM3-VL-2B-GGUF revision
`b36554cfd35854b06429900131730cd80f092cdb`, plus the pinned original VL config,
processor, tokenizer and license. Every source file is checked by SHA-256.

Setup runs entirely in Dart/native code on the device. The Q4 language blocks
are copied unchanged; the existing GGUF adapter converts the Q6_K tied
embedding/head to Q8 for token lookup and output. The vision GGUF's BF16/F32
weights are preserved; its two F16 temporal patch slices widen exactly to F32.
The installer creates a standalone integrity-checked bundle marked
`input_embedding: q8-cache`, then removes downloads and intermediate files.
It does not require Python or download a full BF16 language checkpoint.

The download is about **1.45 GB**, the installed bundle about **1.54 GB**.
Allow at least **5 GB free** during first setup. Cancelled downloads resume;
leaving during preparation waits for it to finish before shutting down and
retains source downloads for the next attempt. QAD uses a separate model ID
and directory from PTQ Q4 and Q8 installations.

The VL chat's **Attach image** button opens the native file picker for one
PNG/JPEG (up to 32 MiB). The draft shows a preview and a remove button. Send it
with a question or by itself. Sent images appear in the conversation; later
text questions retain the visual context. New chat clears that context and
the draft. Image decoding and encoding run in the inference worker. A stop
before the first reply text cancels image processing; after text arrives,
Stop keeps the partial reply. Messages and images stay on the device.

For development, `DIANOIA_MODEL_MIRROR` can serve the two QAD GGUFs plus
`config.json`, `processor_config.json`, `tokenizer.json` and `LICENSE`. Pass the
GGUF and metadata directories to `tool/serve_model.dart`; the pinned sizes and
hashes still apply.

A prepared QAD bundle from `ModelStore` is also accepted: copy its complete contents
(including `vl_bundle.json` and `language/`) into `LFM3-VL-2B-QAD-Q4` under the
app's Documents on iOS/macOS, external files directory on Android, or app
cache directory under Local AppData on Windows. The loader verifies all files.
Android refreshes nested internal copies before loading.

To test an existing real bundle and a Japanese OCR card through the worker:

```sh
cd example
DIANOIA_VL_BUNDLE=/absolute/path/to/qad-bundle \
DIANOIA_VL_IMAGE=/absolute/path/to/ocr-ja.png \
flutter test test/vl_worker_integration_test.dart
```

The optional image test expects the QAD card answer `東京 123`. The
[QAD validation](../docs/lfm3_vl_qad_q4_validation_2026-11-01.md) records direct
GGUF reference responses and byte-identical logits before/after app packaging.

## Command line

`example.dart` is a tool round trip without Flutter:

```bash
dart run example/example.dart path/to/LFM3-1.7B-QAD-Q4
```
