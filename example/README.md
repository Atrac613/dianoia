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

The separate **LFM3-Audio-2B Q4_0** entry opens an experimental microphone
voice conversation, transcription and text-to-speech preview. It offers an in-app model download
and also accepts a prepared local model; see the setup below.

## Audio preview

Select **LFM3-Audio-2B Q4_0** on the root screen and tap **Download**. The
preview shows the pinned model's license and download/storage estimates before
starting. It downloads 1.23 GB from Hugging Face, verifies every source by
SHA-256 and prepares the Q4 language cache on device. Allow at least 4 GB free
during setup; about 1.30 GB remains installed. No Python or external conversion
tool is required in the app.

Download progress includes a cancel button. Leaving the route or backgrounding
the app also cancels installation. Downloaded/partial source files are kept for
the next attempt, which resumes them; conversion cancellation waits for the
current conversion to finish and does not publish a partial bundle. Successful
setup loads the model automatically and removes source downloads. Later visits
load the installed bundle without downloading again. Inference and installation
run in the existing audio worker.

`AudioGgufSpec.lfm3AudioQ4` uses `ModelStore` under the app's support `models`
directory (Local AppData cache on Windows). The installed bundle includes an
`audio_bundle.json` with pinned source identity and hashes/sizes of every
runtime payload. These hashes are verified before loading. Older prepared
research folders remain accepted through their original source receipts.

### Optional prepared model

Prepare the official Q4_0 pair from the repository root:

```sh
dart run tool/prepare_audio_q4.dart build/audio-reference
cd example
flutter run -d macos
```

Select **LFM3-Audio-2B Q4_0**, choose **Choose model folder**, and select the
prepared `build/audio-reference` directory. This needs the matching source
receipt, metadata, Q4 audio mmproj and imported `language` subdirectory.
The preview checks the pinned audio/config/tokenizer hashes and language-cache
identity in the worker before loading it. Choosing a folder does not start a
download or modify that folder.

For mobile sideloading, place the following prepared files under the
`LFM3-Audio-2B-Q4` folder shown by the preview, then tap **Check again**:

```text
LFM3-Audio-2B-Q4/
  audio_config.json
  mmproj-LFM3-Audio-2B-Q4_0.gguf
  language/
    config.json
    tokenizer.json
    model.safetensors
    lfm3-q4.cache
    audio_source.json
    LICENSE
```

The raw language GGUF is not required after preparation. Keep the imported
language files together; the cache and source receipt must match. iOS exposes
the app's Documents folder in Files/Finder. Android uses this app's external
files folder (`Android/data/dev.atrac613.dianoia_example/files`); Windows uses
the local app cache directory. Desktop users can choose an existing prepared
folder directly. See [audio implementation notes](../docs/lfm3_audio_support.md)
for source pins and storage requirements.

For voice conversation, select **Voice conversation (English)** under **Audio
mode**. Tap **Record question**, speak in English, then **Stop and reply**.
The model receives audio directly and generates interleaved text and speech;
it does not route the question through ASR followed by a separate TTS request.
Text appears during generation, and the complete speech plays afterwards.
Record another question to continue with the earlier turns as context.
**New conversation** clears that context. User turns show their duration;
they are not separately transcribed.

The model's voice chat currently supports English only. Japanese remains
available in the separate ASR/TTS mode. Recording and reply speech are limited
to 30 seconds each. The preview uses a 2048-slot language context; when it is
full, start a new conversation rather than silently dropping earlier turns.
The conversation stores audio embeddings and reply text in memory until reset
or route exit. It does not save microphone recordings or persist chat history.

**Cancel voice message** discards the current question; **Stop reply** cancels
generation or playback. Backgrounding or route exit also stops active work.
Cancelled, unaccepted replies never become hidden context. Once a reply is
accepted and visible, stopping its playback keeps that turn for the next
question. Returning to the app never starts recording or playback automatically.
This is a record/send/reply workflow, without simultaneous listening and talking
or streaming device playback. Existing prepared/downloaded Q4 models already
contain all required weights.

For recording-time text, select **Live transcription (English)**. Tap **Start
live transcription** and speak English. The transcript updates roughly once
per second of audio, plus processing time. **Stop live transcription** flushes
the final audio and completes the text. Updates are provisional until stop;
**Cancel recording**, backgrounding or route exit discards the current text.
A new recording starts a fresh live context. LiveASR is English only and uses
the official live-ASR protocol, not repeated full-clip transcription. Capture
still stops at 30 seconds, with the same queue/backpressure limit. No microphone
recording is saved or uploaded.

In **Transcription and TTS** mode, choose Japanese, English or automatic language detection. **Record** requests
microphone permission only when pressed; **Stop and transcribe** flushes the
final PCM and displays text. **Cancel recording**, route exit or backgrounding
discards the current recording. Returning to the app never resumes capture
automatically. Capture stops after 30 seconds. Recordings are not saved or
uploaded.

For TTS, enter a short sentence under **Text to speech**, select the output
language (Japanese/English) and voice instruction (Female/Male), then tap
**Generate and play**. The model generates the complete utterance before
playback; **Stop speech** cancels generation or stops playback. The input limit
is 1000 characters and output is limited to 30 seconds. Context overflow or
unfinished speech is reported as an error; try a shorter sentence. Recording
and TTS are mutually exclusive. Route exit/backgrounding stops speech, and
returning never restarts it.

Generation uses the same loaded language backbone plus lazily loaded
Depthformer/LiquidCodec weights in the worker. Playback uses
[audioplayers 6.8.1](https://pub.dev/packages/audioplayers/versions/6.8.1) with a
private temporary local WAV. The file is removed after completion, stop,
interruption or playback error. Generated speech stays on the device; the
preview does not upload it or provide permanent recording/export storage.

`record` supplies mono PCM16 at 16 kHz. The UI converts it to owned 80 ms float
chunks and permits only one inference request in flight. A one-second queue
limit stops/discards capture if the worker cannot keep up; samples are never
silently dropped. Models, hashing, encoding and incremental prefill live in a
worker isolate. Its lifetime shares a gate with text/VL workers so a model
switch waits for the previous isolate to release GPU resources and exit.

Platform setup follows the [record 7.1.2 guide](https://pub.dev/packages/record/versions/7.1.2):
iOS/macOS microphone usage descriptions, macOS audio-input sandbox entitlement,
and Android `RECORD_AUDIO`. The UI also rejects a changed capture configuration
that is not mono 16 kHz PCM16.

macOS compilation, simulated microphone/lifecycle tests and real Q4 worker
inference are validated. Physical microphone capture, mobile builds and device
audio-session behavior remain unvalidated. Speech playback is implemented but
has not been manually listened to or tested on a physical device. The preview
has no full-duplex voice mode or stable public audio session API. `AudioConversationSession` is available through `lab.dart`. The [TTS UI validation record](../docs/validation/lfm3-audio-example-tts-2026-11-02.md)
records automated checks; device playback and recording remain manual checks.
The [Audio download validation record](../docs/validation/lfm3-audio-download-2026-11-02.md)
covers installation, cancellation, Range resume and real-model worker loading.
The [voice conversation validation](../docs/validation/lfm3-audio-conversation-2026-11-03.md)
covers the native generation protocol and two-turn CPU/Metal examples.
The [LiveASR validation](../docs/validation/lfm3-audio-live-2026-11-03.md) records
recording-time updates, final flush and cancellation checks.

For a local download mirror, `DIANOIA_MODEL_MIRROR` also applies to Audio.
Serve the two pinned GGUFs plus `audio_config.json`, `tokenizer.json` and
`LICENSE` under those names, for example from `build/audio-reference`:

```sh
cd example
dart run tool/serve_model.dart ../build/audio-reference
# In another terminal:
flutter run -d macos --dart-define=DIANOIA_MODEL_MIRROR=http://127.0.0.1:8642
```

Mirrors change only URLs; all published sizes and hashes still apply.

## Text and image model setup

The text model is not bundled. Without one the app offers to download it,
naming its size (1.07 GB) and license; `ModelStore.install` then fetches
Liquid's published files from Hugging Face, checks each one's SHA-256,
builds the bundle on the device and loads it. A cancelled download resumes
where it stopped. On Windows the app keeps the model under
`%LOCALAPPDATA%\dev.atrac613\dianoia_example`, not in the roaming
profile.

For download, setup and inference checks on a physical iPhone, use an AOT
Profile build:

```bash
cd example
flutter run --profile -d <device-id>
```

If a Debug build crashes with `EXC_BAD_ACCESS` / `SIGBUS` in an anonymous
executable JIT mapping, reproduce in Profile mode before attributing it to the
installer. Updating the same app in place preserves partial downloads; select
the model and tap **Download** again to resume. Do not uninstall the app when
switching build modes if you want to keep that progress.

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

Camera photos are downscaled by the model processor to an example-specific
budget of 256 image tokens. The library default remains 1024; this smaller
budget reduces the work of the current CPU reference vision encoder, with
less fine detail available for small text. Before the first reply token, the
chat shows image reading, encoder-stage progress, and reply preparation. An
empty model response is reported as an error instead of leaving a spinner.

The VL chat's **Attach image** button opens the native file picker for one
PNG/JPEG (up to 32 MiB). The draft shows a preview and a remove button. Send it
with a question or by itself. Sent images appear in the conversation; later
text questions retain the visual context. New chat clears that context and
the draft. History retains only a PNG thumbnail of at most 240 by 240 pixels
and the file name, releasing the original attachment after sending. Thumbnail
decoding runs on the UI isolate; model image decoding and encoding run in the
inference worker. A stop
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
