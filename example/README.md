# dianoia examples

## Chat app (Flutter)

A simple chat with LFM3-1.7B running on the phone: replies stream in token
by token, a stop button ends a reply early, and "New chat" starts over
while keeping the system prompt's state. The model runs on the GPU (Metal
on iOS and macOS, Vulkan on Android), or on the CPU kernels when there is
no usable GPU, in a worker isolate so the UI never waits on it.

The model is not bundled. Without one the app offers to download it,
naming its size (1.07 GB) and license; `ModelStore.install` then fetches
Liquid's published files from Hugging Face, checks each one's SHA-256,
builds the bundle on the device and loads it. A cancelled download resumes
where it stopped.

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
tool/install_model.sh ios path/to/LFM3-1.7B-QAD-Q4        # or android
```

`serve_model.dart --mbps=50` slows the mirror enough to try cancelling and
resuming by hand. Android release builds need the INTERNET permission,
which the app's main manifest declares.

| Code | Role |
|---|---|
| `lib/src/chat_worker.dart` | The worker isolate: installs and loads the model, and answers through `ChatSession.send(onText:)` |
| `lib/src/chat_protocol.dart` | The messages between the UI and the worker |
| `lib/src/chat_controller.dart` | The UI-side state, a `ChangeNotifier` |
| `lib/src/model_setup.dart` | The download offer and its progress |
| `lib/src/chat_page.dart`, `message_bubble.dart` | The screen |

Measured on the GPU while the reply streams onto the screen:

| Device | First token | Decode |
|---|---|---|
| iPhone 15 Pro Max | 309 ms | 25.9 tok/s (a 9-token reply) |
| Pixel 8 | 310 ms | about 12 tok/s |

The macOS build has not been run yet. Replies are shown as plain text, so
the model's Markdown (`**bold**`) appears as written.

## Command line

`example.dart` is a tool round trip without Flutter:

```bash
dart run example/example.dart path/to/LFM3-1.7B-QAD-Q4
```
