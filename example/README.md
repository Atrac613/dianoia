# dianoia examples

## Chat app (Flutter)

A simple chat with LFM3-1.7B running on the phone: replies stream in token
by token, a stop button ends a reply early, and "New chat" starts over
while keeping the system prompt's state. The model runs on the GPU (Metal
on iOS and macOS, Vulkan on Android), or on the CPU kernels when there is
no usable GPU, in a worker isolate so the UI never waits on it.

The model is not bundled. Make the Q4 bundle once with the dianoia CLI,
then copy it into the app's data:

```bash
dart run dianoia import-gguf --gguf=LFM3-1.7B-QAD-Q4_0.gguf \
  --model=path/to/LFM3-1.7B --out=path/to/LFM3-1.7B-QAD-Q4

cd example
flutter run --release                     # install and start once
tool/install_model.sh ios path/to/LFM3-1.7B-QAD-Q4        # or:
tool/install_model.sh android path/to/LFM3-1.7B-QAD-Q4
```

Then restart the app. On an iPhone the Files app or Finder can copy the
`LFM3-1.7B-QAD-Q4` folder into the app's documents instead. On Android the
app copies the pushed bundle into its internal storage on first start.

| Code | Role |
|---|---|
| `lib/src/chat_worker.dart` | The worker isolate: loads the model and answers through `ChatSession.send(onText:)` |
| `lib/src/chat_protocol.dart` | The messages between the UI and the worker |
| `lib/src/chat_controller.dart` | The UI-side state, a `ChangeNotifier` |
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
