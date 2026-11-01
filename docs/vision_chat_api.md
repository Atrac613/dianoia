# Experimental image chat API

Apps can import `package:dianoia/dianoia.dart` and use `LocalVisionModel`,
`ChatImage` and `VisionChatSession`. The API streams answers, supports multiple
images and text-only follow-ups, and retains projected image features within
each conversation. `ChatTurn` is the same result type used by text chat.

The language backbone can run on CPU, Metal or Vulkan where the existing engine
supports it. The vision encoder/projector currently uses the float32 CPU
reference implementation. Public-API OCR and follow-up tests have run on macOS
CPU and Metal. Phone memory/latency and Vulkan vision conversations are not yet
validated. Run loading and inference in a single worker isolate: each vision
layer performs synchronous CPU work, yielding between layers for cancellation.

## Prepare local files

Keep the original LFM3-VL checkpoint, including `config.json`, `tokenizer.json`,
`processor_config.json`, `LICENSE` and `model.safetensors`. Import a matching language
backbone using `dianoia import-vl-backbone`. Build its Q8 cache with `vision-run`
as described in the [real-checkpoint report](lfm3_vl_validation_2026-11-01.md).
`LocalVisionModel.load` uses an existing Q4 cache if present, otherwise Q8, matching
`LocalModel` bundle selection. The recorded real VL runs use Q8.

The loader validates source/backbone import fingerprints, processor dimensions
and vision special-token IDs. As with existing caches, fingerprints cover the
safetensors header/size and config; verify downloaded payload hashes separately.
`ModelStore` currently installs text checkpoints only. Loading does not download
weights, prepare a backbone or decode image files.

## Build a standalone bundle

After creating the language cache, package only the runtime tensors:

```sh
dianoia pack-vl --model=/path/to/LFM3-VL-2B \
  --backbone=/path/to/imported-backbone --out=/path/to/portable-vl --weights=q8
```

The output must not exist. Packing copies the original vision/projector tensors,
BF16 token embeddings and small language tensors, plus the existing Q8 or Q4
matrix cache. It does not requantize. It includes configuration, tokenizer and
upstream license, writes a SHA-256 manifest, and publishes the directory only
after all files are complete. The original directories are unnecessary at runtime.
Q4 packing requires an existing matching Q4 cache; real-model bundle validation
currently covers Q8, with Q4 covered by synthetic runtime tests.

Load the resulting directory with the app-facing API:

```dart
final model = await LocalVisionModel.loadBundle(
  '/path/to/portable-vl',
  backend: InferenceBackend.auto,
  contextLength: 2048,
);
```

Loading verifies the size and SHA-256 of every required file before native
loading. This reads the entire bundle on each load; the manifest must come from
a trusted installation source. The checks detect changed files and incomplete
copies, and do not authenticate the manifest. Move/copy the whole directory
without changing its layout. `ModelStore` does not yet download VL bundles.

The measured Q8 bundle is 2,885,204,285 bytes (2.69 GiB), compared with
9,275,569,800 bytes (8.64 GiB) for the original checkpoint plus imported backbone
and Q8 cache. This saves 68.9% of disk space in that preparation layout; it does
not reduce the vision encoder's float32 working memory. See the
[bundle validation report](lfm3_vl_bundle_2026-11-01.md).

## Send an image and follow up

```dart
import 'dart:typed_data';
import 'package:dianoia/dianoia.dart';

Future<void> askImage(Uint8List rgb, int width, int height) async {
  final model = await LocalVisionModel.load(
    '/path/to/LFM3-VL-2B',
    backboneDir: '/path/to/LFM3-VL-2B-backbone',
    backend: InferenceBackend.auto,
    contextLength: 2048,
  );
  try {
    final chat = model.chat(maxNewTokens: 128);
    final image = ChatImage.rgb(rgb, width: width, height: height);
    final turn = await chat.send(
      '画像を日本語で説明してください。',
      images: [image],
      onText: (text) {
        print(text); // Cumulative text, with incomplete trailing UTF-8 hidden.
        return true;
      },
    );
    print(turn.reply.content);
    final detail = await chat.send('文字があれば、そのまま書いてください。');
    print(detail.text);
  } finally {
    await model.close();
  }
}
```

Supply packed RGB uint8 data with exactly `width * height * 3` bytes. The caller
must decode PNG/JPEG, convert color channels and apply orientation first.
`ChatImage.rgb` copies pixels when constructed. Image lists, messages and tool
schemas are copied before asynchronous work can observe caller mutations.
The API never fetches URLs or file paths in content parts.

## Message layout and tools

`send(text, images: ...)` inserts image parts before the text. For interleaved
images/text or tool results, use `sendMessage` with Hugging Face-shaped content:

```dart
await chat.sendMessage({
  'role': 'user',
  'content': [
    {'type': 'text', 'text': 'First image: '},
    {'type': 'image'},
    {'type': 'text', 'text': 'Second image: '},
    {'type': 'image'},
  ],
}, images: [firstImage, secondImage]);
```

Images bind to placeholders in message/content order. The message must be a
`user` or `tool` message; assistant turns are recorded from generated output.
Offer tool definitions with `model.chat(tools: ...)`, inspect `turn.reply.toolCalls`
and return tool results with `sendMessage({'role': 'tool', ...})`. This API does
not execute tools. Video and unbound or malformed image placeholders are rejected.
`messages` provides deeply read-only message history with image placeholders;
RGB bytes and visual features are kept private.

## Cancellation, failures and lifetime

- Returning `false` from `onText` stops gracefully and commits the partial reply.
- Calling `chat.cancel()` aborts that session's active turn. The send future
  throws `VisionRequestCancelled`; its draft and partial streamed output are
  not committed. Prior history remains intact, and the session can be reused.
- Cancellation is checked between vision stages, prefill and decode steps. It
  also interrupts waiting on an asynchronous `onText`. User callback futures
  already started can still finish after cancellation.
- Context overflow is checked from image dimensions and token counts before
  pixel preprocessing/encoding. It throws `ContextLengthExceeded` without
  appending a message or allocating expanded image-patch tensors.
- Inference or callback errors roll back the draft. The next turn starts by
  resetting language state. Only one visual request per loaded model may run;
  another session's request fails with `StateError` and cannot cancel the owner.
- `chat.clear()` releases its retained image features and history. Calling it
  during an active send throws; cancel and await the send first.
- `model.close()` requests cancellation, waits for active work to settle,
  drops vision weights and closes language resources. It is idempotent. New
  sends and new sessions are rejected once closing starts.

Each successful visual turn re-prefills the language prompt from reset; image
features are reused, but `ChatTurn.reusedTokens` is currently zero.
`prefillSeconds` includes any new image encoding. `model.backend` describes the
language backend; it does not imply GPU image encoding. `onLogits` is a
synchronous observer of a temporary model view: copy logits before retaining
them across model calls.

Synthetic runtime tests cover ownership, multiple images, follow-ups, rollback,
context overflow, cancellation, callback failures, overlapping sessions and close
from an asynchronous callback. Optional real-checkpoint tests use only this
public API to read `東京 123`, stream complete UTF-8 and answer a text-only visual
follow-up on CPU and Metal. See `test/vision_public_model_test.dart`; enable with
`DIANOIA_VL_DIR`, `DIANOIA_VL_BACKBONE` and `DIANOIA_VL_VALIDATION_DIR` (the extended
validation directory containing `reference-ocr-ja/input.rgb`). No test downloads
weights.

Set `DIANOIA_VL_BUNDLE` as well to compare every OCR logit byte between the
original and standalone layouts on both CPU and Metal.
