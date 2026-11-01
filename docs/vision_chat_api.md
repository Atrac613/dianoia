# Experimental image chat API

Apps can import `package:dianoia/dianoia.dart` and use `LocalVisionModel`,
`ChatImage` and `VisionChatSession`. The API streams answers, supports multiple
images and text-only follow-ups, and retains projected image features within
each conversation. `ChatTurn` is the same result type used by text chat.

The language backbone can run on CPU, Metal or Vulkan where the existing engine
supports it. The vision encoder/projector currently uses the float32 CPU
reference implementation. Public-API OCR and follow-up tests have run on macOS
CPU and Metal. Initial physical iPhone timing/memory results are recorded in
the [device report](lfm3_vl_iphone_validation_2026-11-01.md); Android/Vulkan vision
conversations remain unvalidated. Run loading and inference in a single worker isolate: each vision
layer performs synchronous CPU work, yielding between layers for cancellation.

## Prepare local files

Keep the original LFM3-VL checkpoint, including `config.json`, `tokenizer.json`,
`processor_config.json`, `LICENSE` and `model.safetensors`. Import a matching language
backbone using `dianoia import-vl-backbone`. Build its Q8 cache with `vision-run`
as described in the [real-checkpoint report](lfm3_vl_validation_2026-11-01.md).
`LocalVisionModel.load` uses an existing Q4 cache if present, otherwise Q8, matching
`LocalModel` bundle selection. Real Q4 and Q8 desktop results are compared in
[the Q4 validation report](lfm3_vl_q4_validation_2026-11-01.md).

The loader validates source/backbone import fingerprints, processor dimensions
and vision special-token IDs. As with existing caches, fingerprints cover the
safetensors header/size and config; verify downloaded payload hashes separately.
`ModelStore` installs text checkpoints, upstream VL checkpoints and prebuilt VL bundles. Loading itself
does not download weights or prepare a backbone. Use the image adapters below
to decode local PNG/JPEG inputs.

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
Q4 packing requires an existing matching Q4 cache. Real-model packing preserves
all OCR logits on CPU and Metal for both formats. The
[Q4 comparison](lfm3_vl_q4_validation_2026-11-01.md) records its size/speed gains
and Japanese OCR whitespace difference. PTQ Q4 and Q8 remain available through
`VisionCheckpointSpec`; the example selects the official QAD Q4 preset.
A separate [official QAD Q4 evaluation](lfm3_vl_qad_q4_validation_2026-11-01.md)
checks the GGUF import against llama.cpp and the installed app bundle.

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
without changing its layout. `ModelStore` can install this layout from a host
you choose, using the pinned manifest as described below.

The measured Q8 bundle is 2,885,204,285 bytes (2.69 GiB), compared with
9,275,569,800 bytes (8.64 GiB) for the original checkpoint plus imported backbone
and Q8 cache. This saves 68.9% of disk space in that preparation layout; it does
not reduce the vision encoder's float32 working memory. See the
[bundle validation report](lfm3_vl_bundle_2026-11-01.md).

## Install the official QAD Q4 GGUFs

The example's default downloads paired official QAD language and BF16 mmproj
GGUFs, with pinned config/tokenizer/processor/license files:

```dart
final spec = VisionGgufSpec.lfm3VlQadQ4;
final store = ModelStore(modelsDirectory);
await store.install(spec).drain<void>();
final model = await LocalVisionModel.loadBundle(store.bundlePath(spec));
```

The 1.45 GB source download produces an approximately 1.54 GB bundle; allow
5 GB free during setup. Language Q4 blocks and scales remain unchanged. The
Q6_K embedding/output head uses the existing Q8 adapter. Both GGUFs are pinned
at revision `b36554cfd35854b06429900131730cd80f092cdb`; metadata retains the
original VL pin. `fromMirror` preserves all hashes. Setup runs in an isolate,
cleans intermediates on failure/cancellation, and publishes the bundle atomically.
QAD has its own ID, separate from BF16-derived Q4 and Q8. A successful install
writes `qad_source.json`, recording all six source roles, names, byte lengths and
pinned SHA-256 values, together with the generated bundle manifest's hash.
`isInstalled` requires that record to match the requested spec; changing only
the mirror URL preserves identity. Older installs without this record must be
removed with `store.remove(spec)` and reinstalled. Payload hashes are still
verified by `loadBundle`. Catalog `installedBytes` and `setupBytes` provide
optional numeric storage estimates; custom specs can omit them.

The version-one manifest's optional `input_embedding` field defaults to `bf16`
for existing bundles. QAD bundles explicitly select `q8-cache`: the BF16 table
must be absent and the cache must contain a complete Q8 embedding/head. Both
policies are validated. The lab packing API accepts `useCacheEmbedding: true`
for such GGUF imports; `ModelStore` selects it automatically.

## Install the official VL checkpoint

`ModelStore` can download the pinned upstream BF16 checkpoint and build a Q4 or Q8
bundle on the device, without an app-hosted converted model:

```dart
final spec = VisionCheckpointSpec.lfm3VlQ4; // Or lfm3VlQ8.
final store = ModelStore(modelsDirectory);
await store.install(spec).drain<void>();
final model = await LocalVisionModel.loadBundle(store.bundlePath(spec));
```

This catalog entry pins LiquidAI/LFM3-VL-2B revision
`ba44bd51db20a2753f28c273a1571ccaa210d954` and all five source files' SHA-256
hashes. Both presets download 3.85 GB; Q4 installs about 2.06 GB and Q8 about
2.89 GB. Their model IDs differ, so an existing Q8 bundle cannot be reused as Q4.
Allow at least 13 GB of free storage for the checkpoint, temporary imported
backbone and packed output together. Quantization writes one matrix at a time and releases its native allocation
after writing it, including on write failure. The Q8 head is quantized from
bounded row chunks. Quantization and packing use separate isolates, without
loading a complete inference model to build the cache.
Downloads resume and are checked again before reuse. Cancellation during
conversion waits for completion, discards unpublished output and retains the
downloads. Failed builds remove temporary output; retry rebuilds it.
`fromMirror` changes the source directory while preserving every pinned hash.
This does not change the trusted-manifest API for prebuilt hosted bundles below.

## Install a hosted VL bundle

Host the whole `pack-vl` directory under an HTTP(S) base URL with its relative
paths intact. Include its exact `vl_bundle.json` in a trusted app asset or release
configuration; pass those reviewed bytes as `manifestJson`. The store pins the
remote manifest itself as well as every payload's size and SHA-256. Do not obtain
the expected manifest through an unverified fetch from the payload host.

```dart
Future<LocalVisionModel> installVision({
  required ModelStore store,
  required String reviewedManifestJson,
  required String distributionUrl,
}) async {
  final spec = VisionModelSpec.fromManifest(
    id: 'lfm3-vl-2b-q8',
    name: 'LFM3-VL-2B (Q8)',
    license: 'LFM Open License v1.0',
    licenseUrl: '$distributionUrl/LICENSE',
    baseUrl: distributionUrl,
    manifestJson: reviewedManifestJson,
  );
  // Show spec.name, spec.license and spec.downloadBytes before downloading.
  await for (final progress in store.install(spec)) {
    print('${progress.stage.name}: ${progress.bytes}/${progress.totalBytes}');
  }
  return LocalVisionModel.loadBundle(store.bundlePath(spec));
}
```

Use `spec.fromMirror(baseUrl)` to change the host without changing pinned bytes.
The manifest text, including whitespace, must match the hosted file exactly.
`downloadBytes` includes the manifest and all payload files. No built-in hosted
VL catalog entry is supplied; applications choose their own distribution URL.

`ModelStore.install` shares the text installer's resumable downloads, retry policy,
progress stream and per-model exclusion. Cancelling its subscription waits for
file closure and retains downloads; a later install resumes them. Completed
files are also checked against their pinned size and hash before reuse. The
`building` stage validates the complete VL bundle in a worker isolate, then
renames the downloads directory into the installed path. This avoids another
bundle-sized copy and does not quantize. Cancelling during verification waits
for that verification and leaves the downloads unpublished.

`store.isInstalled(spec)` checks directory presence and the pinned VL manifest,
not every tensor payload; `loadBundle` checks the full payload on each load.
Use a distinct model ID for a new release. A different manifest cannot overwrite
an installed VL model under the same ID: close any loaded model, then explicitly
`await store.remove(spec)` before installing its replacement. Removal is rejected
while that model is being installed. IDs must be single directory names starting
with a letter or digit, using only letters, digits, dots, underscores or hyphens.

Synthetic HTTP tests cover successful loading, nested Range resume, retry after
an interrupted response, cancellation during download/verification, changed
payloads/manifests, stale completed downloads and inconsistent provenance.
An optional real integration test installs the Q8 bundle over loopback HTTP and
reads `東京 123` through the installed public API on CPU and Metal:

```sh
DIANOIA_VL_BUNDLE="$PWD/build/vl-validation/portable-q8" \
DIANOIA_VL_VALIDATION_DIR="$PWD/build/vl-validation/extended" \
dart test test/vision_model_store_integration_test.dart
```

This test temporarily needs another bundle-sized directory. It uses local files
and does not publish or fetch external weights. The 2026-11-01 M2 Max run
successfully installed 2,885,204,285 bytes and read `東京 123` on CPU and Metal;
see the [installation record](validation/lfm3-vl-store-2026-11-01.json).

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

Supply packed RGB uint8 data with exactly `width * height * 3` bytes to
`ChatImage.rgb`; it copies pixels when constructed. For caller-decoded images,
convert to RGB and apply orientation before construction. Image lists, messages and tool
schemas are copied before asynchronous work can observe caller mutations.
The API never fetches URLs or file paths in content parts.

## PNG/JPEG input

Use encoded bytes or a local file directly:

```dart
final fromBytes = ChatImage.encoded(pngOrJpegBytes);
final fromFile = await ChatImage.fromFile('/path/to/photo.jpg');
final turn = await chat.send('画像を説明してください。', images: [fromFile]);
```

The adapter detects PNG/JPEG by their contents and returns owned RGB8 pixels.
JPEG EXIF orientations 1–8 (including reflections) are applied exactly once;
`width` and `height` describe the oriented image. PNGs use their stored pixel
orientation; PNG EXIF metadata is not applied. Gray, palette, RGB and alpha PNGs
are converted to RGB, including 16-bit grayscale. Alpha is composited over white
in the encoded channel space, with nearest-integer RGB8 rounding. ICC color
profile conversion is not performed. Use `ChatImage.rgb` for a platform decoder's
color-managed pixels or other file formats.

By default, both adapters accept at most **32 MiB encoded bytes** and
**16,000,000 decoded pixels**. Limits must be positive and can be configured:

```dart
final image = await ChatImage.fromFile(
  '/path/to/photo.jpg',
  maxEncodedBytes: 16 * 1024 * 1024,
  maxPixels: 8000000,
);
```

The byte limit is checked before decoding; file reads are bounded even if the
file grows after the initial size check. Pixel limits are passed to both codecs
before raster allocation, with header and output checks. Unsupported, malformed,
animated or over-budget input raises `FormatException`; invalid limits raise
`ArgumentError`. Local file I/O errors propagate. Animated PNG (including a
one-frame APNG) is rejected. No network URLs are fetched.

`encoded` decodes synchronously. `fromFile` awaits file I/O, then decodes on the
calling isolate. Run these in the same worker isolate as the model. Decoding is
not part of an active chat request and is not interrupted by `chat.cancel()`.
Both adapters decode at the original resolution; the model's processor resizes
later, within the [image token budget](#image-token-budget). A `ChatImage` is
created before the conversation's context-budget check, so its decode budgets
apply independently of token limits.
The implementation uses [image](https://pub.dev/packages/image) `^4.12.0`, whose
PNG/JPEG decoders expose per-instance pixel limits.

Tests use independently generated Pillow fixtures for PNG channel conversion,
all eight JPEG EXIF orientations, unsupported animation, bad checksums, large
headers and byte budgets. A file-growth test checks the read limit; a synthetic
chat test requires PNG input and manually constructed RGB to produce identical
logits. Optional real-model tests in `vision_public_model_test.dart` compare all
RGB/PNG logit bytes and read the same Japanese OCR card from an oriented JPEG
on CPU and Metal. Both backends passed on 2026-11-01; see the
[image-input validation record](validation/lfm3-vl-image-input-2026-11-01.json).
Enable `DIANOIA_VL_BUNDLE` and `DIANOIA_VL_VALIDATION_DIR`.

## Image token budget

The checkpoint's processor allows 16,777,216 pixels per image, 16,384 image
tokens: a 4032×3024 photo keeps 3008×4032 pixels and becomes 11,844 tokens,
more than an on-device context holds. `LocalVisionModel.load` and `loadBundle`
therefore lower the budget to `maxImageTokens`, by default 1024
(`LocalVisionModel.defaultMaxImageTokens`):

```dart
final model = await LocalVisionModel.loadBundle(
  '/path/to/portable-vl',
  contextLength: 2048,
  maxImageTokens: 512,
);
print(model.maxImageTokens); // 512
```

Larger images are downscaled with the upstream resize rule, as the upstream
processor does with `longest_edge` set to `maxImageTokens` × 32 × 32 pixels. With
the default, the photo above becomes 864×1152 pixels and 972 tokens. Images
within the budget are processed exactly as before, so the recorded validation
images are unaffected. A request above the checkpoint's own budget leaves that
budget in place; `maxImageTokens` reports the budget in effect. A request below
the checkpoint's minimum (64 tokens) throws `ArgumentError` before native model
resources are allocated. As upstream, a side that would shrink below 32 pixels
is kept at 32, so a very elongated image can exceed the budget; the context
check still applies to it.

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
