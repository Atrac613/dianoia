# LFM3-VL-2B integration

Status: experimental lab support. Real-checkpoint RGB-to-answer smoke checks
pass for CPU float32 and a CPU vision encoder with Q8/Metal language inference.
An experimental [app-facing image chat API](vision_chat_api.md) is available.
Optimized vision inference is not implemented.

## Verified upstream structure

Sources inspected on 2026-10-31:

- https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/main/config.json
- https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/main/modeling_lfm3_vl.py

The checkpoint has model type `lfm3_vl` and a nested `text_config` whose model
is LFM3 (2048 hidden dimensions, 24 layers). Its vision encoder has 14 layers,
1024 hidden dimensions, 16x16 patches, gated depthwise 2D convolutions and
bidirectional attention with axial RoPE. These are different operators from
the causal language backbone.

The projector applies LayerNorm over 1024 dimensions, merges four spatial
patches, applies a biased 4096-to-4096 linear layer, GELU, and a biased
4096-to-2048 linear layer. The resulting rows replace image-token embeddings.
Image token ID is 125030, video token ID is 125029; vision delimiters are
125009 and 125010. Use VL checkpoint language weights, not the text-chat
checkpoint's weights.

## Implemented

The lab API `Lfm3Model.forward(embeddingOverrides: ...)` accepts projected
features at positions relative to the submitted token list. It supports CPU,
Metal and Vulkan through the existing residual input buffers. All dimensions,
positions and finite values are checked before inference; input rows are copied
before asynchronous execution.

Overridden positions are stored as -1 in token history, including persisted
states. Text-only prefix reuse therefore stops before visual input. Multimodal
callers must reset before submitting a new visual prompt until feature-aware
prefix reuse is implemented.

The lab API `Lfm3VlConfig.load` now parses the nested language configuration,
vision encoder dimensions and layer types, projector parameters and special
vision token IDs. It validates axial RoPE dimensions, spatial merging and
supported projector options. `languageTensorName` maps engine language tensor
names to the VL checkpoint namespace. The regular text loader still rejects
VL checkpoints. The official config metadata is covered by a checked-in
fixture; no weights are included.

## Importing the language backbone

Download the full BF16 checkpoint (single `model.safetensors`, `config.json`,
`tokenizer.json`) to a local directory, then use a new output directory:

```sh
dart run dianoia import-vl-backbone --model=/path/to/LFM3-VL-2B \
  --out=/path/to/LFM3-VL-2B-backbone
```

The importer validates every required language tensor's name, shape, dtype and
byte range, then writes the original tensor bytes under engine language names.
It copies only the language backbone, tokenizer and nested text config, retaining
the original config as `vl_config.json`. It also writes `vl_backbone.json` with
source and output cache fingerprints for the validation command. Vision weights remain in the source.
Output is published after the copy completes; existing output is rejected.
Copying uses a 1 MiB buffer rather than materializing large tensors in memory.
The output is a full-weight checkpoint, not a quantized device bundle: existing
`run`, `pack` and Q8/Q4 cache paths can process it. Use token-ID prompts for
backbone comparisons; the text-only chat renderer is not a VL chat template.

Tests compare imported BF16 bytes and CPU logits against a canonical miniature
checkpoint containing all three language layer types, and check truncated or
incomplete input rejection. Actual checkpoint metadata was inspected at revision
`ba44bd51db20a2753f28c273a1571ccaa210d954`; real VL-weight inference results are recorded in
[the validation report](lfm3_vl_validation_2026-11-01.md). Only the official single-file checkpoint layout is supported.

## RGB image preprocessing

The lab API `Lfm3ImageProcessor.load(modelDir)` reads `processor_config.json`.
`prepareRgb(rgb, height: ..., width: ...)` accepts packed RGB uint8 pixels and
returns float32 `pixelValues`, `gridThw`, dimensions and the image-token count.
It implements aspect-ratio-preserving resize to patch/merge multiples,
Python's ties-to-even dimension rounding, antialiased uint8 bicubic resizing,
fused normalization and spatially grouped patches with duplicated temporal
samples. Pixel budgets are areas, despite the upstream `shortest_edge` and
`longest_edge` field names. `withMaxImageTokens(n)` lowers the budget to `n`
merged tokens (`n` × (patch × merge)² pixels) without changing the resize rule;
`LocalVisionModel` applies it with a default of 1024 tokens.

The caller must decode image files, convert to RGB and apply orientation before
calling this API. PNG/JPEG decoding, video and encoder execution are not
implemented here. Unsupported processor settings fail explicitly.

`tool/vision_image_oracle.py` regenerates compact reference hashes using the
pinned upstream `smart_resize` and `patchify` functions and torchvision resizing
and normalization. Install torch and torchvision in a separate Python environment,
then run the script from the repository root. The fixture records source hash
and runtime versions. Nine cases cover no resize, enlargement, reduction,
single-axis resizing, singleton input, ties-to-even rounding and the official
16-pixel patch configuration. All resulting float32 patch bytes match the
reference hashes, also when each case's budget is reached by capping a
four-times-larger one with `withMaxImageTokens`.

## CPU reference encoder and projector

`Lfm3VisionEncoder.load(modelDir)` reads the vision/projector tensors from the
original VL checkpoint, validating required shapes through the safetensors
reader. `encode(image)` consumes a `VisionImageInput` from the RGB processor
and returns flattened `[imageTokens, textHiddenSize]` projected embeddings.
Optional capture callbacks receive independent snapshots after embedding norm,
each vision layer and the projector.

The synchronous float32 CPU implementation includes patch projection, RMSNorm,
gated depthwise 2D convolution, bidirectional grouped-query attention with axial
2D RoPE, SwiGLU feed-forward layers and the biased LayerNorm/GELU projector.
It uses existing Dart matrix kernels and one score row at a time for attention.
GELU uses the erf formula with an erf approximation (absolute error about 1.5e-7).
It supports one still image per call; encoding images separately isolates their
attention. Weights are widened to float32 on the Dart heap. This path establishes
correctness and has not been optimized or benchmarked for phones.

`tool/vision_encoder_oracle.py` runs the pinned upstream PyTorch operator classes
with reproducible synthetic parameters. Tests generate those parameters at
runtime; no model weights are committed. Four grids (2x2, 2x4, 4x6 and 6x4)
exercise convolution boundaries, spatial merge order, nontrivial RoPE frequencies,
grouped-query attention and the projector. Every captured stage must satisfy
`abs(error) <= 2e-5 + 2e-5 * abs(reference)`. These are numerical comparisons,
not byte-identical parity claims or evidence of real-model image quality.

## Visual prompts and language prefill

`renderVisionChat` renders still-image parts using the official vision
start/image/end delimiters, preserving the existing tool and thinking handling.
`prepareVisionPrompt` expands each image placeholder to the supplied image-token
count, checks the tokenizer's special IDs against VL configuration, and returns
immutable token IDs and image spans. Images are bound in message/content order.
Video, unbound raw image tokens and unsupported image part types are rejected.

`VisionPrompt.prefill(languageModel, projectedImages)` checks image counts, row
shapes, finite values and context capacity before resetting language-model state.
It copies all rows before yielding, resets the sequence, and passes projected
rows to the shared CPU/GPU forward path. Visual prefix reuse is deliberately
conservative: each complete visual prompt starts from reset.

Example with an imported VL backbone and an already decoded RGB image:

```dart
final encoder = Lfm3VisionEncoder.load(vlDir);
final processor = Lfm3ImageProcessor.load(vlDir);
final image = processor.prepareRgb(rgb, height: height, width: width);
final tokenizer = Lfm3Tokenizer.load(vlDir);
final prompt = prepareVisionPrompt([
  {'role': 'user', 'content': [
    {'type': 'image'},
    {'type': 'text', 'text': 'Describe this image.'},
  ]},
], tokenizer: tokenizer, config: encoder.config,
   imageTokenCounts: [image.imageTokens], enableThinking: false);
// languageModel must be loaded from this VL checkpoint's imported backbone.
await prompt.prefill(languageModel, [encoder.encode(image)]);
// languageModel.logits now predicts the first generated token.
```

This low-level lab API requires the caller to use matching VL encoder and
language weights; it does not establish checkpoint identity or manage generation.
The public `LocalModel` / `ChatSession` APIs accept text; `LocalVisionModel`
and `VisionChatSession` add decoded-RGB image conversations.

`tool/vision_prompt_oracle.py` checks the pinned Jinja template and processor
image-token replacement via Hugging Face. Thirty cases include multiple images,
follow-ups, preserved reasoning, tool results and deferred tools, with thinking
enabled and disabled. The rendering and expanded IDs match the oracle. Separate
CPU/Metal integration tests feed synthetic encoder outputs to the existing Q4
language model and match independent manual embedding placement. Those tests
validate the connection, not real VL image-answer accuracy. The CPU-to-Metal
sequence also exercises Q8 embedding lookup after cache reload. It exposed a
crash in temporary external typed-view reads under the Dart 3.13.5 JIT; direct
native-pointer reads preserve the lookup arithmetic and pass this regression.

## Real-checkpoint validation tools

See [the initial validation report](lfm3_vl_validation_2026-11-01.md) and
[the natural-image/OCR/Japanese follow-up](lfm3_vl_extended_validation_2026-11-01.md) for
reproduction commands, checkpoint identity, numerical bounds and measured results.
The `vision-run` CLI accepts a decoded RGB file and a newly imported backbone,
checks the import fingerprints, and dumps preprocessing, every vision stage,
prompt IDs, prefill/decode logits and generated token IDs. The fingerprints use
the existing cache policy (safetensors header/size and config), not a cryptographic
hash of the tensor payload. The recorded checkpoint was separately verified
against its upstream LFS SHA-256.

`tool/vision_reference.py` creates matching dumps with the checkpoint's actual
Hugging Face processor and model, CPU float32 eager attention and greedy decoding.
`tool/vision_compare.py` requires exact preprocessing, IDs and generated tokens,
with explicit numerical bounds for vision and language outputs. Quantized logit
bounds can be set separately without weakening encoder checks. Five Python tests
ensure incomplete evidence, changed tokens, nonfinite/truncated features and
preprocessing differences cannot silently pass.

## Remaining integration

1. Broaden evaluation beyond one photograph and two clean OCR cards, including
   varied scenes, documents, small text and multi-image prompts.
2. Optimize the vision encoder/projector with native or GPU execution and broaden
   target-device evaluation. `vision-bench` and the device app now record public-API
   timings and process memory; see the [benchmark guide](lfm3_vl_device_benchmark.md).
3. Publish a hosted VL catalog entry.
   Standalone Q8/Q4 packaging is implemented by `pack-vl` and
   `LocalVisionModel.loadBundle`; `VisionModelSpec.fromManifest` and
   `ModelStore.install` install app-hosted bundles using pinned manifest bytes.
   See the [app guide](vision_chat_api.md) and [bundle report](lfm3_vl_bundle_2026-11-01.md).
   PNG/JPEG input is supported by `ChatImage.encoded` and `ChatImage.fromFile`.
   Add other image formats and video after single-image correctness.

Metadata and synthetic operator tests remain fast automated checks. The separate
real-checkpoint runs validate three synthetic color/spatial cases and four
natural-image/OCR/Japanese prompts. They do not establish general image-answer
accuracy or phone performance.
