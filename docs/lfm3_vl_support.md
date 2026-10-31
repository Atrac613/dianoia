# LFM3-VL-2B integration

Status: implementation in progress. Importing the VL language backbone is
supported; a CPU reference vision encoder is implemented. Public image chat
and optimized vision inference are not implemented.

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
the original config as `vl_config.json`. Vision weights remain in the source.
Output is published after the copy completes; existing output is rejected.
Copying uses a 1 MiB buffer rather than materializing large tensors in memory.
The output is a full-weight checkpoint, not a quantized device bundle: existing
`run`, `pack` and Q8/Q4 cache paths can process it. Use token-ID prompts for
backbone comparisons; the text-only chat renderer is not a VL chat template.

Tests compare imported BF16 bytes and CPU logits against a canonical miniature
checkpoint containing all three language layer types, and check truncated or
incomplete input rejection. Actual checkpoint metadata was inspected at revision
`ba44bd51db20a2753f28c273a1571ccaa210d954`; full VL-weight inference remains
unverified. Only the official single-file checkpoint layout is supported.

## RGB image preprocessing

The lab API `Lfm3ImageProcessor.load(modelDir)` reads `processor_config.json`.
`prepareRgb(rgb, height: ..., width: ...)` accepts packed RGB uint8 pixels and
returns float32 `pixelValues`, `gridThw`, dimensions and the image-token count.
It implements aspect-ratio-preserving resize to patch/merge multiples,
Python's ties-to-even dimension rounding, antialiased uint8 bicubic resizing,
fused normalization and spatially grouped patches with duplicated temporal
samples. Pixel budgets are areas, despite the upstream `shortest_edge` and
`longest_edge` field names.

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
reference hashes.

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

## Remaining integration

1. Compare the imported real VL backbone's intermediate outputs against upstream
   inference, including projected input embeddings.
2. Compare the full-size vision encoder and projector against upstream using
   real VL weights, then optimize native/GPU execution.
3. Render the VL chat template with the exact number of image placeholders,
   pass projected rows through the implemented input path, and compare logits
   and generated answers with the reference implementation.
4. Add a downloadable model specification and public image chat API after
   end-to-end parity. Add video after single-image correctness.

Metadata tests validate the published VL configuration and rejection of
unsupported or inconsistent configurations. Existing Q4 language-model tests
exercise embedding replacement across chunk
boundaries and protection against text-only prefix reuse. The separate encoder tests validate synthetic CPU arithmetic; real-model image
understanding and VL model quality remain unverified.
