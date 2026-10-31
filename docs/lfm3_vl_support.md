# LFM3-VL-2B integration

Status: implementation in progress. Importing the VL language backbone is
supported; the vision encoder and public image chat API are not implemented.

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

## Remaining integration

1. Compare the imported real VL backbone's intermediate outputs against upstream
   inference, including projected input embeddings.
2. Implement image resize, normalization, patch ordering and spatial merging
   against the upstream processor; establish reference tensors for fixtures.
3. Implement the vision encoder and projector, comparing intermediate outputs
   against upstream CPU inference before optimizing native kernels.
4. Render the VL chat template with the exact number of image placeholders,
   pass projected rows through the implemented input path, and compare logits
   and generated answers with the reference implementation.
5. Add a downloadable model specification and public image chat API after
   end-to-end parity. Add video after single-image correctness.

Metadata tests validate the published VL configuration and rejection of
unsupported or inconsistent configurations. Existing Q4 language-model tests
exercise embedding replacement across chunk
boundaries and protection against text-only prefix reuse. They do not validate
the VL encoder, image understanding, or VL model quality.
