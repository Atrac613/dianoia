# LFM3-VL-2B integration

Status: implementation in progress. Loading a VL checkpoint and passing images
through the public chat API are not supported yet.

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

## Remaining integration

1. Load the nested VL config and checkpoint tensor prefixes without allowing a
   VL checkpoint to be mistaken for a ready-to-use text bundle.
2. Implement image resize, normalization, patch ordering and spatial merging
   against the upstream processor; establish reference tensors for fixtures.
3. Implement the vision encoder and projector, comparing intermediate outputs
   against upstream CPU inference before optimizing native kernels.
4. Render the VL chat template with the exact number of image placeholders,
   pass projected rows through the implemented input path, and compare logits
   and generated answers with the reference implementation.
5. Add a downloadable model specification and public image chat API after
   end-to-end parity. Add video after single-image correctness.

Existing Q4 language-model tests exercise embedding replacement across chunk
boundaries and protection against text-only prefix reuse. They do not validate
the VL encoder, image understanding, or VL model quality.
