# Standalone LFM3-VL bundle validation — 2026-11-01

`pack-vl` removes the duplicate full language tensors from the prepared runtime
layout. Apps can load the resulting directory with `LocalVisionModel.loadBundle`
without retaining the original checkpoint or imported backbone. No matrix is
requantized during packing.

## Recorded layout

The source is LiquidAI/LFM3-VL-2B revision
`ba44bd51db20a2753f28c273a1571ccaa210d954`, with the original safetensors SHA-256
`a9bb8b1c159c9d6c3007bc1c7ad8018b599f759d97da2701714e37426283e131`.
The existing language cache uses Q8. Measurements use an Apple M2 Max with
64 GB memory and Dart 3.13.5. Sizes are logical file bytes, not allocated blocks.

| Files | Bytes | GiB |
| --- | ---: | ---: |
| Original checkpoint directory | 3,854,507,901 | 3.59 |
| Imported backbone directory including Q8 cache | 5,421,061,899 | 5.05 |
| Preparation total | 9,275,569,800 | 8.64 |
| Standalone Q8 bundle | 2,885,204,285 | 2.69 |
| Saved | 6,390,365,515 | 5.95 |

The bundle saves 68.9% compared with this preparation layout. Its primary files
are the 378,777,872-byte vision/projector safetensors, 525,078,056-byte language
safetensors and 1,945,513,984-byte native Q8 cache. The language safetensors keeps
BF16 embeddings for unchanged token lookup, plus small norms/convolution kernels.
The Q8 tied output head remains in the native cache. Configuration, tokenizer,
license and import fingerprints are included. The tokenizer is retained at both
root and language paths for compatibility with the existing loaders.

The [machine-readable record](validation/lfm3-vl-bundle-2026-11-01.json) contains
all file sizes and SHA-256 hashes, including the bundle manifest hash. Model
weights, caches and generated bundles are local ignored build artifacts.

## Reproduce

First prepare the source and matching Q8 backbone/cache using the
[initial report](lfm3_vl_validation_2026-11-01.md). Then run:

```sh
dart build cli -o build/vl-cli
build/vl-cli/bundle/bin/dianoia pack-vl \
  --model=build/vl-validation/model \
  --backbone=build/vl-validation/backbone \
  --out=build/vl-validation/portable-q8 --weights=q8

DIANOIA_VL_DIR="$PWD/build/vl-validation/model" \
DIANOIA_VL_BACKBONE="$PWD/build/vl-validation/backbone" \
DIANOIA_VL_VALIDATION_DIR="$PWD/build/vl-validation/extended" \
DIANOIA_VL_BUNDLE="$PWD/build/vl-validation/portable-q8" \
dart test test/vision_public_model_test.dart --name='standalone VL bundle'
```

The RGB card comes from the [extended evaluation](lfm3_vl_extended_validation_2026-11-01.md).
These tests compare the original and packed layouts using the same backend and
public API, checking the answer `東京 123`, generated token count, and SHA-256 of
every complete float32 logit vector observed during prefill/decode. They do not
require CPU and Metal logits to equal each other.

## Results

Both real-checkpoint bundle comparisons passed: the answer was `東京 123` and
all observed logit bytes matched the original layout on CPU Q8 and Metal Q8.
The complete Dart suite passed 138 tests; two optional model-installation tests
were skipped because `DIANOIA_LAB_MODELS` was unset. Formatting and analysis were
clean, and all five Python comparison tests passed. Required lab parity passed
for both chat and long prompts on CPU Q8 and Metal Q8/Q4, with hidden states,
logits, generated IDs and generated logits all byte-identical.

## Integrity and scope

Packing validates tensor shapes/types/ranges and cache matrix coverage, writes
into a sibling staging directory, then publishes after the manifest is complete.
It refuses an existing output. Loading checks every required file's size and
SHA-256 before opening model resources, rejects unknown manifest versions or
file lists, and rejects an unrecorded alternative Q4/Q8 cache. Every load reads
the full bundle for integrity verification. The manifest itself must come from
a trusted installation source; these checks do not authenticate it.

Synthetic tests check exact retained tensor/cache bytes, standalone relocation
with original directories hidden, source-versus-bundle inference, Q4 packing,
corruption and incomplete-input cleanup. Real-checkpoint Q4 bundle quality is
not yet evaluated. Disk savings do not reduce the CPU vision encoder's float32
working memory, or establish phone memory/latency. `ModelStore` now installs
app-hosted bundles via `VisionModelSpec.fromManifest`; see the
[installation guide](vision_chat_api.md#install-a-hosted-vl-bundle). A built-in
hosted catalog entry and device performance work remain. PNG/JPEG decoding is
available through `ChatImage.encoded` and `ChatImage.fromFile`.
