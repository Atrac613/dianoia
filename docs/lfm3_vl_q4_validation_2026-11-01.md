# LFM3-VL Q4 versus Q8 validation — 2026-11-01

Q4 reduces the standalone bundle from 2,885,204,285 to 2,060,427,581 bytes
(28.6%). On an Apple M2 Max, generation throughput increased in this small
comparison, while image-to-first-text latency remained essentially unchanged.
Japanese OCR omitted a space: Q8 returned `東京 123`, Q4 returned `東京123`.
The example uses the validated Q4 preset.

## Model and method

This evaluates dianoia's existing BF16-to-Q4 quantizer: 153 language matrices
are Q4 and the tied output head remains Q8. The vision/projector payload, BF16
token embedding and other retained small tensors are byte-identical to the Q8
bundle. The CPU vision encoder is common to both. The source is
LiquidAI/LFM3-VL-2B revision `ba44bd51db20a2753f28c273a1571ccaa210d954`;
its 3,836,417,920-byte safetensors SHA-256 was checked again against
`a9bb8b1c159c9d6c3007bc1c7ad8018b599f759d97da2701714e37426283e131`.
This is post-training quantization of that BF16 checkpoint; upstream QAD GGUF
quality is outside this measurement.

The host has an Apple M2 Max and 64 GiB RAM. Each backend/format runs in a fresh
AOT process, sequentially in Metal Q8, Metal Q4, CPU Q8, CPU Q4 order. All use
four native threads, a 512-token context, at most 32 generated tokens and no
system prompt. Each of seven prompts over six images gets three fresh visual
turns and one ungraded text-only follow-up retaining visual features: 84 fresh
turns plus 28 follow-ups. The three fresh answers are identical in every case
and configuration, and CPU/Metal return the same text for each weight format.
Thermal-state samples remain nominal (0).

The [machine-readable evidence](validation/lfm3-vl-q4-2026-11-01.json) records
input hashes, source/binary hashes, bundle manifests, every response and turn
timing, memory summaries, and packing-parity logit hashes. Raw 100 ms memory
samples and logs remain under ignored `build/vl-q4-validation/results/`.
Weights and caches are not committed.

## Answers

| Case | Q8, CPU and Metal | Q4, CPU and Metal |
| --- | --- | --- |
| Solid red | `Red` | `Red` |
| Solid blue | `Blue` | `Blue` |
| Left red, right blue: left color | `Red` | `Red` |
| Astronaut clothing, English | `Suit` | `Space suit` |
| Astronaut clothing, Japanese | `宇宙飛行士の服を着ています。` | `宇宙服です。` |
| Latin OCR | `AB42` | `AB42` |
| Japanese OCR | `東京 123` | `東京123` |

The three color/spatial tasks and Latin OCR pass exact matching. Japanese OCR
passes character-content matching after whitespace removal, but fails the
original exact criterion on both Q4 backends; the expected string was not
relaxed. Both existing Q8-gold packing tests fail at that exact-answer assertion
when pointed at Q4. Independent Q4 packing tests check preservation of every
prefill/decode logit without conflating that check with Q8's expected wording.

The harness's predeclared photo keyword checks are retained in the JSON as
lexical diagnostics: `spacesuit` does not match `Space suit`, and `宇宙服` does
not match `宇宙飛行士の服`. Manual inspection finds Q4's clothing answers more
specific in this one photograph. Those lexical booleans are not an overall
accuracy score. The fixtures cover a small smoke test, not a representative
vision/language dataset.

## Timing and memory

| Backend | Weights | First text, median (s) | Decode (tok/s) | Load with integrity (s) | Peak RSS (GB) | Sampled peak footprint (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Metal | Q8 | 4.603 | 98.3 | 19.32 | 1.092 | 1.105 |
| Metal | Q4 | 4.601 | 122.3 | 14.00 | 1.092 | 1.104 |
| CPU | Q8 | 5.612 | 39.7 | 19.20 | 3.055 | 1.094 |
| CPU | Q4 | 5.644 | 44.5 | 13.69 | 2.230 | 1.092 |

GB is decimal. First text is the median over 21 fresh turns per configuration
and includes image encoding. Decode throughput is total generated tokens divided
by total decode time over those turns, excluding follow-ups. Q4 is about 24%
faster in this Metal decode sample and 12% faster on CPU. Replies are short and
not always the same length/text across formats; these rates need longer-output,
sustained runs before extrapolation. Each format has one process/load trial.

RSS is the process-lifetime high-water mark. Physical footprint is sampled at
100 ms plus phase boundaries; synchronous work can hide intermediate peaks.
These are process-memory measurements, not a direct GPU-VRAM measurement. CPU
peak RSS decreases by about 0.825 GB (27%). The measured Metal process-memory
metrics barely change. Every run closes with zero live GPU buffers.

## Packing and verification

The separate optional Q4 test compares original-checkpoint plus Q4-backbone
inference with the standalone Q4 bundle, on CPU and Metal. It requires identical
response text, token count, and SHA-256 for every complete float32 logit vector.
The recorded tests and lab parity establish arithmetic preservation separately
from the Japanese OCR whitespace difference above.

Formatting and static analysis pass. The default Dart suite passes 174 tests,
with 24 optional model tests skipped; the separate real Q4 packing run passes
both CPU and Metal tests. Lab parity compares chat and long
prompts on CPU Q8 and Metal Q8/Q4; compared dumps are removed immediately after
matching to keep disk use bounded. Exact verification counts are in the JSON.

## Reproduce

Start with the pinned original checkpoint, its imported full BF16 language
backbone, the existing standalone Q8 bundle, and the six validation images from
the earlier [extended validation](lfm3_vl_extended_validation_2026-11-01.md) and
[color validation](lfm3_vl_validation_2026-11-01.md). Use a separate writable
backbone directory for Q4 so the Q8 baseline's cache selection stays unchanged.

```sh
dart build cli --target=tool/validate_vl_quantization.dart \
  -o build/vl-q4-validation/runner
RUN=build/vl-q4-validation/runner/bundle/bin/validate_vl_quantization
"$RUN" --prepare=q4 --backbone=PATH_TO_FULL_BF16_BACKBONE

dart run bin/dianoia.dart pack-vl --model=PATH_TO_ORIGINAL_VL \
  --backbone=PATH_TO_FULL_BF16_BACKBONE --weights=q4 --out=NEW_Q4_BUNDLE

# Run four separate processes, changing --bundle and --backend for each.
"$RUN" --bundle=NEW_Q4_BUNDLE --images=build/vl-validation \
  --backend=gpu --runs=3 --out=NEW_RESULT_JSON

DIANOIA_VL_DIR=PATH_TO_ORIGINAL_VL \
DIANOIA_VL_Q4_BACKBONE=PATH_TO_FULL_BF16_BACKBONE \
DIANOIA_VL_Q4_BUNDLE=NEW_Q4_BUNDLE \
DIANOIA_VL_Q4_IMAGE=build/vl-validation/extended/ocr-ja.png \
DIANOIA_VL_Q4_EVIDENCE=NEW_PACKING_EVIDENCE_JSON \
dart test test/vision_q4_bundle_test.dart
```

## Adoption scope

Q4 is useful for reducing installed size and shows higher generation throughput
in these desktop samples. The current BF16-download conversion path still needs
about 3.85 GB of network transfer. The example now selects Q4, while Q8 remains available through the installation
API. Phone measurements and broader image/OCR coverage remain outstanding; exact
transcription has an observed whitespace difference to account for.
