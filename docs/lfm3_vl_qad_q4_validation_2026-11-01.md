# Official LFM3-VL QAD Q4 validation — 2026-11-01

The official QAD Q4_0 language GGUF with its paired BF16 vision/projector GGUF
completed seven visual prompts in llama.cpp and in dianoia. Every fresh dianoia
answer matches the direct GGUF reference on CPU and Metal, three times per case.
Japanese OCR returns `東京 123`, recovering the space omitted by the earlier
BF16-to-Q4 PTQ run. This is a small fixture comparison, not a broad accuracy claim.

The previous BF16, Q8 and PTQ-Q4 validation weight directories have been deleted.
Test images, configuration/tokenizer/license files, logs and published measurement
records remain. Only this task's QAD validation weights are retained.

## Sources and execution

Both official files come from
[LiquidAI/LFM3-VL-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/tree/b36554cfd35854b06429900131730cd80f092cdb),
revision `b36554cfd35854b06429900131730cd80f092cdb`. SHA-256 was checked on download
and checked again after inference:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `LFM3-VL-2B-QAD-Q4_0.gguf` | 1,049,615,648 | `621c455376cb96a20891697deb9975c6fb47dd5ea98ed3ad1f3b7e8f24f0b1e6` |
| `mmproj-LFM3-VL-2B-BF16.gguf` | 378,996,032 | `8e75a88afc0e6a3bcd637becbe720447ea0794934b7b417a1f42d5cfe0d9c288` |

Their combined GGUF payload is 1,428,611,680 bytes. Config, processor, tokenizer
and license metadata are retained from the previously pinned original VL
checkpoint revision `ba44bd51db20a2753f28c273a1571ccaa210d954`.

The direct reference uses llama.cpp commit
`060b2fea4ed528810ddc91d48fb22b3e4d63eb85`, built locally under a temporary
directory. The installed build 7790 did not recognize architecture `lfm3`; it
was not used for the reported reference results. The reference uses the GGUF's
Jinja template with `enable_thinking` forced false, greedy decoding, repetition
penalty 1, four threads, 512 context tokens and a 32-token generation limit.
The language model runs on Metal and its vision encoder on CPU. This reference
has a different encoder implementation, so its wall times are diagnostic rather
than a dianoia performance baseline.

Dianoia uses the same image files/questions and the same thread, context,
generation-limit and thinking settings as the previous PTQ comparison. It runs
three fresh visual turns plus an ungraded visual-context follow-up for each of
seven prompts over six images, in separate sequential Metal and CPU AOT
processes: 42 fresh turns and 14 follow-ups. All fresh answers repeat exactly;
thermal-state samples remain nominal (0).

## Import scope

`tool/prepare_vl_qad.py` reconstructs the vision checkpoint layout from the
*paired official mmproj*, preserving BF16 and F32 payload bytes. It reverses
llama.cpp's documented split of the temporal patchifier: the two F16 slices are
interleaved as `[output, channel, temporal, patch_y, patch_x]` and widened exactly
to F32. The mapping follows the pinned reference's `conversion/lfm3.py`. Synthetic
unit tests verify raw BF16/F32 preservation and temporal ordering.

`tool/prepare_vl_qad.dart` uses dianoia's existing `importGguf` for language weights.
All **153 Q4 matrices** retain their exact nibbles and half-scale bytes; each is
hashed against its official GGUF tensor. No Q4 language matrix is requantized.

There is one material adapter difference: the existing importer converts the
GGUF's tied **Q6_K embedding/output head to Q8**, and uses that Q8 matrix for both
token lookup and output projection. Consequently these results validate the
*dianoia import* against the official reference on these fixtures; they do not
establish full numerical equivalence to native Q6_K GGUF inference.
The app installer packages this layout into a portable version-one VL bundle
with explicit `input_embedding: q8-cache`; app packaging is checked separately
below.

## Answers

The reference and both dianoia backends return the following text. Each dianoia
entry matches in all three fresh repeats:

| Case | Official QAD reference and dianoia QAD | Archived PTQ Q4 |
| --- | --- | --- |
| Solid red | `Red` | `Red` |
| Solid blue | `Blue` | `Blue` |
| Left red, right blue: left color | `Red` | `Red` |
| Astronaut clothing, English | `Suit` | `Space suit` |
| Astronaut clothing, Japanese | `宇宙飛行士の服を着ています。` | `宇宙服です。` |
| Latin OCR | `AB42` | `AB42` |
| Japanese OCR | `東京 123` | `東京123` |

QAD's seven answers also match the archived Q8 answers. The photo answers are
plausible but the English `Suit` is less specific than `Space suit` in PTQ Q4.
There is no uniform quality ordering to infer from one photograph and two OCR
cards. The inherited `spacesuit`/`宇宙服` keyword booleans in raw runs remain
lexical diagnostics, not an aggregate accuracy score.

## Timing and memory

Apple M2 Max, 64 GiB RAM. GB below is decimal.

| Dianoia backend | First text, median (s) | Decode (tok/s) | Peak RSS (GB) | Sampled peak footprint (GB) |
| --- | ---: | ---: | ---: | ---: |
| Metal, QAD Q4 import | 4.587 | 123.8 | 1.086 | 1.103 |
| CPU, QAD Q4 import | 5.658 | 43.8 | 2.229 | 1.093 |

First text includes CPU image encoding. Decode throughput is total generated
tokens divided by total decode time over fresh turns. The earlier PTQ-Q4 values
were Metal 4.601 s / 122.3 tok/s and CPU 5.644 s / 44.5 tok/s. These are historical
observations: those weights were deleted before this run, replies differ, and
this import changes the embedding representation. The small timing differences
are insufficient to attribute a performance gain to QAD training.

The imported runtime files total **1,540,282,921 bytes**, excluding source GGUFs,
preparation tools and temporary configuration copies. Its smaller size than the
2.06 GB PTQ bundle primarily reflects omission of that bundle's retained BF16
input embedding, not an intrinsic QAD storage advantage. The layout is not the
same packaged format. RSS is process-lifetime high water; 100 ms footprint
samples can miss synchronous peaks. These are process-memory metrics, not a
direct GPU-VRAM measurement. Both runs close with zero live GPU buffers.

## Cleanup

The following directories were removed and their absence checked again after
measurement:

- `build/vl-validation/model`
- `build/vl-validation/backbone`
- `build/vl-validation/portable-q8`
- `build/vl-q4-validation/backbone`
- `build/vl-q4-validation/portable-q4`

The deletion inventory totals approximately **15.34 GB of allocated blocks**
after deduplicating hard-linked inodes. This is an allocation estimate, rather
than an assertion about snapshot accounting. Images, raw activation evidence,
old benchmark JSON, source code and unrelated work remain. Current QAD weights
and their imported layout live under ignored `build/vl-qad-validation/`.

## Verification and evidence

[Machine-readable results](validation/lfm3-vl-qad-q4-2026-11-01.json) contain
all 153 Q4 payload audits, source file hashes, reference text, every dianoia turn,
memory summaries, executable hashes, and the deletion inventory. Full memory
samples and diagnostic logs remain in `build/vl-qad-validation/results/`.

Dart formatting/analysis, the default Dart suite, Python conversion tests and
CPU/Metal lab parity were checked. Exact counts are recorded in the JSON.
The example uses `VisionGgufSpec.lfm3VlQadQ4`, the official QAD Q4 preset. Phone measurements and broader visual coverage
remain outstanding before generalizing these results.

## App installation and packaging

`VisionGgufSpec.lfm3VlQadQ4` now installs the two pinned GGUFs and four metadata
files through `ModelStore`. The example labels the model `LFM3-VL-2B (QAD Q4_0)`
and uses a separate `lfm3-vl-2b-qad-q4-b36554c` store ID and
`LFM3-VL-2B-QAD-Q4` sideload directory. Download size is 1,446,534,952 bytes;
installed size is approximately 1.54 GB. First setup allows 5 GB working space.

The runtime importer is Dart/native code under `lib/src/vision_gguf.dart`, with
no Python dependency in the app. A real loopback-HTTP installation checks all
six pinned downloads, prepares and packs the model, removes downloads and
intermediates, then loads it through `LocalVisionModel.loadBundle`. CPU and
Metal OCR text and every prefill/decode logit hash are byte-identical to the
previously validated unpacked QAD import. All 153 Q4 payloads, all 138 vision
payloads and the adapted Q8 embedding/head are also byte-identical to the
validated import. The optional test is
`test/vision_qad_model_store_integration_test.dart`. Synthetic GGUF tests cover
cancellation/resume, tensor ordering, malformed input and policy validation.
The current default suite passes 180 Dart tests (25 optional tests skipped);
the real HTTP installation test passes separately, and the example passes
19 Flutter tests including real QAD image response and stop behavior.

## Reproduce

Acquire the two pinned GGUF files above and the four pinned metadata files.
Create a new vision directory and language import, then build the validation
runner. The preparation audit pins every generated runtime file for each load.

```sh
python3 tool/prepare_vl_qad.py --mmproj=MM_PROJ_GGUF \
  --config=PINNED_METADATA_DIRECTORY --out=NEW_VISION_DIRECTORY

dart build cli --target=tool/prepare_vl_qad.dart -o build/qad-prepare
build/qad-prepare/bundle/bin/prepare_vl_qad \
  QAD_GGUF NEW_VISION_DIRECTORY NEW_BACKBONE_DIRECTORY NEW_AUDIT_JSON

dart build cli --target=tool/validate_vl_quantization.dart -o build/qad-runner
build/qad-runner/bundle/bin/validate_vl_quantization \
  --source=NEW_VISION_DIRECTORY --backbone=NEW_BACKBONE_DIRECTORY \
  --audit=NEW_AUDIT_JSON --images=build/vl-validation \
  --backend=gpu --runs=3 --out=NEW_METAL_RESULT_JSON
# Repeat in a new process with --backend=cpu and a different --out.

python3 tool/validate_vl_qad_reference.py --cli=PINNED_LLAMA_MTMD_CLI \
  --model=QAD_GGUF --mmproj=MM_PROJ_GGUF \
  --baseline=docs/validation/lfm3-vl-q4-2026-11-01.json \
  --out=NEW_REFERENCE_JSON
```
