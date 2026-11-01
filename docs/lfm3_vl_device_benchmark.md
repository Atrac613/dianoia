# VL timing and process-memory benchmark

`vision-bench` and the device benchmark app use the same lab helper,
`benchmarkVisionBundle`, exercising `LocalVisionModel.loadBundle`, image-file
input and `VisionChatSession`. This measures the current CPU vision encoder
with a quantized CPU or GPU language backbone. It does not change arithmetic.

## Desktop CLI

Build the CLI, then run each backend in a separate process so process-lifetime
memory high-water marks start afresh:

```sh
dart build cli -o build/vision-bench-cli
build/vision-bench-cli/bundle/bin/dianoia vision-bench \
  --bundle=build/vl-validation/portable-q8 \
  --image=build/vl-validation/extended/ocr-ja.png \
  --prompt='画像に書かれた文字を、そのまま書き起こしてください。' \
  --expected='東京 123' --backend=gpu --runs=3 --max-new=32 --max-seq=512 \
  --out=build/vision-gpu-benchmark.json
```

The output must be a new file. Use `--backend=cpu` for a CPU process and
`--native-threads=4` to set language kernel threads. `--max-image-tokens`
(`max_image_tokens` in the device configuration) sets the
[image token budget](vision_chat_api.md#image-token-budget), 1024 by default;
the record keeps the request and the budget in effect. Quantization comes from the
bundle; it is not rebuilt. Without `--expected`, matching fresh answers is a
repeatability check, not evidence of answer correctness. The command returns
nonzero on a quality mismatch or an exception.

## Physical iPhone

The existing `tool/device_bench` app accepts `mode: "vision"` in its documents
`bench_config.json`. It runs the same helper in its worker isolate and saves
`Documents/bench/vision-latest.json`. The dedicated script builds a signed Release
app, installs it, copies the local bundle/image, launches a fresh process, polls
its record, and restores the earlier text benchmark configuration on exit:

```sh
VL_BUNDLE="$PWD/build/vl-validation/portable-q8" \
IMAGE="$PWD/build/vl-validation/extended/ocr-ja.png" BACKEND=gpu \
tool/device_bench/run_vision_ios.sh DEVICE_ID
```

Pass an explicit physical device identifier from `xcrun devicectl list devices`.
Keep the device unlocked and the app in the foreground. The configured Flutter
installation and Xcode signing team must support this device. Set `FLUTTER` to
a specific Flutter executable if needed. This is local device deployment and
does not publish a model or upload a store build.

The default case is the Japanese OCR card from the [extended evaluation](lfm3_vl_extended_validation_2026-11-01.md).
Set `PROMPT`, `EXPECTED`, `RUNS` (default 3), `WAIT_SECONDS` (default 600), or
`OUT` to override the run. `BACKEND` accepts `cpu` or `gpu`. The default output
is an ignored timestamped directory under `build/device_bench/vision/`.
A bundle directory keyed by its manifest hash is retained on the phone for
repeat runs; every model load still verifies the complete payload. The image is
replaced per run. This script does not erase earlier benchmark model bundles.

A missing or unfinished record is an interrupted measurement, not a valid
latency result or proof of an out-of-memory termination. Inspect the last phase,
`Documents/bench/error.txt` and device crash diagnostics. Device records include
the build source revision, dirty-tree flag and benchmark-helper hash. The
[2026-11-01 iPhone report](lfm3_vl_iphone_validation_2026-11-01.md) records the
first physical iPhone run. Android device execution remains unvalidated.

## Read the record

- `load_s` includes bundle SHA-256 verification and model initialization.
  File-system cache state is not controlled; this is not a cold-storage test.
- `image_decode_s` measures local file I/O and PNG/JPEG decode separately from
  model inference. The image hash is computed outside this timer.
- Each `fresh_N` starts a new conversation and attaches the decoded image again,
  so the vision encoder runs for every fresh case. This is independent of the
  first model-load timing. No unrecorded inference warm-up is performed.
- `follow_up` uses the last conversation without a new image, retaining projected
  image features. The whole prompt is still re-prefilled; reused tokens are zero.
- `first_text_s` is wall time until the first text callback, including image
  encoding, prefill and the first decode step. A callback can contain empty text
  while UTF-8 is incomplete; this counter measures callback arrival. `prefill_s` includes image encoding
  and prompt prefill, and ends before that first callback. `decode_s` and
  `decode_tokens_per_s` use the existing `ChatTurn` counters and include normal
  event-loop/callback overhead. Short OCR continuations are not sustained decode
  benchmarks.
- `rss_bytes` and `process_peak_rss_bytes` describe process RSS and its lifetime
  high-water mark. `physical_footprint_bytes` is the native process footprint
  (iOS memory accounting; RSS fallback on Linux/Android). These are distinct
  measurements; neither is a measurement of total system/GPU residency.
- Memory is sampled at phase boundaries, first text, and by a 100 ms timer. Dart
  timers cannot interrupt synchronous work, so the sampled physical-footprint
  maximum can miss transient peaks. The OS RSS high-water value is not reset
  between turns. Negative native footprint values mean unavailable.
- Apple `thermal_state` values are captured with each memory sample. Compare
  workload, build, context, power/thermal state and cache conditions before drawing
  performance conclusions. There is no forced garbage collection after close.
  Native GPU buffer counts are recorded; retained Dart heap after close alone
  does not establish a resource leak.

Phase checkpoints are written before loading and after major operations. An OS
termination can leave only the last completed checkpoint; it cannot run Dart
cleanup or write a final error record. Timing includes sampling overhead and
occasional checkpoint serialization. Raw records stay under ignored `build/`;
commit compact results and reproduction details, not model weights or caches.
