# Physical iPhone VL measurements — 2026-11-01

LFM3-VL-2B Q8 completed three fresh image turns and a retained-image follow-up
on a physical iPhone 15 Pro Max, with both CPU and Metal language backends.
Every answer was `東京 123`. The shared public-API benchmark ran in a worker
isolate in the dedicated Flutter Release app; the vision encoder used the current
CPU implementation in both runs. Android was not connected and was not measured.

## Workload and identity

The bundle came from LiquidAI/LFM3-VL-2B revision
`ba44bd51db20a2753f28c273a1571ccaa210d954`. Its manifest SHA-256 was
`6084c52d761459c8c7078f9a703e76e2f6d6c7c2e65b5af3403095e26e4600c9`.
It is the same 2,885,204,285-byte Q8 bundle from the
[packaging report](lfm3_vl_bundle_2026-11-01.md).

The input was the 384x192 Japanese card from the
[extended evaluation](lfm3_vl_extended_validation_2026-11-01.md), read from PNG.
The context limit was 512, language CPU threads 4, and output limit 32. Each
fresh turn had 97 prompt tokens; the follow-up had 124. Answers contained only
three generated tokens, so decode rates below are short-answer observations.
The runtime reported iOS 26.7.2 (23H128), six CPU cores and Dart 3.13.5 AOT.

The [compact record](validation/lfm3-vl-iphone-2026-11-01.json) includes input and
bundle hashes, source/helper hashes, full turn counters, and process-memory
summaries. Device identifiers and app-container paths are not included. Raw
checkpoint records remain under ignored `build/device_bench/vision/`.

## Observed timings

| Measurement | Metal language / CPU vision | CPU language / CPU vision |
| --- | ---: | ---: |
| Verified model load | 16.92 s | 16.81 s |
| PNG file read/decode | 0.020 s | 0.014 s |
| First fresh turn, first text | 9.10 s | 8.22 s |
| Next two fresh turns, first text | 7.14–7.30 s | 8.20–8.37 s |
| Median of all three fresh first-text times | 7.30 s | 8.22 s |
| Retained-image follow-up, first text | 1.72 s | 3.34 s |
| Fresh-turn median decode rate, three-token answers | 17.35 tok/s | 17.51 tok/s |
| Apple thermal state, process start → end | 0 → 1 | 1 → 1 |

These are two sequential runs, not a controlled backend ranking. The file caches,
power conditions and thermal states were not equalized. Model loading includes
full SHA-256 verification; the current measurement does not separate hashing,
file reading and weight initialization. Fresh turns re-encode the image, whereas
the follow-up retains features and uses a longer prompt. Their timing difference
is not a direct isolated measurement of the encoder.

The next profiling target is the approximately 17-second verified load, followed
by a breakdown of image encoding versus language prefill. The short decode samples
do not establish sustained speed or explain a backend's maximum throughput.

## Process memory and cleanup

| Measurement | Metal language | CPU language |
| --- | ---: | ---: |
| OS process RSS high-water | 913.61 MiB | 2730.98 MiB |
| Maximum sampled physical footprint | 1024.35 MiB | 985.58 MiB |
| Live native GPU buffers, maximum → after close | 52 → 0 | 0 → 0 |

The two memory measures have different accounting. Process RSS and physical
footprint are not total system/GPU residency; this table does not establish a
threefold difference in total model memory. Physical footprint was sampled at
phase boundaries, first text and by a 100 ms Dart timer, which can miss peaks
inside synchronous work. RSS high-water marks cover the entire fresh process.

After close, process RSS was about 823 MiB for Metal and 863 MiB for CPU. No GC
was forced, so retained Dart heap pages are included. Native GPU buffers returned
to zero. This is a successful cleanup observation, not a long-run leak test.
No phone out-of-memory termination occurred in either recorded run.

## Reproduce and verification

Follow the [benchmark guide](lfm3_vl_device_benchmark.md). Run the iPhone script
once with `BACKEND=gpu` and once with `BACKEND=cpu`; it starts a fresh app process
for each backend and restores the earlier text benchmark configuration on exit.
The benchmark source hash in both device records matches the committed helper.
The app was built from the working tree before the implementation commit; the
records retain that source revision and dirty-tree flag rather than implying
that a later commit was already installed.

The full Dart suite passed 180 tests with all local-model integration tests
enabled. Analysis and formatting were clean. Five Python comparison tests passed.
The required lab comparison was byte-identical for chat and long cases on CPU
Q8 and Metal Q8/Q4. The AOT desktop CLI is also exercised with the same real OCR
bundle; those desktop smoke timings are not part of the phone table.
