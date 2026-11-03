# Long-running English LiveASR validation

Date: 2026-11-03 (Asia/Tokyo). Host: macOS arm64. This extends the
[initial LiveASR implementation](lfm3-audio-live-2026-11-03.md) by removing the
live mode's cumulative 30-second recording limit.

## Context rollover

The pinned official `transcribe_live` implementation accepts the original
prompt again to reset language context while preserving its encoder cache.
The dianoia session now does this automatically before the next block would
exceed either 375 encoded audio rows (approximately 30 seconds) or the language
model's configured token capacity. Normal 12-row blocks therefore finish the
first context at 372 rows and start the next at the following block.

Only the language backbone is reset. The deferred old text-end marker is
replaced by the original prompt. Microphone capture, frontend overlap and
encoder convolution/attention state continue, without finalization padding at
the boundary. Completed text remains visible; continued text starts a new
paragraph. Raw generated IDs are retained only for the current context, while
the total generated-token counter spans the full stream. Cumulative plain text
grows with recording length; it is not a bounded-size transcript buffer.

The example removes both the live-mode wall-clock timer and sample-count cap.
Ordinary clip ASR and voice conversation retain their 30-second recording
limits. The one-second backpressure bound and stop/cancel/background behavior
remain in place. The PCM API bounds each submitted chunk to 30 seconds for
temporary work, but accepts arbitrarily many chunks. The CLI reads its source
in 80 ms chunks instead of loading a whole file into memory.

## Automated checks

| Check | Result |
|---|---|
| `dart format .` | Passed |
| Root `dart analyze`, example `flutter analyze` | No issues |
| Root default tests | 309 passed, 30 skipped |
| Example default tests | 66 passed, 5 skipped |
| LiveASR tests with real Q4 enabled | 11 passed, no skips |
| Real audio worker tests | 4 passed, no skips |
| `flutter build macos --debug` | Passed; app was not launched |
| Existing text lab parity | 24 byte-identical dumps: chat/long, CPU Q8 and Metal Q8/Q4 |

The protocol fixture executes the pinned upstream method with a second prompt,
confirming replacement of deferred text-end and continued encoder-cache calls.
Native session tests check audio-window rollover, earlier token-capacity
rollover, cumulative text, generated/retained token counts, and cancellation
during a reset. A forced short-window real-model test finishes with 34 total
encoder rows across two language contexts, demonstrating that the encoder did
not restart with the language model.

Controller tests send 32 seconds of PCM without triggering sample-count stop.
A widget test advances its fake clock past 31 seconds and remains recording.
Existing normal-ASR wall-clock-stop tests still pass. These tests use fake
microphones; they do not access audio devices.

## Real stream beyond 30 seconds

The existing clean synthesized English clip was repeated 13 times, for
**34.71975 seconds** of input. Each clip says:

`The quick brown fox jumps over the lazy dog.`

The CPU worker received the PCM in 80 ms chunks. Updates continued after 30
seconds, retained their earlier prefix, and finalized successfully with 584
visible characters. The same worker then supported cancellation and ordinary
ASR reuse. The full four-test worker file passed.

The Metal CLI produced:

| Metric | Result |
|---|---:|
| Input seconds | 34.71975 |
| Live text blocks | 37 |
| Language contexts | 2 |
| Total encoded audio rows | 434 |
| Retained generated IDs in the last context | 20 |
| Final visible characters | 584 |

The first rollover was reported on block 32, after 30.8 seconds of delivered
input. That block belongs to the new context; the previous context covered
372 rows, approximately 29.76 seconds of audio. The final two-row block flushed
the remaining audio and closed naturally. No encoder reset/padding was inserted
at the context boundary. These are unpaced file simulations, not measurements
of microphone latency or a guarantee of recognition accuracy at every boundary.

The samples contain repeated clean synthetic speech. General long-form,
noisy-speech and context-boundary recognition quality remain unvalidated.

## Reproduction and manual path

```sh
# english-long.f32 is the existing mono 16 kHz English clip repeated 13 times.
dart run tool/audio_q4_live_transcribe.dart \
  build/audio-reference build/audio-reference/english-long.f32 --gpu

DIANOIA_AUDIO_EXAMPLE_DIR=build/audio-reference \
DIANOIA_AUDIO_EXAMPLE_PCM=build/audio-reference/english.f32 \
  dart test test/audio_live_test.dart
```

For worker testing, set the same variables to absolute paths and run
`flutter test test/audio_worker_integration_test.dart` from `example`.

In the app, choose **Live transcription (English)** and record beyond 30
seconds. The text should continue, with paragraph breaks at language-context
rollovers. Stop finalizes the accumulated transcript; cancellation or
backgrounding still discards the ongoing recording's text. Physical microphone
capture, app interruption behavior and prolonged device throughput remain
manual checks. No app launch, microphone use or playback was performed here.
