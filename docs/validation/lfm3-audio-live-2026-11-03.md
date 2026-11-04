# English LiveASR validation

Date: 2026-11-03 (Asia/Tokyo). Host: macOS arm64.

## Scope

The audio preview implements recording-time English transcription using the
pinned official `transcribe_live` protocol. It does not repeatedly transcribe
the full recording. The frontend and encoder follow one stream, and the
language context alternates projected audio rows and recognized text inside
the assistant turn. No additional model weights are needed.

The source remains `LiquidAI/LFM3-Audio-2B-GGUF` revision
`02355c1a141e464a064f2ab2c276638dbcb7f38a` with original model/config revision
`0295e4130594c06800fd25451d97139ac2da1aff`. The pinned
[model card](https://huggingface.co/LiquidAI/LFM3-Audio-2B/blob/0295e4130594c06800fd25451d97139ac2da1aff/README.md)
specifies English-only LiveASR. No Japanese live mode or alternative repeated
ASR scheme was implemented.

## Protocol and bounds

- Decode after each 96 mel frames, normally 12 projected audio rows (about
  0.96 seconds of audio). PCM chunk boundaries do not define language blocks.
- Defer the initial prompt until audio rows complete. A nonfinal block queues
  text-end with the next audio chunk, even if its text-token budget is exhausted.
- Suppress audio placeholders on every block. Nonfinal blocks also suppress
  audio/mixed start, turn-end, mixed-end and end-of-text. Final blocks can emit
  the mixed-end marker and stop on text-end or the configured final stops.
- Do not feed the sampled stop marker. Feed every generated text token,
  including the last token of a capped block. Flush an empty final audio tail
  when necessary. Visible text excludes special markers.
- Limit text generation to 48 tokens per nonfinal block and 96 on the final
  block. An unclosed final block is reported as an error. The preview limits
  PCM to 30 seconds and model context to 2048 slots.
- After a successful PCM add, fewer than 96 unprocessed mel frames remain.
  Frontend/encoder caches remain bounded; language context grows to its cap.

`tool/audio_live_oracle.py` executes the actual pinned `transcribe_live` method
with scripted logits and synthetic encoder rows. The fixture records source
URL/SHA-256 and Torch version. It covers suppression/final stops, capped blocks,
deferred markers and no-new-audio calls. These are protocol comparisons;
encoder and language arithmetic reuse the existing numerical/parity checks,
not a new full-checkpoint Torch LiveASR numerical comparison.

## Automated checks

| Check | Result |
|---|---|
| `dart format .` | Passed |
| Root `dart analyze`, example `flutter analyze` | No issues |
| Root default tests | 306 passed, 30 skipped |
| Example default tests | 64 passed, 5 skipped |
| LiveASR tests with real Q4 enabled | 8 passed, no skips |
| Real audio worker integration tests | 4 passed, no skips |
| `flutter build macos --debug` | Passed; app was not launched |
| Existing text lab parity | 24 byte-identical dumps: chat/long, CPU Q8 and Metal Q8/Q4 |

Session tests check deferred prompt/text-end consumption, finalization with no
new audio, input validation before language mutation, context bounds,
concurrent calls, cancellation, external backbone changes, feature bounds and
callback failure. Real tests compare 80/100 ms PCM schedules, check early text
before finalization, and flush a zero-row final tail. Controller/widget tests
use fake devices to check partial replacement, final text, forced English,
late-update rejection after cancel and a new recording, and the live controls.

The real worker test also rejects Japanese LiveASR requests, cancels a live
recording and then reuses the same worker for ordinary ASR. The previous
ASR/TTS transport tests continue to pass. No microphone or playback device is
used by these tests.

## Real English clip

An existing clean synthesized clip, 2.67075 seconds long, says:

`The quick brown fox jumps over the lazy dog.`

| Input delivered | Projected rows in block | Cumulative visible text |
|---:|---:|---|
| 1.04 s | 12 | The quick |
| 2.00 s | 12 | The quick brown fox jumps |
| 2.67075 s, finalized | 10 | The quick brown fox jumps over the lazy dog. |

CPU and Metal produced the same block IDs and final text on this clip. CPU
80/100 ms PCM schedules also matched IDs. All three blocks closed naturally.
The final raw token sequence starts with the mixed-end marker, which is not
shown in the visible transcript. These are unpaced file simulations: input-time
markers are not measured microphone latency or a general ASR accuracy result.

## Reproduction

```sh
python tool/audio_live_oracle.py
DIANOIA_AUDIO_EXAMPLE_DIR=build/audio-reference \
DIANOIA_AUDIO_EXAMPLE_PCM=build/audio-reference/english.f32 \
  dart test test/audio_live_test.dart

dart run tool/audio_q4_live_transcribe.dart \
  build/audio-reference build/audio-reference/english.f32
# Append --gpu for a supported GPU language backbone.
```

Set the same environment variables to absolute paths and run
`flutter test test/audio_worker_integration_test.dart` from `example` for the
worker path. The CLI reads mono 16 kHz f32le PCM and emits JSON-line block,
cumulative-text and done events.

## Manual checks remaining

Open **LFM3-Audio-2B Q4_0**, select **Live transcription (English)** and press
**Start live transcription**. English text should appear while recording;
**Stop live transcription** finalizes it. **Cancel recording**, backgrounding
and route exit discard the current text. Returning never restarts the mic.
The existing one-second PCM backpressure limit also applies.

Physical microphone behavior, interruption handling, actual processing delay
and noisy/diverse speech remain manual checks. No app launch, microphone use
or listening was performed. Mobile/Windows builds and text-context restarts
for recordings longer than 30 seconds were not validated or added here.
