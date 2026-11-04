# Native LFM3-Audio voice conversation validation

Date: 2026-11-03 (Asia/Tokyo). Host: macOS arm64.

## Scope and sources

The session feeds projected microphone/audio rows directly into LFM3-Audio and
generates native interleaved text and audio codes. It does not transcribe the
question and send its text to a separate chat/TTS pipeline. The example uses a
record/send/reply workflow with text updates during generation and complete-WAV
playback afterwards. It keeps multiple accepted turns and offers an explicit
new-conversation action.

The official Q4_0 pair remains pinned at GGUF revision
`02355c1a141e464a064f2ab2c276638dbcb7f38a`; configuration and reference graphs
remain pinned at `0295e4130594c06800fd25451d97139ac2da1aff`. The pinned
[model card](https://huggingface.co/LiquidAI/LFM3-Audio-2B/blob/0295e4130594c06800fd25451d97139ac2da1aff/README.md)
specifies English-only voice chat. Existing ASR/TTS language options remain
separate. No new weights or model downloads are required for voice chat.

`tool/audio_conversation_oracle.py` executes the actual AST of the pinned
`_generate` per-row block with scripted text tokens/audio frames and zero
embedding rows. Its fixture records source URLs, SHA-256 hashes and Torch
version. It tests six-text/twelve-audio alternation, an early text-end marker,
speech EOS inside a burst, reopening mixed/audio modes, and turn termination.
It also independently renders the first prompt with the official processor's
template. These are protocol tests, not a new whole-model numerical oracle.
Encoder/depth/codec arithmetic is covered by the previous
[ASR](lfm3-audio-q4-2026-11-02.json) and
[TTS](lfm3-audio-tts-2026-11-02.json) reference comparisons.

## Automated checks

| Check | Result |
|---|---|
| `dart format .` | Passed |
| Root `dart analyze`, example `flutter analyze` | No issues |
| Root default tests | 299 passed, 29 skipped |
| Example default tests | 61 passed, 4 skipped |
| Conversation tests with real Q4 enabled | 9 passed, no skips |
| Real conversation worker integration | 1 passed, no skips |
| `flutter build macos --debug` | Passed; app was not launched |
| Existing text lab parity | 24 byte-identical dumps: chat/long, CPU Q8 and Metal Q8/Q4 |

Default session tests exercise input validation, context rejection before
state changes, concurrent calls, cancellation, callback failure, foreign model
state, and closed generation weights. Real session tests additionally verify
speech EOS normalization/feedback, owned PCM chunk assembly, deferred reply
acceptance, discarded replies, context retention across an unrelated backbone
operation, a semantic follow-up, explicit reset and bounded history storage.

Worker integration tests exercise 80 ms PCM transport, partial reply text,
discarding a completed but unaccepted reply, accepting two turns, cancelling a
generating reply, and clearing/restarting the conversation. Controller/widget
tests use fake devices and cover recording, send/reply, mode exclusion,
acknowledgement races, playback cancellation, background interruption, no
automatic resume and visible history reset.

## Two-turn real-model smoke run

The input questions were synthesized locally with the existing Q4 TTS tool,
then resampled to mono 16 kHz f32le. The voice session generated replies
directly from these audio inputs:

1. `What is two plus two?`
2. `And what is that number plus three?`

| Backbone | Turn | Reply text | Audio frames | Audio seconds | History slots |
|---|---:|---|---:|---:|---:|
| CPU | 1 | It's four. | 23 | 1.84 | 76 |
| CPU | 2 | It's seven. | 25 | 2.00 | 148 |
| Metal | 1 | It's four. | 15 | 1.20 | 68 |
| Metal | 2 | That's seven. | 25 | 2.00 | 140 |

The CPU output WAVs were resampled to 16 kHz and passed through existing ASR.
The recognized text was exactly `It's four.` and `It's seven.`. This checks two
short generated clips; it does not establish general pronunciation or voice
quality. No listening or playback was performed. The follow-up answers use
the previous audio turn as context; the follow-up contains no standalone
reference to the number four.

After the second CPU turn, retained committed embeddings occupied 851,968
bytes; after the second Metal turn, 786,432 bytes. These figures exclude model
weights, KV state, pending input/output and scratch. History is bounded by the
2048-slot context, and overflow asks the user to start a new conversation.
The preview reserves 768 reply slots and limits each recording and reply speech
to 30 seconds. It never silently evicts earlier turns.

Dart sampling seeds differ from Torch seeds. CPU and Metal reply frames/text
also differ, as shown above; no cross-backend sampled-output identity is
claimed. Existing text lab dumps, rather than sampled conversation outputs,
are the byte-identical arithmetic regression check.

## Reproduction

```sh
python tool/audio_conversation_oracle.py
DIANOIA_AUDIO_CONVERSATION_DIR=build/audio-reference \
DIANOIA_AUDIO_CONVERSATION_FIRST=build/audio-reference/conversation-question.f32 \
DIANOIA_AUDIO_CONVERSATION_FOLLOWUP=build/audio-reference/conversation-followup.f32 \
  dart test test/audio_conversation_test.dart

dart run tool/audio_q4_converse.dart build/audio-reference build/voice-replies \
  build/audio-reference/conversation-question.f32 \
  build/audio-reference/conversation-followup.f32
# Append --gpu for a supported GPU backbone.
```

For worker transport, set the same three environment variables to absolute
paths and run `flutter test test/audio_conversation_integration_test.dart` from
`example`. These tools/tests never open audio devices.

## Manual checks remaining

Physical microphone capture, playback quality, routing and OS interruption
behavior remain manual checks. iOS/Android/Windows builds were not run here.
There is no full-duplex listening/talking, voice activity detection, automatic
microphone restart, streamed device playback, persisted conversation history,
live ASR or audio tool-call execution in this preview.

Open **LFM3-Audio-2B Q4_0**, load the model and select **Voice conversation
(English)** under **Audio mode**. Use **Record question**, then **Stop and
reply**. Record again for a follow-up. Check **Stop reply**, backgrounding,
route exit and **New conversation**. An unaccepted/cancelled answer does not
enter history; stopping playback of an already accepted answer keeps the
visible turn for the next question.
