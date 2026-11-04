# LFM3-Audio example TTS validation

Date: 2026-11-02. Host: macOS arm64. Scope: Flutter example integration of the
existing official Q4_0 text-to-speech engine. Source pins and arithmetic checks
are recorded in [the engine validation](lfm3-audio-tts-2026-11-02.json).

## Behavior

- The audio preview accepts short text, Japanese/English output and a
  female/male voice instruction. Generation completes before playback starts.
- The existing worker owns both ASR and TTS, sharing one language backbone.
  Generation weights load lazily. Recording and TTS cannot overlap.
- Generation is limited to 375 frames (30 seconds). Empty or unfinished
  output is reported as an error rather than played. The input field allows
  up to 1000 characters, subject to the model's context limit.
- Stop, route exit and app backgrounding cancel synthesis and stop playback.
  Late results are discarded; resume never restarts playback or capture.
- The lazy local player uses a private temporary WAV. Completion, stop,
  interruption and playback errors release the source and delete that file.
  The player waits for pending source preparation before finishing a stop,
  preventing a late native start from outliving cancellation.

## Automated checks

| Check | Result |
|---|---|
| `dart format .` | Passed |
| Root `dart analyze` and example `flutter analyze` | No issues |
| Root `dart test --reporter expanded` | 287 passed, 28 skipped |
| Example `flutter test --reporter expanded` | 50 passed, 2 skipped |
| Optional prepared-model worker tests | 3 passed, no skips |
| `flutter build macos --debug` | Passed; app was not launched |
| `tool/lab_parity.sh` | 24 byte-identical dumps across chat/long, CPU Q8 and Metal Q8/Q4 |

The macOS build emitted Swift actor-isolation/capture warnings from
`audioplayers_darwin 6.5.0`; compilation succeeded with the current toolchain.

Controller tests verify exclusion of capture during TTS, option forwarding,
blank-text errors, cancellation before a late response, playback failure and
reuse, and player/worker disposal. Widget tests cover explicit start/stop,
background cancellation and no automatic resume. The player adapter tests use
a fake native player to exercise delayed source preparation, interruptions,
error cleanup and temporary-file deletion. These tests open no audio device.

The optional real worker test uses:

```sh
cd example
DIANOIA_AUDIO_EXAMPLE_DIR=/absolute/path/to/build/audio-reference \
DIANOIA_AUDIO_EXAMPLE_PCM=/absolute/path/to/build/audio-reference/english.f32 \
  flutter test test/audio_worker_integration_test.dart --reporter expanded
```

It verifies the existing English ASR transcript, generates a nonempty 24 kHz
RIFF WAV from `Hello world.`, cancels another synthesis request, then verifies
ASR reuse and worker shutdown. This run used the CPU backbone. It generates
bytes only and never instantiates a playback device.

## Manual checks still needed

No listening, microphone capture, app launch or physical-device testing was
performed. Native playback, audio routing, OS interruptions and switching
between recording and playback remain manual checks. iOS/Android/Windows
builds were not run in this change. No model weights or generated speech were
committed.

To check manually, open **LFM3-Audio-2B Q4_0** with a prepared local model,
enter a short sentence under **Text to speech**, and use **Generate and play**.
Check both output languages and voice instructions, stop during generation
and playback, then record again. Also leave the route or background the app
while it is active and confirm returning does not restart audio.
