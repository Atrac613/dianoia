# LFM3-Audio in-app download validation

Date: 2026-11-02. Host: macOS arm64. Scope: paired official Q4_0 installation
through `ModelStore` and the Flutter audio preview.

## Sources and storage

`AudioGgufSpec.lfm3AudioQ4` reuses all five pins from
`Lfm3AudioQ4Source`: the language/audio GGUF pair at revision
`02355c1a141e464a064f2ab2c276638dbcb7f38a` and config/tokenizer/license at
metadata revision `0295e4130594c06800fd25451d97139ac2da1aff`.
Total download: **1,228,932,715 bytes**. License: LFM Open License v1.0.

Installed payloads are approximately 1.30 GB. The UI recommends 4 GB free
during setup. Downloads live beside unpublished `bundle.partial` staging.
Successful conversion, copying and manifest hashing finish before the staging
directory is renamed to `bundle`. Source downloads are removed afterwards.
Conversion keeps Q4 blocks and the existing Q6_K-to-Q8 embedding/head adapter.

The manifest records the original pins and all runtime hashes/sizes. It also
anchors retained original payloads to their published hashes. The worker
verifies all installed payload hashes before loading, in addition to the
existing source/config/cache checks. Existing prepared research folders remain
accepted without a managed-install manifest.

## Automated checks

| Check | Result |
|---|---|
| `dart format .` | Passed |
| Root `dart analyze`, example `flutter analyze` | No issues |
| Root `dart test --reporter expanded` | 291 passed, 28 skipped |
| Example `flutter test --reporter expanded` | 55 passed, 3 skipped |
| Real-model HTTP download/worker integration | 2 passed, no skips |
| `flutter build macos --debug` | Passed; app was not launched |
| `tool/lab_parity.sh` | 24 byte-identical dumps for chat/long, CPU Q8 and Metal Q8/Q4 |

The four root installer tests use tiny runtime-generated language/audio GGUFs
served over loopback. They cover real HTTP cancellation and Range resume,
concurrent-install rejection, conversion cancellation, stale staging removal,
hash/graph rejection, installed-cache tampering, runtime loading, successful
download cleanup and mirror pin preservation. No checkpoint weights are stored
in fixtures.

Controller/widget tests cover explicit download consent, source/license/storage
display, progress, cancellation, rejection of a late completion, exclusion of
capture/TTS during setup, automatic loading after setup and discovering an
existing installation without another download. A worker test leaves an HTTP
response body pending and then issues cancel followed immediately by close,
checking that both finish before the next model-worker lease is released.

## Official model integration

```sh
cd example
DIANOIA_AUDIO_DOWNLOAD_SOURCE=/absolute/path/to/build/audio-reference \
  flutter test test/audio_download_integration_test.dart --reporter expanded
```

The five already available official source files were served over loopback;
no files were fetched again from Hugging Face. A preseeded 1 MiB language part
was resumed with `Range: bytes=1048576-`. The worker received five HTTP
requests, verified all pins, converted and atomically published the bundle,
recognized the installation, checked the installed hashes and loaded the CPU
backbone. It then generated a nonempty **1.12-second, 24 kHz PCM16 WAV** from
`Hello world.`. Source downloads were gone after installation. The run took
39,880 ms including local transfer, preparation, load and synthesis; this is
a smoke result, not a network or device latency benchmark. Test-only model
copies were removed at completion.

No app launch, microphone operation, listening or audio-device playback was
performed. Physical-device installation and iOS/Android/Windows builds remain
unvalidated in this change. A model with an invalid installed manifest is
rejected and must be removed before replacement; it is not silently overwritten.

## Manual path

Open **LFM3-Audio-2B Q4_0** on the root screen and tap **Download**. Check the
progress, cancel/resume, automatic readiness after setup and reuse after leaving
and reopening the screen. Recording and **Generate and play** become available
after loading. Backgrounding cancels active work; returning does not restart
downloads, capture or playback. Setup cancellation waits for an in-flight
conversion and keeps downloads for the next attempt.
