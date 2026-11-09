# Live translation validation

Date: 2026-11-07 (Asia/Tokyo). Host: macOS arm64 (Apple silicon).

## Scope

The audio preview's **Live translation (English)** mode feeds the English
LiveASR transcript, sentence by sentence, to a text-only copy of the same
LFM3-Audio-2B Q4_0 language backbone and shows the translations as text. No
speech is generated and no extra weights are downloaded. The bundle was
prepared with `dart run tool/prepare_audio_q4.dart build/audio-reference`.

## Design checks

- The translator cannot share the LiveASR model: `AudioLiveSession` rejects
  any change to its language state between blocks, and rolling contexts reset
  it. A second `Lfm3Model` on the same `language/` directory is used instead.
- The Metal backend keeps one global command buffer, so a second GPU user in
  another isolate is unsafe. The translator runs on the CPU kernels
  (`InferenceBackend.cpu`, 4 threads, 512-token context) in its own isolate.
- A bare system prompt ("Translate the user's English text…") let
  LFM3-1.7B answer or echo some sentences instead of translating them. Putting
  the instruction in the user message as well fixed every probe sentence on
  both LFM3-1.7B and the Audio backbone.
- The Audio backbone once emitted `</think>` and repeated its answer.
  `cleanTranslation` keeps the text after the last `</think>`.

## End-to-end run

A five-sentence English clip was synthesized with macOS `say` (Samantha),
converted to mono 16 kHz float32 (13.3 s) and streamed to the audio worker in
80 ms chunks at real time, while `LiveTranslator` translated into Japanese:

| LiveASR | ASR lag at clip end | First translation | All translated |
|---|---|---|---|
| GPU (Metal) | 0.00 s | 6.2 s | 14.6 s |
| CPU | 0.00 s | 6.4 s | 14.8 s |

Recognized sentences and translations (identical on both runs):

| Transcript | Translation |
|---|---|
| Good morning, everyone, thanks for joining today's meeting. | 皆さん、こんにちは。今日の会議にご参加いただきありがとうございます。 |
| We ship the new build last night, and so far the crash rate looks much lower. | 昨日、新しいビルドを配達しましたが、今のところクラッシュ率は大幅に低くなっています。 |
| Could you send me the slides after the call? | 会議の後、スライドを送っていただけますか？ |
| I think we should wait until Monday before we decide. | 月曜日まで待ってから決めましょう。 |

"Ship" for "shipped" is a recognition error; "配達" (deliver a parcel) for
shipping a build is a translation error a larger model would likely avoid.

## Reproduction

```sh
say -v Samantha -o build/meeting.aiff "Good morning, everyone. ..."
afconvert -f caff -d LEF32@16000 -c 1 build/meeting.aiff build/meeting.caf
# Extract the CAF data chunk as raw float32 into build/meeting.f32.
cd example
DIANOIA_AUDIO_EXAMPLE_DIR=$PWD/../build/audio-reference \
DIANOIA_TRANSLATION_EXAMPLE_PCM=$PWD/../build/meeting.f32 \
DIANOIA_AUDIO_EXAMPLE_GPU=1 \
  flutter test test/live_translation_integration_test.dart
DIANOIA_AUDIO_EXAMPLE_DIR=$PWD/../build/audio-reference \
  flutter test test/live_translation_test.dart
```

## Not validated

Physical microphone capture, phone throughput (where CPU translation will be
slower and may fall further behind), memory use on phones, the Vulkan backend,
and translation targets other than Japanese.
