# LFM3-Audio-2B support

## Current scope

The research path now transcribes a complete audio clip with the official
**Q4_0 language GGUF and Q4_0 audio mmproj GGUF**. It includes the log-mel
frontend, LFM2 encoder, projector, ASR prompt expansion and greedy text decoding.
CPU and Metal language inference have been exercised; the audio encoder always
runs on the CPU. Chunked PCM input can prefill the language model before the
recording finishes; transcription output starts after finalization. This is
not yet an app API, an audio chat session or a
`ModelStore` preset. The Flutter example now has an experimental microphone
preview for prepared local bundles, including text-to-speech playback. Physical
microphone/device behavior is not yet validated. Text-to-speech is also
available through `lab.dart` and the research CLI, producing mono 24 kHz PCM
or WAV.

## Q4_0 target and research commands

The source is pinned to
[LiquidAI/LFM3-Audio-2B-GGUF revision 02355c1](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF/tree/02355c1a141e464a064f2ab2c276638dbcb7f38a).
`Lfm3AudioQ4Source` records exact sizes and SHA-256 hashes:

| File | Bytes |
|---|---:|
| `LFM3-Audio-2B-Q4_0.gguf` | 1,049,613,856 |
| `mmproj-LFM3-Audio-2B-Q4_0.gguf` | 161,393,408 |

The pair is 1,211,007,264 bytes, plus approximately 18 MB of pinned config,
tokenizer and license metadata. Preparation also writes the imported language
bundle (approximately 1.14 GB). Allow at least 3 GB of working disk space.
These files are named Q4_0, not QAD. Liquid AI states that parts of the Audio
model were trained with QAT for Q4_0 in the
[official discussion](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF/discussions/2).

From the repository root:

```sh
dart run tool/prepare_audio_q4.dart build/audio-reference
# Decode/resample a clip to mono 16 kHz, little-endian float32 PCM first:
ffmpeg -i speech.wav -ac 1 -ar 16000 -f f32le speech.f32
dart run tool/audio_q4_transcribe.dart build/audio-reference speech.f32 Japanese
# Accelerate the language backbone on a supported GPU:
dart run tool/audio_q4_transcribe.dart build/audio-reference speech.f32 Japanese --gpu
# Simulate arriving 80 ms PCM chunks, with incremental language prefill:
dart run tool/audio_q4_transcribe.dart build/audio-reference speech.f32 Japanese --gpu --chunk-ms=80
```

Preparation resumes and hash-verifies downloads, validates the encoder layout,
and publishes the imported language directory after conversion succeeds. It
refuses to overwrite an existing language bundle. The receipt records original
source hashes for research provenance; it is not an app installation receipt.
The research runner expects prepared files and does not rehash them every run.
The runner accepts clips up to 30 seconds and emits JSON with text, token IDs,
timings and the stop reason. Omitting the language asks the model to infer it.
Chunk mode is an unpaced file simulation; it reads a local PCM file and is not
microphone capture or a realtime latency benchmark. Its JSON also records how
many audio rows were prefilled before finish and the peak encoder cache size.

Q4 matrix blocks are copied directly into native Q4 storage, without
dequantization/requantization, in the encoder and language matrices. The
language GGUF's **Q6_K token embedding/head** uses the existing Q8 adapter.
Convolution, bias and norm weights remain F32. The encoder defaults to float
activation inputs, isolating weight quantization in the parity check.

`Lfm3AudioEncoder` validates all consumed tensor shapes/types before allocating
matrices. It loads neither Depthformer nor codec generation tensors. The actual
GGUF encoder FFN width is **1792**, after LFM2's auto-adjustment of the config's
2560; the width comes from GGUF metadata. Its attention uses a 32-token causal
window, full half-split RoPE with theta 1e6, and no LFM3 attention gate or
sandwich norms. Projected rows feed the existing embedding-override path.

`prepareAudioTranscriptionPrompt` implements only single-clip ASR: ChatML with
BOS, `Perform ASR in <language>.`, repeated audio tokens and an assistant
prefix. It suppresses audio placeholders and speech/mixed generation modes,
and stops on both `im_end` and `text_end`, matching upstream text mode.

## Audio frontend

`package:dianoia/lab.dart` exports `Lfm3AudioFrontend` and its independent
stream states. The frontend accepts mono,
finite `Float32List` PCM at **16000 Hz**, with the usual normalized PCM scale;
it rejects other rates rather than silently treating them as 16 kHz. File
decoding, channel mixing and resampling must happen before extraction.

```dart
final frontend = Lfm3AudioFrontend();
final features = frontend.extract(pcm, samplingRate: 16000);
// features.values: [128 mel bins, features.frames], contiguous float32.
// features.encoderTokens: ceil(features.frames / 8).

final stream = frontend.startStream();
final first = stream.add(firstPcmChunk, samplingRate: 16000);
final next = stream.add(nextPcmChunk, samplingRate: 16000);
final tail = stream.finish();
// Concatenate first, next and tail along the frame axis, per mel bin.
```

The stream keeps fewer than 400 unconsumed samples after each successful add.
Incomplete windows wait for the next chunk; only finalization pads with zeros.
An empty clip produces one silent frame, matching the official extractor.
Finalized streams reject further calls. Invalid chunks do not consume state.
These are feature-extraction states, not audio-encoder streaming caches.

The current direct DFT is a correctness reference. It has not been benchmarked
for realtime microphone input. FFT/native acceleration can replace it after
parity is established, while keeping the feature pipeline in Dart.

## Streaming encoder

`Lfm3AudioEncoder.startStream()` shares the loaded Q4 weights and owns separate
downsampler context, two gated-input history rows per short-convolution layer,
and up to 31 key/value rows per attention layer. RoPE positions advance across
calls. The downsampler retains at most 15 feature frames, including the seven
frames of lookahead needed before a row can be emitted. Work and temporary
storage scale with the current chunk; the entire recording is not replayed.

```dart
final frontend = Lfm3AudioFrontend().startStream();
final encoder = Lfm3AudioEncoder.load(mmprojPath);
final stream = encoder.startStream();
for (final pcm in pcmChunks) {
  final rows = stream.add(frontend.add(pcm, samplingRate: 16000));
  // Newly completed projected rows: [row count, encoder.config.projectionSize].
  // A small chunk may emit no rows. Feed nonempty rows to the language model.
}
final lastComplete = stream.add(frontend.finish());
final paddedTail = stream.finish();
encoder.close();
```

The stream releases its caches after finish, close or computation failure.
Invalid input is rejected before advancing state, so it can be retried.
Exceptions from a capture callback terminate the stream, since earlier layers
may already have changed state. Reentrant stream operations are rejected.
Closing the encoder closes all its active streams. Feature and encoder calls
remain synchronous CPU work; microphone capture and language prefill are not
implemented by this class.

## Incremental ASR prefill

`AudioTranscriptionPrefill` connects encoder rows to the language backbone
without retaining or replaying earlier projected rows. Start resets the model
and feeds the system/user prefix. Each `addProjected` feeds new audio rows;
`finish` appends the user-turn end and assistant prefix, leaving `model.logits`
ready for greedy text output. The caller owns the language model and must use
it exclusively throughout input and generation.

```dart
final input = await AudioTranscriptionPrefill.start(
  model: backbone,
  tokenizer: tokenizer,
  hiddenSize: encoder.config.projectionSize,
  language: 'Japanese',
  reservedTokens: 128,
);
final frontend = Lfm3AudioFrontend().startStream();
final stream = encoder.startStream();
await for (final pcm in pcmChunks) {
  await input.addProjected(stream.add(frontend.add(pcm, samplingRate: 16000)));
}
await input.addProjected(stream.add(frontend.finish()));
await input.addProjected(stream.finish());
await input.finish();
// Decode from backbone.logits with input.greedyToken/input.stopTokenIds.
// The caller closes the encoder and backbone when no longer needed.
```

Input validation and context overflow fail before advancing language state;
the context check reserves room for the assistant suffix and requested output
tokens. Concurrent input calls are rejected. A failed forward pass or explicit
close terminates the operation without closing the caller-owned model. A close
cannot interrupt an in-flight kernel: await the pending operation before reusing
the model. No partial transcript is emitted while audio is still arriving.

## Reference and validation

The contract is pinned to official
[LiquidAI/LFM3-Audio-2B revision 0295e413](https://huggingface.co/LiquidAI/LFM3-Audio-2B/tree/0295e4130594c06800fd25451d97139ac2da1aff),
specifically `config.json` and `feature_extraction_lfm3_audio.py`:

- Periodic Hann window, 400 samples, 160-sample hop, no centering.
- Right-padding to the next complete frame, minimum 400 samples.
- Unnormalized one-sided power spectrum, 128 Slaney mel filters with Slaney
  area normalization over 0–8000 Hz.
- Base-10 logarithm clamped to -12, then `(log_mel + 4) / 4`.

`tool/audio_features_oracle.py` downloads and imports the pinned official
extractor, without weights, and generates
`test/fixtures/audio/features_oracle.json`. It records source SHA-256 and Python
library versions. Use a separate environment with
`tool/requirements-vision-validation.txt` to regenerate it:

```sh
python tool/audio_features_oracle.py
dart test test/audio_features_test.dart
```

Tests compare all features against Torch with an absolute tolerance of 2e-5
for deterministic broadband PCM, silence and frame-boundary cases. Streaming
must match Dart batch extraction exactly across multiple chunk sizes, including
single-sample chunks. The fixture does not establish model-level accuracy.

`tool/audio_encoder_oracle.py` extracts the pinned official downsampler,
encoder and projector classes and uses Transformers' LFM2 implementation.
The default fixture uses tiny deterministic synthetic Q4 blocks; Dart compares
each stage, including an attention-window boundary, at 3e-5 absolute/relative
tolerance. No checkpoint weights are stored in fixtures. For real validation:

```sh
python tool/audio_encoder_oracle.py \
  --gguf build/audio-reference/mmproj-LFM3-Audio-2B-Q4_0.gguf \
  --gguf-python /path/to/llama.cpp/gguf-py \
  --out build/audio-reference/encoder_oracle.json
DIANOIA_AUDIO_Q4_MMPROJ=build/audio-reference/mmproj-LFM3-Audio-2B-Q4_0.gguf \
DIANOIA_AUDIO_ENCODER_ORACLE=build/audio-reference/encoder_oracle.json \
  dart test test/audio_encoder_test.dart
```

The real oracle widens the exact published Q4 blocks with gguf-py and runs
the upstream Torch graph. It compares 1, 8, 9, 33 and 265 input frames, including
34 encoder tokens (beyond the 32-token window), at every layer, the final norm
and projector, with tolerance `2e-4 * (1 + abs(reference))`.
`tool/audio_prompt_oracle.py` independently renders the pinned official ASR
template. Dart unit tests also check malformed layouts, Q4 scale/nibble order,
buffer ownership, invalid inputs, lifecycle and generation-mode suppression.

The encoder oracle tool also writes `encoder_stream_oracle.json` beside its
batch output, using the pinned upstream encoder's actual streaming method.
It tests zero-length chunks, short inputs, exact/downsampler boundaries, an
irregular chunk schedule and the final flush. Dart additionally compares every
streamed stage with batch execution across seven chunk sizes. Set
`DIANOIA_AUDIO_ENCODER_STREAM_ORACLE` alongside `DIANOIA_AUDIO_Q4_MMPROJ` to run
the optional real-GGUF streaming comparison. Its 265-frame case crosses the
32-token attention window. The tolerance is the same as for real batch parity;
different GEMM batch sizes need not produce bit-identical floating-point rows.

Streaming encoder validation on macOS arm64 (2026-11-02): 272 default tests
passed, 27 optional/environment tests skipped; all 13 streaming tests passed
with the real Q4 oracle enabled. Format and analysis passed. Existing text
`chat`/`long` CPU Q8 and Metal Q8/Q4 dumps remained byte-identical to the lab.

See [the Q4 validation record](validation/lfm3-audio-q4-2026-11-02.json) for
the exact sources, commands, transcripts and limitations.

The [streaming/prefill validation record](validation/lfm3-audio-stream-2026-11-02.json)
records 281 default tests passed (27 skipped), 19 optional real-GGUF tests
passed, and unchanged byte-identical text lab parity. The English and Japanese
80 ms simulations produced the same token IDs as batch on CPU and Metal.
Before finalization, 33/34 English and 52/53 Japanese audio tokens had already
been prefilled. Peak encoder cache storage was 735,232 bytes, within the
735,744-byte bound; this excludes shared weights, per-chunk scratch and the
language KV context. These two synthesized clips do not establish general ASR
accuracy or microphone latency.

## Text-to-speech

`Lfm3AudioGenerator` loads the generation tensors from the same official Q4_0
mmproj. It implements the noncausal three-layer Depthformer, three MaskGIT
passes over 12 codebooks, per-codebook repetition penalties and the published
feedback embeddings. `LiquidCodecStream` decodes frame-major codes using the
FSQ table, sixfold upsampling, causal LFM3 blocks with a 30-position window and
1280-point inverse FFT/overlap-add. Generation and codec arithmetic run on CPU;
`--gpu` accelerates only the language backbone. Q4 blocks remain unchanged.

```sh
dart run tool/audio_q4_tts.dart build/audio-reference \
  "Hello world." build/audio-reference/hello.wav --language=en
dart run tool/audio_q4_tts.dart build/audio-reference \
  "こんにちは。今日は良い天気ですね。" build/audio-reference/hello-ja.wav \
  --language=ja --voice=female --gpu
```

The command writes a mono 24 kHz PCM16 WAV and JSON with frames, duration,
stop reason, peak and clipping count. It never opens an audio device. Use
`--voice=male` for the other voice instruction, `--seed=0` for a Dart random
seed, `--temperature=0` for greedy code selection or `--max-frames=375` to bound
output to 30 seconds. The hard maximum is 750 frames (60 seconds), subject to
language context capacity. The CLI returns exit code 2 for a length-limited
partial WAV; this must not be mistaken for a complete utterance. Ctrl-C cancels
generation between operations. Preparation is shared with ASR.

For worker-isolate integration, `AudioSynthesizer` owns one operation at a time
but does not own the supplied model or generator:

```dart
final generator = Lfm3AudioGenerator.load(mmprojPath);
final synth = AudioSynthesizer(backbone, tokenizer, generator);
final result = await synth.synthesize(
  'Hello world.',
  language: 'en',
  voice: 'female',
  onAudio: (pcm) { /* Owned mono 24 kHz chunks, including the final tail. */ },
);
final wav = encodeSpeechWav(result.samples, sampleRate: result.sampleRate);
// result.stopReason is 'eos' or 'length'.
// synth.cancel() interrupts the operation at its next cancellation point.
// Close generator and backbone after all operations have completed.
```

Use the backbone exclusively during synthesis. The model's borrowed
`lastHiddenState` is the final-normalized state used to condition Depthformer;
it is invalidated by reset or a forward pass without logits. PCM chunks are
copied for callbacks so the caller cannot mutate the collected result.
Cancellation throws `AudioSynthesisCancelled` and closes codec state; already
emitted callback chunks remain the caller's responsibility. Each codec stream
retains bounded convolution/KV history and a 960-sample overlap tail. The
synthesizer retains output PCM and code history up to `maxFrames`.

`tool/audio_generation_oracle.py` imports the pinned official Depthformer and
LiquidCodec graphs. Its default fixture contains only synthetic Q4/F32 weights.
Tests compare depth input, all three layers, normalized embedding, logits,
feedback and complete/streamed waveforms at `3e-5 * (1 + abs(reference))`.
Codec cases include six frames (36 internal positions), crossing the causal
attention window. For optional real-weight comparison:

```sh
python tool/audio_generation_oracle.py \
  --gguf build/audio-reference/mmproj-LFM3-Audio-2B-Q4_0.gguf \
  --gguf-python /path/to/llama.cpp/gguf-py \
  --out build/audio-reference/generation_oracle.json
DIANOIA_AUDIO_Q4_MMPROJ=build/audio-reference/mmproj-LFM3-Audio-2B-Q4_0.gguf \
DIANOIA_AUDIO_GENERATION_ORACLE=build/audio-reference/generation_oracle.json \
  dart test test/audio_generation_test.dart
```

The real-weight comparison passes at `3e-4 * (1 + abs(reference))`.
The [TTS validation record](validation/lfm3-audio-tts-2026-11-02.json) records
English and Japanese WAV generation ending at EOS, exact ASR round trips,
a Metal language-backbone smoke run and the numerical checks. ASR round trips
are a limited intelligibility check, not listening or voice-quality evaluation.
Dart and Torch RNGs differ; matching seeds do not promise identical code
sequences. CPU and Metal may produce different samples under stochastic
sampling. No weights or generated speech are included in test fixtures.

## Flutter speech preview

The example's audio screen accepts short Japanese or English text and a
female/male voice instruction. **Generate and play** uses the existing worker
and loaded language backbone, lazily loads generation weights, and returns a
complete 24 kHz PCM16 WAV before playback. The worker rejects empty or
length-limited speech rather than playing it as a complete utterance. The
30-second output limit and 1000-character input field bound each request;
context capacity can reject input earlier.

**Stop speech**, backgrounding or route exit cancel generation and stop
playback. Recording and TTS cannot overlap. A stopped operation cannot start
playback when its late result arrives, and resume never restarts it. The local
player uses a private temporary WAV, removed after completion, interruption,
stop or error. Recordings remain unsaved. Tests inject a fake player and never
open an audio device. See the [example guide](../example/README.md) for setup
and the [example TTS validation record](validation/lfm3-audio-example-tts-2026-11-02.md).

## Remaining implementation

1. Validate the example's microphone stream and OS audio-session behavior on
   real devices, including interruptions, permission denial and backgrounding.
   The preview implementation and simulated lifecycle tests do not establish
   actual hardware capture quality or latency.
2. Broaden real speech/noise/language validation beyond the clean synthetic
   English/Japanese clips, and evaluate pronunciation and voice quality by
   listening. No manual listening or device playback was performed here.
3. Integrate pinned app installation bundles and public audio sessions. Broaden
   CPU/Metal/Vulkan device validation. The example preview currently uses local
   prepared bundles and its own worker/capture controller.

The official [model card](https://huggingface.co/LiquidAI/LFM3-Audio-2B)
describes text/audio input, text/speech output and multilingual transcription
and synthesis. Full upstream feature coverage remains a target; the current
dianoia subset covers single-clip transcription and text-to-speech.
No weights are downloaded by the frontend or default Dart tests;
only the explicitly invoked preparation tool downloads the pinned models.
