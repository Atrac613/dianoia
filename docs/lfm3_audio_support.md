# LFM3-Audio-2B support

## Current scope

The research path now transcribes a complete audio clip with the official
**Q4_0 language GGUF and Q4_0 audio mmproj GGUF**. It includes the log-mel
frontend, LFM2 encoder, projector, ASR prompt expansion and greedy text decoding.
CPU and Metal language inference have been exercised; the audio encoder always
runs on the CPU. This is not yet an app API, an audio chat session or a
`ModelStore` preset. Speech generation and microphone UI remain unimplemented.

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
```

Preparation resumes and hash-verifies downloads, validates the encoder layout,
and publishes the imported language directory after conversion succeeds. It
refuses to overwrite an existing language bundle. The receipt records original
source hashes for research provenance; it is not an app installation receipt.
The research runner expects prepared files and does not rehash them every run.
The runner accepts clips up to 30 seconds and emits JSON with text, token IDs,
timings and the stop reason. Omitting the language asks the model to infer it.

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

## Remaining implementation

1. Connect the streaming frontend/encoder to incremental language prefill and
   validate resulting transcripts before adding microphone input.
2. Broaden real speech/noise/language validation beyond the two clean synthetic
   English/Japanese clips, and evaluate any activation quantization changes.
3. Implement the noncausal three-layer Depthformer, 12-codebook parallel
   prediction and LiquidCodec streaming decoder for 24 kHz speech output.
4. Add pinned app installation bundles, public sessions, cancellation and resource
   limits, then the example UI and CPU/Metal/Vulkan device validation.

The official [model card](https://huggingface.co/LiquidAI/LFM3-Audio-2B)
describes text/audio input, text/speech output and multilingual transcription
and synthesis. Full upstream feature coverage remains a target; the current
dianoia subset is complete-clip transcription.
No weights are downloaded by the frontend or default Dart tests;
only the explicitly invoked preparation tool downloads the pinned models.
