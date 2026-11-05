# LFM3-Omni-2B support

## Current scope

Phase 1 is a research path for the official **Q4_0 language GGUF and Q4_0
mmproj GGUF**. With one prepared directory it covers:

- image question answering (CPU float32 vision tower, CPU or Metal language);
- clip and streaming transcription (ASR);
- speech synthesis (TTS);
- interleaved English voice conversation.

All of these reuse the LFM3-VL and LFM3-Audio implementations. Not yet
implemented: prompts that mix images and audio (for example a spoken question
about a photo), video input, LiveASR, `ModelStore` installation, the Flutter
example and the app API.

## What Omni shares with Audio and VL

Sources inspected on 2026-11-05:
[LiquidAI/LFM3-Omni-2B@21d7512](https://huggingface.co/LiquidAI/LFM3-Omni-2B/tree/21d75123f86c1f7ffb4bef1ea202710c4e7ebe67)
and
[LiquidAI/LFM3-Omni-2B-GGUF@d508b88](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/tree/d508b88f08a5b4286f2d2805a6a8d69a0ff44c72).

| Part | Relation to existing support |
|---|---|
| Language GGUF (1.05 GB) | Same tensor names, shapes and types as LFM3-Audio Q4_0 |
| Audio encoder, Depthformer, LiquidCodec (`a.*`, `mm.a.*`) | Same layout as the LFM3-Audio mmproj |
| Vision tower and projector (`v.*`, `mm.0`, `mm.2`, `mm.input_norm`) | LFM3-VL layout, with Q4_0 matrices instead of BF16 |
| Feature extractor | Same parameters as LFM3-Audio |
| Tokenizer | Byte-identical to LFM3-Audio's `tokenizer.json` |

The weights themselves differ from both siblings. Omni ships one 270 MB mmproj
that holds both the vision and the audio towers.

## Prompt differences

`tool/omni_prompt_oracle.py` renders the pinned `chat_template.jinja` as data
(jinja2 and the pinned tokenizer only; none of the checkpoint's Python runs).
It writes `test/fixtures/omni/prompt_oracle.json`. Compared with LFM3-Audio:

- **Input audio is wrapped** as `<|vision_start|><|audio|>…<|vision_end|>`, the
  same span markers images use (`audio_markers: true` in the processor).
- **Every assistant turn opens with `<think></think>`.** Omni is a
  non-reasoning model and its template defaults `enable_thinking` to false.
  TTS appends `<|audio_start|>` after that, and conversation appends
  `<|mixed_start|>`.
- **Text mode also suppresses `<|image|>` and `<|video|>`**, as upstream
  `_placeholder_token_ids` does.

`Lfm3SpeechVariant` (`audio` or `omni`) holds these conventions. The encoder
config and the generator detect it from the mmproj: an Omni mmproj carries an
`lfm3vl` vision projector. `AudioSynthesizer` and `AudioConversationSession`
follow `generator.variant`. ASR callers pass `encoder.config.variant` to
`prepareAudioTranscriptionPrompt` or `AudioTranscriptionPrefill.start`. The
default stays `audio`, so existing callers are unchanged.

## Speech sampling

Each mmproj stores its official sampling defaults under
`clip.gen.audio.sampling.*`. `Lfm3SpeechSampling.fromGguf` reads them, and
`Lfm3AudioGenerator.sampleFrame` uses them when an option is omitted:

| | temperature | top-k | penalty | penalty window |
|---|---:|---:|---:|---:|
| LFM3-Audio | 1.0 | 128 | 1.1 | 4096 frames |
| LFM3-Omni | 0.4 | all codes | 1.2 | 64 frames |

These match `generation_config.json` and `sample_audio_frame` in
`modeling_lfm3_omni.py`. The MaskGIT schedule is still three passes of four
codebooks. Audio's 4096-frame window is longer than any reply, so LFM3-Audio
output does not change.

## Vision tower from the mmproj

`loadVisionEncoderGguf` reads the vision and projector tensors directly from
the mmproj and widens F32, F16, BF16 and Q4_0 payloads to float32 exactly
(`dequantizeQ4_0` follows ggml's `dequantize_row_q4_0`). It writes no
intermediate safetensors file. Audio tensors are ignored, and every vision
tensor must be present. `Lfm3VlConfig.fromOmniJson` parses Omni's
`config.json`. That file does not name the span marker IDs, so
`loadOmniVisionConfig` takes them from the tokenizer.

The vision tower is still the correctness-first CPU float32 reference. On an
Apple silicon Mac, a 640×480 image (300 tokens) took about 72 s to encode, so
this path is research-only until the encoder uses native Q4 matmuls.

## Research commands

From the repository root (allow about 2.6 GB of disk space):

```sh
dart run tool/prepare_omni_q4.dart build/omni-reference
# Image question answering:
dart run tool/omni_q4_image_chat.dart build/omni-reference photo.png "Describe this image." --gpu
# The audio tools accept an Omni directory; mono 16 kHz f32le PCM input:
dart run tool/audio_q4_transcribe.dart build/omni-reference speech.f32 Japanese --gpu
dart run tool/audio_q4_tts.dart build/omni-reference "Hello" out.wav --gpu
dart run tool/audio_q4_converse.dart build/omni-reference out-dir q1.f32 q2.f32 --gpu
```

Preparation verifies the pinned sizes and SHA-256 hashes, validates the audio
encoder, generator and vision tensors in the mmproj, and only then converts
the language GGUF. `speechMmprojIn` picks the Omni mmproj when one is present.
`audio_q4_live_transcribe.dart` refuses an Omni directory, because the Omni
GGUF publishes no live-ASR template (`tokenizer.chat_template.asr`).

## Validation so far

These smoke checks ran on Apple silicon with Metal language inference:

- ASR on the LFM3-Audio English and Japanese reference clips gives the same
  text as LFM3-Audio, in batch and in 80 ms streaming mode.
- TTS stops on EOS in English (3.5 s) and Japanese (3.4 s), and transcribing
  the speech back returns the input text exactly.
- A two-turn conversation ("two plus two", then "add three") gives correct,
  context-aware replies, and their speech transcribes back to the reply text.
- A synthetic 640×480 image with text and shapes is described correctly,
  including its "OMNI 2026" caption.

Unit tests cover the prompts against the oracle, Omni config parsing, sampling
metadata, Q4_0 widening, and equivalence of mmproj vision loading with the
existing import.

The graph code was compared with the sources as well as the shapes. After
renaming `Lfm3Omni*` to `Lfm3Audio*` and stripping comments and docstrings,
`Encoder`, `Downsampler`, `Projector`, `DepthDecoder`, `CodebookEmbedding` and
`RMSNorm` in `modeling_lfm3_omni.py` are identical to the pinned
`modeling_lfm3_audio.py`. No numerical oracle has been run against the Omni
PyTorch weights yet: the BF16 checkpoint is 4.3 GB.
