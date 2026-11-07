# LFM3-Omni-2B support

## Current scope

Experimental support for the official **Q4_0 language GGUF and Q4_0 mmproj
GGUF**, through `lab.dart`, research tools and the Flutter example:

- one turn mixing text, images, audio clips and a video with its soundtrack,
  answered in text, speech or interleaved text and speech
  (`prepareOmniPrompt`, `decodeSpeechReply`);
- clip and streaming transcription (ASR), speech synthesis (TTS) and
  interleaved English voice conversation, through the LFM3-Audio APIs;
- installation with `ModelStore` (`OmniGgufSpec.lfm3OmniQ4`);
- an **Ask about a photo** mode in the example's Omni preview.

Not implemented: multi-turn history for mixed media prompts, LiveASR (Omni
publishes no live template), video in the example app, and a stable app API.

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

By default Q4_0 linear weights stay native Q4 blocks, multiplied by the native
kernels with float inputs; `nativeQ4: false` widens everything to float32
instead. Attention runs per head as batched matrix products. On an Apple
silicon Mac, a 640×480 image (300 tokens, 1200 patches) now encodes in about
9.5 s instead of 72 s, and the projected rows differ from the all-float32
encoder by a relative L2 of 3e-6. The VL vision tower shares the faster
attention.

## Mixed prompts

`renderOmniChat` renders the official template for content parts
`{'type': 'image'}`, `{'type': 'audio'}` and `{'type': 'video'}`, and appends
the reply marker for `AudioReplyMode.text`, `audio` or `mixed`.
`prepareOmniPrompt` then expands the placeholders as `processing_lfm3_omni.py`
does:

- an image or audio clip repeats its placeholder once per encoder row inside
  one `<|vision_start|>`…`<|vision_end|>` span;
- a video becomes, per frame group, `<t seconds>`, the group's span-wrapped
  video rows and, when an `OmniVideoLayout` declares a soundtrack, that
  group's share of the soundtrack's audio rows, all inside one outer span.
  A soundtrack row is 80 ms; each group's share starts at the row nearest its
  first frame, as upstream `_video_expansion` computes it. Times use Python's
  `.1f` formatting, ties to even (`formatOmniSeconds`).

`OmniPrompt.prefill` validates and copies every row set before resetting the
model, then feeds the prompt in chunks. `decodeSpeechReply` decodes the
reply: greedy text, sampled speech frames and the six/twelve interleaving
schedule. It is the same loop `AudioConversationSession` now uses.

## Video

`Lfm3VideoProcessor` follows the LFM3-VL video processor as Omni configures
it: frames sampled at 2 fps (4 to 384), indices as `numpy.linspace(...).round()`,
one size for every frame from the whole-clip pixel budget (`smart_resize` with
factor 32 and temporal factor 2), and frames paired into temporal patches,
the odd last frame repeated. The vision tower attends within each frame group
only (upstream `cu_seqlens` per frame), so each group encodes like an image
whose two temporal slots hold two frames. Decoding the container is the
caller's job; the research tool uses ffmpeg.

## Research commands

From the repository root (allow about 2.6 GB of disk space):

```sh
dart run tool/prepare_omni_q4.dart build/omni-reference
# Images, audio clips (mono 16 kHz f32le), video and text, in one turn:
dart run tool/omni_q4_chat.dart build/omni-reference --image=photo.png \
  --text="Describe this image." --gpu
dart run tool/omni_q4_chat.dart build/omni-reference --image=photo.png \
  --audio=question.f32 --mode=mixed --out=reply.wav --gpu
dart run tool/omni_q4_chat.dart build/omni-reference --video=clip.mp4 \
  --soundtrack --system="Answer questions about the video using what you see and hear." \
  --text="What happens, and what can you hear?" --gpu
# The audio tools accept an Omni directory:
dart run tool/audio_q4_transcribe.dart build/omni-reference speech.f32 Japanese --gpu
dart run tool/audio_q4_tts.dart build/omni-reference "Hello" out.wav --gpu
dart run tool/audio_q4_converse.dart build/omni-reference out-dir q1.f32 q2.f32 --gpu
```

`--max-image-tokens` and `--max-video-pixels` bound the vision work. An
installed `OmniGgufSpec` bundle has the same layout and works with the same
commands.

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
- A spoken question about that image ("What shapes and colors do you see in
  this picture?") is answered correctly in text, and in interleaved mode
  with 11.3 s of speech that transcribes back to the reply text.
- A 4 s synthetic video (a red square sliding across a blue screen, with a
  spoken narration) is described correctly, and the narration is quoted
  exactly.
- `ModelStore` installs the official pair from a local mirror in 31 s into a
  1.41 GB bundle that passes full verification and transcribes correctly.
- The example's worker answers typed and spoken photo questions and
  transcribes, on the real model (`example/test/omni_worker_integration_test.dart`).
- Refactoring the conversation onto `decodeSpeechReply` reproduces the
  earlier Omni two-turn replies byte for byte, and LFM3-Audio TTS output is
  byte-identical to before these changes.

Unit tests cover the prompts against the oracle, mixed and video expansion,
Python-compatible time formatting and frame sampling, Omni config parsing,
sampling metadata, Q4_0 widening, mmproj vision loading, and installation
with synthetic GGUFs over loopback.

The graph code was compared with the sources as well as the shapes. After
renaming `Lfm3Omni*` to `Lfm3Audio*` and stripping comments and docstrings,
`Encoder`, `Downsampler`, `Projector`, `DepthDecoder`, `CodebookEmbedding` and
`RMSNorm` in `modeling_lfm3_omni.py` are identical to the pinned
`modeling_lfm3_audio.py`. No numerical oracle has been run against the Omni
PyTorch weights yet: the BF16 checkpoint is 4.3 GB.
