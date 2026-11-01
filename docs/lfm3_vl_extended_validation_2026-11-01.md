# LFM3-VL natural-image, OCR and Japanese validation

This follow-up to the [initial color/spatial checks](lfm3_vl_validation_2026-11-01.md)
uses one natural photograph, English and Japanese questions, and two generated
OCR cards. All four prompts match the official implementation's generated token
IDs on both CPU float32 and CPU-vision/Q8-Metal-language inference. The eight
numerical comparisons pass the previously recorded bounds without changes.

## Inputs and observed answers

| Case | Input and question | Upstream, CPU F32 and Q8/Metal answer |
|---|---|---|
| English photograph | Astronaut photograph; “What is the person wearing? Answer in a short phrase.” | `Suit` |
| Japanese photograph | Same photograph; “この人物は何を着ていますか。日本語で短く答えてください。” | `宇宙飛行士の服を着ています。` |
| English OCR | White card with `CODE` above `AB42`; return the code only | `AB42` |
| Japanese OCR | White card with `東京 123`; transcribe the text | `東京 123` |

The photograph's English answer is generic: it says “Suit” rather than identifying
a spacesuit specifically. The Japanese answer describes astronaut clothing.
These are observed responses, not a broad object-recognition accuracy score.
For both OCR cards, the complete output matches the known rendered text exactly,
including the Japanese card's space. Output agreement and OCR text correctness
are recorded separately in the [machine-readable results](validation/lfm3-vl-extended-2026-11-01.json).

## Numerical results

| Case | CPU prefill logits max absolute error | Q8/Metal prefill logits max absolute error |
|---|---:|---:|
| English photograph | 0.0000610 | 0.1682 |
| Japanese photograph | 0.0000629 | 0.2096 |
| English OCR | 0.0000486 | 0.2033 |
| Japanese OCR | 0.0000331 | 0.1313 |

Normalized patch bytes, prompt IDs and generated IDs are identical in all eight
comparisons. Every vision stage, projector output, prefill logits and decode
logits is compared element by element. The float32 bound remains
`abs(error) <= 0.002 + 0.0002 * abs(reference)`. Q8/Metal keeps the same vision
bound and uses a logit-only absolute bound of 0.5, with zero relative tolerance.
The vision encoder remains CPU float32 in both dianoia backends.

The setup is unchanged: the pinned `ba44bd51db20a2753f28c273a1571ccaa210d954`
checkpoint on an Apple M2 Max with 64 GB RAM, Dart 3.13.5 AOT, PyTorch 2.15.0,
Transformers 5.21.0 and torchvision 0.29.2. The checkpoint SHA-256 was rechecked
and matches the previously recorded upstream hash. These runs disable thinking,
use greedy decoding, allow 32 generated tokens and use a 512-token dianoia
context. Single-run timings are retained as diagnostics, not performance claims.

## Reproducing the input images

The source photograph is the pinned
[scikit-image astronaut PNG](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/astronaut.png).
Its [upstream description](https://github.com/scikit-image/scikit-image/blob/v0.19.3/skimage/data/_fetchers.py#L405-L424)
records the NASA origin and public-domain status. Source SHA-256:
`88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5`.
The preparation script verifies that hash and makes a 256x256 RGB bicubic
thumbnail. It also renders the two 384x192 OCR cards.

Use Pillow 13.0.0 from `tool/requirements-vision-validation.txt` and provide a
local Japanese-capable font. The recorded run uses Hiragino Kaku Gothic W3;
the font name and SHA-256 are in the results. Font files and image assets are
not committed. A different font intentionally produces different image hashes.

```sh
python tool/vision_validation_cases.py \
  --astronaut=/path/to/astronaut-source.png \
  --font=/path/to/Japanese-capable-font.ttc \
  --out=build/vl-validation/extended-inputs
```

The resulting `cases.json` contains complete prompts, dimensions and decoded RGB
hashes. Regenerating the four inputs with the recorded font produced identical
RGB hashes to those used in inference. Follow the reference, `vision-run` and
comparison commands from the [initial report](lfm3_vl_validation_2026-11-01.md),
using each case's prompt and dimensions, `--max-new=32`, `--max-seq=512` for
dianoia, and a fresh output directory per run. Q8/Metal comparisons retain
`--logits-atol=0.5 --logits-rtol=0`.

These cases cover a single photograph and clean, large-font OCR. They do not
establish behavior on varied photographs, small or handwritten text, documents,
multi-image prompts or physical phones. No inference change was needed for the
observed cases. Broader evaluation and the public image-chat API remain next.
