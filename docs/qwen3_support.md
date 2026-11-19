# Qwen3 support

dianoia runs Qwen3 dense checkpoints (`model_type: qwen3`) for text, tool
calling and thinking, on the CPU kernels and on Metal or Vulkan. It is the
engine's second architecture after LFM3, and the first one imported from
upstream BF16 weights rather than a prequantized GGUF.

## What is supported

| | |
|---|---|
| Checkpoints | Qwen3 dense with tied embeddings: Qwen3-0.6B and Qwen3-1.7B checked (Qwen3-4B has the same layout but is unchecked) |
| Weights | Official BF16 safetensors, one file or shards with their index, quantized to Q8 or Q4 by dianoia |
| Tokenizer | Byte-level BPE from tokenizer.json (Qwen2 pre-tokenizer, merges always applied) |
| Chat | Qwen3's chat template, `<tool_call>` JSON tool calls, `<tool_response>` results, thinking on or off |
| Decoding | Greedy, or temperature / top-k / top-p / min-p sampling with a seed |
| Backends | CPU (NEON, AVX2), Metal; Vulkan builds but is not yet checked on a device |

Rejected at load, rather than run wrongly: untied embeddings (Qwen3-8B and
up), attention bias, sliding-window attention, `rope_scaling` (YaRN), and
activations other than SiLU. Mixture-of-experts Qwen3 models have a different
`model_type` and are rejected too.

## How it maps onto the engine

A Qwen3 layer is an LFM3 attention layer with less: no per-head output gate,
no sandwich norms and no conv layers. The engine describes both with one
`TextModelConfig` (`attentionOutputGate`, `sandwichNorm`, `layerTypes`), and
its kernels skip the gate when a layer has none. Without post norms, a new
`add` GPU kernel adds each sublayer's output to the residual stream.

Bundles keep LFM3's tensor names. `importHfCheckpoint` (`dianoia import-hf`)
renames a Qwen3 checkpoint once, merging its shards and dropping the redundant
tied `lm_head`:

| Qwen3 | Engine |
|---|---|
| `input_layernorm` | `operator_norm` |
| `post_attention_layernorm` | `ffn_norm` |
| `mlp.gate_proj` / `up_proj` / `down_proj` | `feed_forward.w1` / `w3` / `w2` |
| `model.norm` | `model.embedding_norm` |

Attention projections, `q_norm`, `k_norm` and the embedding keep their names.
Quantized caches are named by architecture: `qwen3-q8.cache` next to LFM3's
`lfm3-q4.cache`.

Since every Qwen3 layer is attention, the KV rows before a position are the
model's whole state there. A prompt that diverges from what the model has
seen is reused up to the divergence itself, not to the last checkpoint. This
matters because Qwen3's template drops the think block of earlier turns, so
each new prompt differs from the previous one inside the last reply.

## Installing

`ModelStore` installs `TextCheckpointSpec`s: it downloads the pinned BF16
files, then renames, quantizes and packs them in an isolate.

| Spec | Revision | Download | Installed | Setup peak |
|---|---|---:|---:|---:|
| `TextCheckpointSpec.qwen3_0_6bQ8` | `c1899de` | 1.51 GB | 0.99 GB | 4.5 GB |
| `TextCheckpointSpec.qwen3_1_7bQ8` | `70d244c` | 4.07 GB | 2.57 GB | 12.5 GB |

The setup peak holds the download, a renamed staging copy, its quantized
cache and the bundle at once.

## Validation

Against Hugging Face transformers in float32 (`tool/qwen3_oracle.py`), on an
M2 Max. Greedy agreement counts generated tokens identical to the
reference's before the first difference.

| Model | Weights | Backend | Max logit difference | Greedy agreement |
|---|---|---|---:|---:|
| 0.6B, 23-token chat | f32 | CPU | 5.8e-5 | 24/24 |
| 0.6B, 23-token chat | Q8 | CPU, Metal | 0.95, 0.57 | 24/24 |
| 0.6B, 23-token chat | Q4 | CPU, Metal | 7.5, 7.4 | 16/24 |
| 1.7B, 34-token chat | f32 | CPU | 7.3e-5 | 32/32 |
| 1.7B, 34-token chat | Q8 | CPU, Metal | 0.86, 0.75 | 32/32 |
| 1.7B, 539-token chat | Q8 | CPU, Metal | 1.6, 1.1 | 19/32 |
| 1.7B, 34-token chat | Q4 | Metal | 10.8 | 22/32 |
| 1.7B, 539-token chat | Q4 | Metal | 13.0 | 1/32 |

Float32 matches the reference, so the graph is right. Q8 follows it closely:
on the long prompt it reproduces the reference word for word until a near
tie. Plain (round-to-nearest) Q4 stays fluent but drifts. On the long prompt
it rewords the answer from the first token and starts repeating a note, so
the specs offer Q8 only. LFM3's Q4 is quantization-aware trained; Qwen3 has
no such build.

The tokenizer matches Hugging Face on all 71 oracle cases but one: Qwen3's
NFC normalizer is not applied (Dart has no Unicode normalization), so
decomposed text such as `e` + U+0301 encodes differently. All 17 chat
template cases match exactly.

Decode speed on Metal, M2 Max, short prompts:

| Model | Q8 | Q4 |
|---|---:|---:|
| Qwen3-0.6B | ~210 tok/s | ~250 tok/s |
| Qwen3-1.7B | ~115 tok/s | |

Qwen3-1.7B Q8 prefills the 539-token prompt in 1.5 s on Metal.

## Using it

```dart
final store = ModelStore(dir);
const spec = TextCheckpointSpec.qwen3_0_6bQ8;
await store.install(spec).drain<void>();
final model = await LocalModel.load(store.bundlePath(spec));
final chat = model.chat(
  enableThinking: true,
  sampling: SamplingParams.qwen3Thinking,
);
final turn = await chat.send({'role': 'user', 'content': 'What is 17 * 23?'});
print(turn.reply.reasoning);
print(turn.reply.content);
```

Qwen recommends sampling over greedy decoding, which tends to loop in long
reasoning: `SamplingParams.qwen3Thinking` (temperature 0.6, top-p 0.95,
top-k 20) with thinking and `qwen3NonThinking` (0.7, 0.8, 20) without.
Tool-call argument values that are not strings arrive as their JSON text,
since `ToolCall.arguments` holds strings for both families.

## Known gaps

- NFC normalization of input text (above).
- Vulkan: compiled and syntax-checked, not yet run on Android or Windows.
- KV cache in float32: every Qwen3 layer is attention, so a token costs
  about 230 KB (0.6B and 1.7B: 28 layers × 2 × 1024 floats × 4 bytes),
  against about 45 KB for LFM3-1.7B. The default 2048-token context needs
  about 470 MB; a 16-bit cache would halve it.
- Setup storage: the staging copy could be avoided by renaming tensors as
  they are read.
- Q4: a quality-preserving 4-bit format (or per-channel scales) would make
  Q4 worth offering for Qwen3.
