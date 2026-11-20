# Qwen3 support

dianoia runs Qwen3 dense checkpoints (`model_type: qwen3`) for text, tool
calling and thinking, on the CPU kernels and on Metal or Vulkan. It is the
engine's second architecture after LFM3. Apps install Qwen's official Q8_0
GGUFs; the BF16 checkpoints import too, for validation in float32.

## What is supported

| | |
|---|---|
| Checkpoints | Qwen3 dense with tied embeddings: Qwen3-0.6B and Qwen3-1.7B checked (Qwen3-4B has the same layout but is unchecked) |
| Weights | Qwen's Q8_0 GGUF, imported as it is; or the official BF16 safetensors (one file or shards), quantized to Q8 or Q4 by dianoia |
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

Bundles keep LFM3's tensor names. `importGguf` (`dianoia import-gguf`) maps
llama.cpp's names onto them, and `importHfCheckpoint` (`dianoia import-hf`)
renames a BF16 checkpoint once, merging its shards and dropping the redundant
tied `lm_head`:

| Qwen3 | GGUF | Engine |
|---|---|---|
| `input_layernorm` | `attn_norm` | `operator_norm` |
| `post_attention_layernorm` | `ffn_norm` | `ffn_norm` |
| `mlp.gate_proj` / `up_proj` / `down_proj` | `ffn_gate` / `ffn_up` / `ffn_down` | `feed_forward.w1` / `w3` / `w2` |
| `self_attn.q_proj` ... `o_proj` | `attn_q` ... `attn_output` | `self_attn.q_proj` ... `o_proj` |
| `self_attn.q_norm` / `k_norm` | `attn_q_norm` / `attn_k_norm` | `self_attn.q_norm` / `k_norm` |
| `model.norm` | `output_norm` | `model.embedding_norm` |
| `model.embed_tokens` | `token_embd` | `model.embed_tokens` |

A Q8_0 block (a half scale and 32 int8 values) is the engine's Q8 block with
its scale widened to float, which is exact: the import copies the values, and
nothing is quantized on the device. The embedding stays in the Q8 cache,
where it is both the LM head and the token lookup table.
Quantized caches are named by architecture: `qwen3-q8.cache` next to LFM3's
`lfm3-q4.cache`.

Since every Qwen3 layer is attention, the KV rows before a position are the
model's whole state there. A prompt that diverges from what the model has
seen is reused up to the divergence itself, not to the last checkpoint. This
matters because Qwen3's template drops the think block of earlier turns, so
each new prompt differs from the previous one inside the last reply.

## Installing

`ModelStore` installs these `ModelSpec`s: the pinned GGUF plus config.json,
generation_config.json, tokenizer.json and LICENSE from the BF16 repository.
The GGUF is imported in an isolate and then deleted.

| Spec | GGUF revision | Download | Installed | Setup peak |
|---|---|---:|---:|---:|
| `ModelSpec.qwen3_0_6bQ8` | `23749fe` | 0.65 GB | 0.68 GB | 1.4 GB |
| `ModelSpec.qwen3_1_7bQ8` | `90862c4` | 1.85 GB | 1.95 GB | 3.8 GB |

Before these, installing quantized the BF16 checkpoint on the device, which
downloaded 1.51 and 4.07 GB and needed 4.5 and 12.5 GB free while building.

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
| 0.6B, 34-token chat | GGUF Q8_0 | CPU, Metal | 0.70, 0.59 | 20/32 |
| 0.6B, 539-token chat | GGUF Q8_0 | CPU, Metal | 0.86, 0.71 | 32/32 |
| 1.7B, 34-token chat | GGUF Q8_0 | CPU, Metal | 1.65, 1.57 | 32/32 |
| 1.7B, 539-token chat | GGUF Q8_0 | CPU, Metal | 1.19, 0.95 | 12/32, 14/32 |

Float32 matches the reference, so the graph is right. Q8 follows it closely:
on the long prompt it reproduces the reference word for word until a near
tie. Plain (round-to-nearest) Q4 stays fluent but drifts. On the long prompt
it rewords the answer from the first token and starts repeating a note, so
the specs offer Q8 only. LFM3's Q4 is quantization-aware trained; Qwen3 has
no such build.

Qwen's Q8_0 GGUF is not dianoia's Q8 of the published BF16 weights. The
embedding matches it value for value, but a third of the layer matrices'
values are one step off and block scales differ by up to 10% in both
directions, so the GGUF was quantized from slightly different weights. On
0.6B it does as well as dianoia's own Q8; on 1.7B it stays further from the
BF16 reference (logit difference 1.6 against 0.8 on the short chat) and
leaves the long chat's reference text earlier, at 12 to 14 tokens against
19. Its answers stay coherent. ModelStore uses it anyway, for the smaller
download and setup; `import-hf` and `pack` still build dianoia's own Q8
from BF16.

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
const spec = ModelSpec.qwen3_0_6bQ8;
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
- Q4: a quality-preserving 4-bit format (or per-channel scales) would make
  Q4 worth offering for Qwen3.
