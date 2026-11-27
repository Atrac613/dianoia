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
| Backends | CPU (NEON, AVX2), Metal, Vulkan (checked on a Pixel 8; Windows unchecked) |

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

Better rounding does not rescue Q4_0 for these sizes.
`tool/qwen3_q4_study.py` fake-quantizes the layer matrices in transformers
(the embedding stays Q8, as in dianoia). It measures KL divergence against
float32 over every position of the oracle's long chat, teacher-forced. Its
Q8 row reproduces the engine's Q8 greedy agreement above.

| Scheme (bits a weight) | 0.6B KL | 1.7B KL |
|---|---:|---:|
| Q8 (9) | 0.0011 | 0.0016 |
| 6-bit, per 32 (6.5) | 0.014 | 0.012 |
| Q5_0 (5.5) | 0.057 | 0.047 |
| Q4_0, round-to-nearest (4.5) | 0.20 | 0.24 |
| Q4_0, least-error scale | 0.18 | 0.22 |
| Q4_0, activation-weighted error (imatrix) | 0.17 | 0.21 |
| Q4_0, GPTQ | 0.15 | 0.15 |
| Q4_0 imatrix, feed-forward kept Q8 | 0.081 | 0.065 |
| Q4_1, round-to-nearest (5) | 0.20 | |

Every Q4_0 variant stays about 100 times further from float32 than Q8.
Quantizing one layer, or one kind of matrix, at a time shows no culprit to
keep at Q8: every layer adds 0.004 to 0.02 and every kind 0.014 to 0.054
(0.6B). Fewer bytes per token for 0.6B and 1.7B would take a 6-bit format,
with about 0.78 times Q8's bytes (the LM head kept Q8) and new kernels on
each backend.

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

### On phones

`tool/device_bench/run_ios_qwen3.sh` and `run_android_qwen3.sh` (with
`SPEC=1.7b` for 1.7B) have the phone install the spec itself from Hugging
Face, then answer a greedy question, a second turn, a tool call and seeded
thinking, and check each reply. iPhone 15 Pro Max (A17 Pro) and Pixel 8
(Tensor G3):

| | iPhone 0.6B | Pixel 8 0.6B | iPhone 1.7B | Pixel 8 1.7B |
|---|---:|---:|---:|---:|
| Download, import | 57 s, 3.0 s | 83 s, 3.8 s | 128 s, 6.0 s | 16 s import |
| Load: GPU, CPU | 0.86 s, 0.31 s | 2.4 s, 0.39 s | 0.85 s, 0.20 s | 4.0 s, 1.1 s |
| Decode, GPU (Metal, Vulkan) | 39-46 tok/s | 20-27 tok/s | 15-18 tok/s | 8-10 tok/s |
| Decode, CPU | 34-39 tok/s | 30-32 tok/s | 16-19 tok/s | 7-8 tok/s |
| Footprint, GPU / CPU | 680 / 240 MB | 2.1 GB / 1.0 GB | 0.9 GB / 210 MB | 3.9 / 2.1 GB |

Every check passes on both phones and both backends: Tokyo, then Paris
reusing the first turn's 20 common tokens, and `get_weather(city: Paris)`.
Thinking reaches 391 on both GPUs. The iPhone's runs match the Mac's token
for token (998 tokens for 0.6B, 1,699 for 1.7B), while Vulkan's arithmetic
takes the sampler down other paths (588 and 2,732 tokens). The Pixel 8's
Mali GPU still decodes Qwen3-0.6B a little slower than the CPU. It was half
the CPU's speed (12-16 tok/s) while Dart read the logits in the GPU's
buffer, which is uncached for the CPU on a phone: an argmax over Qwen3's
151,936 logits took 21 ms a token there. The logits are now copied out in
one 1 ms copy first. Of the 47 ms a token left, the GPU spends 39 ms, 36 of
them in the Q8 matrix-vector products, which read the weights at about
19 GB/s. That is close to what this GPU reads at all: its copy kernel moves
41 GB/s counting reads and writes, about 20 GB/s each way. Timing every
matrix kind across its 28 layers (`GPU_SHAPES=1
tool/device_bench/run_android_qwen3.sh`) shows the same 16-20 GB/s for
small matrices (1,024 rows) and large ones, and 22-29 GB/s for the LM head.
The best rows per subgroup and workgroup size for each kind would save
about 6 of the 37 ms (17%), and no layout beats the CPU, which reads at a
similar rate. Decoding Qwen3 faster on this phone means reading fewer bytes
a token. The 1.7B row predates the copy. A Pixel 8 that has been running
for a while throttles: a second 1.7B run decoded at 5 tok/s on Vulkan and
loaded in 19 s.

So `InferenceBackend.auto` runs Qwen3 on the CPU kernels on Android, where
they decode faster and need less memory. The GPU still prefills about
twice as fast (169 tokens in 1.1 s against 2.2 s on 0.6B), so an app whose
prompts are long can ask for `InferenceBackend.gpu`. LFM3 and other
platforms keep the GPU.

### Speculative decoding

Speculative decoding, with 0.6B drafting tokens for 1.7B to verify in one
pass, was measured and not built. In float32 the draft's next token is the
target's 65 to 87% of the time (76% over six prompts, highest on code and
thinking), sampled with the thinking preset or greedy alike. A pass over
several tokens on the iPhone (`VERIFY_COST=1
tool/device_bench/run_ios_qwen3.sh`), in ms:

| Tokens in the pass | 1 | 2 | 3 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| 1.7B, Metal | 58 | 72 | 73 | 76 | 121 |
| 1.7B, CPU | 61 | 85 | 111 | 137 | 245 |
| 0.6B, Metal | 22 | 29 | 31 | 32 | 46 |

Metal verifies four tokens for 1.3 times one, but each drafted token costs
22 ms, more than a third of the target's step. At 76% the best draft length
(two) decodes 1.13 times as fast on Metal (1.29 at 86%) and no faster on
the CPU, whose cost grows with the tokens. That is not worth a second
model's 0.7 GB. A draft that costs nearly nothing, such as copying a
continuation found earlier in the context, would only need 1.35 tokens
accepted per pass on Metal.

### Long conversations

Every Qwen3 layer is attention, so each token decoded reads the whole KV
cache: in float32 about 230 KB per earlier token (28 layers × 2 × 1,024
floats × 4 bytes), 470 MB at 2,048 tokens, which outweighs 0.6B's 0.65 GB
of weights before long. Two changes cut that cost: attention split by
blocks of keys, then a half-precision cache.

#### Split attention

The attention kernels
gave each query head of a token one workgroup (one task on the CPU) that
walked every key, and each of the two query heads per KV head read it.
Past 512 keys, passes of up to 8 tokens now attend by blocks of keys: one
workgroup or task per KV head and block scores every query head of the
group, reading each key once, and a second pass merges the blocks. The
sums run in another order: attention outputs and Metal's logits stay
within 1e-5 of the whole kernels, while on the CPU's int8-input path a
logit can move by 0.2, as any reordering moves it there (float inputs:
6e-6). Shorter passes keep the old kernels, so the lab's reference
prompts are unchanged.

One token deep into the cache, Q8, in ms, whole and split. An M2 Max
(single-token `TextModel.forward` passes, 4 CPU threads):

| | 64 | 2,048 | 4,000 |
|---|---:|---:|---:|
| 0.6B, Metal | 4.5 | 9.9 → 6.1 | 15.2 → 7.8 |
| 1.7B, Metal | 9.1 | 14.4 → 10.2 | 19.5 → 12.2 |
| 0.6B, CPU | 15.6 | 40.6 → 33.5 | 77.5 → 57.3 |

An iPhone 15 Pro Max (`CONTEXT_COST=1 CONTEXT_COMPARE=split
tool/device_bench/run_ios_qwen3.sh`, whole and split passes alternating,
on a phone already warm, both with a float32 cache):

| | 1,024 | 2,048 | 3,000 |
|---|---:|---:|---:|
| 0.6B, Metal | 29.8 → 30.0 | 37.4 → 36.7 | 43.5 → 43.4 |
| 1.7B, Metal | 63.0 → 63.6 | 68.2 → 67.6 | 74.0 → 75.1 |
| 0.6B, CPU | 62.8 → 45.4 | 109.0 → 67.6 | 186.5 → 108.1 |
| 1.7B, CPU | 132.2 → 113.0 | 201.9 → 145.5 | 332.6 → 202.1 |

The CPU kernels split everywhere: 1.6 to 1.7 times as fast at 3,000
tokens on the phone. The phone's GPU gains nothing, since its few cores
already read the cache near its memory bandwidth, so the GPU splits by
default on macOS only (`GpuForward.attentionSplitFrom`). The Vulkan
kernels have not run on a device yet and stay off. What is left on the
phone's GPU is the cache's bytes.

#### Half-precision cache

`KvCacheFormat.f16` stores keys and values as IEEE halves, rounded to
nearest even as rows are written and widened to float32 as attention reads
them, so a token costs about 115 KB: 235 MB at 2,048 tokens. Qwen3 uses it
by default on the CPU and on Metal (`TextModel.load(kvCache:)` chooses);
LFM3, whose cache is a fifth of Qwen3's, keeps float32 and with it the
lab's arithmetic, and Vulkan keeps float32 until its half kernels have run
on a device. Over 3,500 tokens of real text (this repository's README and
this page), the next-token distribution moves from the float32 cache's by
a mean KL of about 1e-6 at 70 positions (Q8 weights: 1e-3), on Metal and on
the CPU with float inputs, for 0.6B and 1.7B, and every argmax agrees.

One token deep into the cache, float32 → half, in ms. The M2 Max, 0.6B:

| | 1,024 | 2,048 | 4,000 |
|---|---:|---:|---:|
| CPU | 24.9 → 16.9 | 37.1 → 28.1 | 55.3 → 31.0 |
| Metal | 5.3 → 5.3 | 5.7 → 6.1 | 8.0 → 7.7 |

The iPhone (`CONTEXT_COST=1 tool/device_bench/run_ios_qwen3.sh`, the two
caches alternating, attention split as by default):

| | 64 | 1,024 | 2,048 | 3,000 |
|---|---:|---:|---:|---:|
| 0.6B, Metal | 23.6 → 24.0 | 30.4 → 26.9 | 35.0 → 29.1 | 41.4 → 32.5 |
| 1.7B, Metal | 60.3 → 59.9 | 64.3 → 62.3 | 68.3 → 64.2 | 75.7 → 69.2 |
| 0.6B, CPU | 27.8 → 26.7 | 48.2 → 42.0 | 66.7 → 51.0 | 93.6 → 70.5 |
| 1.7B, CPU | 68.7 → 69.7 | 129.2 → 120.6 | 209.2 → 161.9 | 213.1 → 169.9 |

On the phone's GPU, which split attention left as it was, the half cache
halves what 3,000 tokens of context add to a 0.6B token (17.8 ms to 8.5)
and makes 1.7B 9% faster there; the CPU gains another 1.2 to 1.3 times.
The Mac's Metal has bandwidth to spare and does not change.

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
For reasoning that runs long, `thinkingBudget`, `repetitionGuard`,
`SamplingParams.presencePenalty` and `ChatSession.send`'s `checkLine` and
`insert` keep it in bounds and on track; docs/long_reasoning.md measures
how long 0.6B and 1.7B stay right with each.

`LocalModel.load` gives Qwen3 a 4,096-token context unless the app passes
`contextLength`: thinking runs long (998 tokens for 0.6B and 1,699 for
1.7B on one question), and the half KV cache holds 4,096 tokens in the
470 MB that 2,048 took in float32. On Vulkan, whose cache is still
float32, the default stays 2,048. With `enableThinking` and a sampling
temperature above 0, `chat` lets a reply run to the end of the context
unless `maxNewTokens` caps it, since a reply cut inside its reasoning has
no answer; greedy thinking, which tends to loop, keeps the 512-token cap. On the iPhone 15 Pro Max at these defaults,
Qwen3-1.7B on Metal answered after 1,699 tokens of thinking at 13.7 tok/s,
the process footprint at 675 to 685 MB on Metal and 160 to 180 MB on the
CPU.

## Known gaps

- NFC normalization of input text (above).
- Vulkan on Windows is unchecked. On a Pixel 8 it is right but decodes
  Qwen3 no faster than the CPU (above): its Q8 matrix-vector products are
  bound by memory bandwidth in every layout, so `auto` uses the CPU there.
  Decoding faster on that GPU needs fewer bytes per weight.
- Split attention and the half KV cache are unchecked on a Vulkan
  device, so Vulkan keeps whole attention and a float32 cache: 230 KB a
  token, 470 MB at 2,048 tokens (Long conversations).
- Q4: no Q4_0 rounding keeps Qwen3-0.6B or 1.7B close to float32
  (Validation). Qwen ships Q4_K_M only from Qwen3-4B up, which would need
  K-quant kernels.
