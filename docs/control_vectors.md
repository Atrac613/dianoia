# Control vectors (experimental)

A control vector steers the model by changing its residual stream: per
layer, a direction is applied to the hidden state leaving that layer, on
every token, as llama.cpp's `--control-vector` does. It is research tooling
under `package:dianoia/lab.dart`, not part of the app API.

- **CPU and GPU.** The CPU forward pass and the GPU one (Metal or Vulkan,
  the `steer` kernel) both apply vectors. What has been run: the CPU path
  and Metal end to end on an Apple silicon Mac (LFM3-1.7B Q4); the Vulkan
  kernel only on its own, in a host check on MoltenVK, lavapipe and
  SwiftShader on that Mac. The Vulkan forward pass with a vector has been
  compiled into this layer but not run on Android or Windows.
- **It changes the model's output.** Keep it off for quality benchmarks,
  `tool/lab_parity.sh` and any comparison with reference dumps. With no
  vector set the arithmetic is exactly what it was without this feature.

## In an app

Experimental and lab-only: `package:dianoia/dianoia.dart` does not expose
it. Importing `package:dianoia/lab.dart` adds `setControlVector` and
`controlVector` to `LocalModel` and `LocalVisionModel` (the extensions
`LocalModelControlVector` and `LocalVisionModelControlVector`).

```dart
import 'package:dianoia/lab.dart';

final model = await LocalModel.load(bundleDir);
final chat = model.chat(system: 'Answer briefly.');
await chat.send({'role': 'user', 'content': 'Hi!'});

await model.setControlVector(
  ControlVector.load(vectorPath, mode: ControlMode.project, firstLayer: 8),
);
final steered = await chat.send({'role': 'user', 'content': 'And now?'});
await model.setControlVector(null); // back to the plain model
```

Call it between replies: during a reply or a warm-up it throws
`StateError` and changes nothing. A different vector resets the model's
computed state; conversations keep their messages, and their next `send`
prefills them again under the new vector, replying exactly as a model
steered from the start would. A saved warm-up state is not loaded under
another vector (it is recomputed and saved again). Visual turns prefill
from scratch anyway, so a vector on a `LocalVisionModel` applies from the
next turn.

## Modes

For the residual row `x` leaving layer `l`, with the direction `d` of that
layer:

| Mode | Update | Notes |
|---|---|---|
| `ControlMode.add` | `x += scale * d` | `scale * d` is rounded to float32 once |
| `ControlMode.project` | `x -= scale * (x · u) u`, `u = d / \|d\|` | `scale = 1` removes the component along `d`; a zero direction is an error |

Only layers from `firstLayer` to `lastLayer` (inclusive, zero-based) that
have a direction are changed.

## CPU and GPU

The GPU uploads, when a vector is set, what `apply` uses per layer
(`ControlVector.appliedVector`: `scale * d` for add, the unit direction for
project) into one buffer of `numLayers × hiddenSize` floats, released when
the vector is replaced or cleared and when the model closes. After each
steered layer's last residual add (the post-FFN norm), one `steer` dispatch
updates the chunk's rows in place, one threadgroup per row, before the
capture copy, the next layer and the LM head read them. Without a vector
nothing is dispatched: the command buffer is exactly what it was.

- **Add** rounds the same as the CPU: the GPU's rows are bit-identical to
  `apply` on the same input.
- **Project** sums `x · u` per row in float32 across the threadgroup
  (strided partial sums, then a SIMD/subgroup reduction), where the CPU
  sums in double in index order, so rows differ in the last bits (about
  1e-7 relative).

The GPU and CPU forward passes already differ without a vector (the CPU
quantizes matrix inputs to int8), so greedy replies may part ways at a
close call with or without one; a vector does not widen the gap.

A GPU library passed as `gpuLib` that lacks `lab_gpu_enc_steer` (an older
build, such as the lab's) refuses vectors with `UnsupportedError`.

## Lab API

```dart
import 'package:dianoia/lab.dart';

final vector = ControlVector.load(
  'happy.safetensors',
  mode: ControlMode.add,
  scale: 0.8,
  firstLayer: 10,
  lastLayer: 14,
);
await model.reset();            // the sequence must be empty
model.setControlVector(vector); // null stops steering
```

`ControlVector(directions, hiddenSize: ...)` builds one from a map of
zero-based layer to `Float32List`; `withSettings` changes mode, scale or
range; `apply(rows, rowCount, layer)` is the arithmetic, in place.

`Lfm3Model.forward`'s capture sees the steered residual: slot `l + 1` is
the output of layer `l` after its direction is applied, and the next layer
and the logits read the same values.

### State

KV rows, conv histories and checkpoints computed under one vector are wrong
under another, so `setControlVector` throws `StateError` unless the sequence
is empty (`length == 0`, after `reset`) or the new vector steers exactly as
the current one. `reset` keeps the vector. `saveState` records the vector's
fingerprint (a SHA-256 of mode, scale, steered layers and their directions)
in the state key, and `loadState` returns false for a state saved under a
different vector or none; states saved without a vector keep their key.

## Files

A file holds directions only; mode, scale and range are chosen when it is
loaded. Every tensor is an F32 vector named `direction.<l>`:

- `ControlVector.save` writes safetensors (`writeSafetensorsF32`).
- `ControlVector.load` reads that, and llama.cpp's GGUF control vectors
  (detected by the `GGUF` magic; reading GGUF needs the native kernels).

`direction.<l>` steers the output of zero-based layer `l`. That is what
llama.cpp's runtime does with a file: it adds `direction.N` to `l_out-N`.
llama.cpp never stores layer 0; this format may. Note that llama.cpp's own
`cvector-generator` names the difference taken at `l_out-N` as
`direction.N+1`, one layer later than where its runtime applies it; files
from it therefore act one layer deeper than they were measured, here as in
llama.cpp.

## Tools

Both take a model bundle with `lfm3-q4.cache` or `lfm3-q8.cache` and run on
the CPU kernels, or with `--gpu` on the GPU. Prompt files are plain text,
one user turn per non-empty line; bring your own contrasting sets.

```bash
dart run tool/control_vector_extract.dart <bundle> a.txt b.txt \
  --out=vector.safetensors [--min-separation=2] [--system=prompt] \
  [--threads=4] [--gpu]
```

Renders each prompt through the chat template as a user turn with the
assistant generation prompt (thinking off), runs it, and keeps the residual
leaving each layer at the last token (in memory: one row of hidden size per
prompt and layer). The direction of layer `l` is `d = mean(A) - mean(B)`.
Per layer it prints:

- `|direction|`, the mean residual norm and their ratio;
- `held-out A (sd)`, `held-out B (sd)`: the mean and sample standard
  deviation of each set's leave-one-out projections. Each prompt is
  projected on the unit direction built without it: `mean(A without r) -
  mean(B)` for a prompt r of A, `mean(A) - mean(B without r)` for one of B;
- `d'`, how far apart the two sets lie along those directions in units of
  their spread: `(mean A - mean B) / sqrt((sd A² + sd B²) / 2)` over the
  held-out projections (`LayerSeparation` in the lab API);
- the cosine with the previous layer's direction.

Layers without a direction, and layers whose `d'` is below
`--min-separation` (default 2), are left out of the file and listed with
their `d'`; `--min-separation=0` keeps every layer with a direction, and
the tool fails, writing nothing, if no layer is left. `d'` needs at least
two prompts per set. Last it suggests a `--layers=FIRST-` for the probe:
the layer from which `d'` stays at or above 3 to the last layer. That is a
starting point, not a guarantee; when no trailing range stays at or above
3 it says so instead, and the range is yours to pick from the `d'` column.

The file keeps the direction from all prompts; only the statistic is held
out. Measured on the prompts that define it, a direction always looks as if
it separates them: with few prompts and a hidden size in the thousands it
fits their noise. Held out, a direction that does not generalise scores
near zero or below zero, so a low or negative `d'` means that layer's
direction would not carry over to a new prompt. With eight loosely matched
prompts per set every layer of LFM3-1.7B scored between -0.8 and 0.5 (and
the tool wrote nothing), where the same prompts had looked like 0.5 to 4
in sample.

`d'` says how well a layer separates the two sets given, not whether the
direction means what you intended. Prompts that differ only in fixed
wording (`... in a joyful tone` against `... in a miserable tone`, the same
topics) separate perfectly from the shallowest layers on, through those
words; within a set the last-token residuals are then nearly identical, and
`d'` is very large or infinite. Check a vector's effect on ordinary prompts
with the probe.

**The pitfall this guards against.** In project mode a direction is
normalised, so a layer where the two sets barely differ still gets a
full-strength projection, along what is mostly noise. Both sets may then
have a large component along it: on LFM3-1.7B, layers 1 and 2 had
direction norms of 0.006 and 0.041, yet every prompt projected about a
third of its residual onto them, and removing that changed ordinary replies
far more than the layers that do separate the sets. Restrict the layer
range to where the sets separate.

```bash
dart run tool/control_vector_probe.dart <bundle> vector.safetensors prompts.txt \
  [--mode=add|project] [--scale=1] [--layers=10-14] [--max-new-tokens=48] [--gpu]
```

Generates a greedy reply to each prompt from a reset model, without and
with the vector, and prints them one after the other with their decoding
rates. `--layers` takes `FIRST-LAST`, `FIRST-` or a single layer.

### On a device

```bash
dart run tool/control_vector_bench.dart <bundle> vector.safetensors prompts.txt \
  [--refusals=prompts.txt] [--mode=project] [--scale=1] [--layers=8-] \
  [--out=record.json] [--compare=device.json]
MODE=project SCALE=1 FIRST=8 tool/device_bench/run_ios_control_vector.sh \
  vector.safetensors prompts.txt [refusals.txt] [DEVICE_ID]
```

`benchmarkControlVector` (lab API) runs the same checks on any device, on
the CPU and the GPU: greedy reply token ids without and with the vector,
how many refusal prompts get a reply opening with a refusal (counts only),
decoding and prefill rates alternating without and with the vector, a
self-check (in project mode the largest |x · u| / |x| left in a steered
layer, and whether the layer before the range is unchanged), and whether
setting and clearing the vector restores the logits bit for bit. The tool
runs it on this machine and, with `--compare`, matches the token ids of a
record from the device bench app; the script does the whole iPhone round
trip with a Q4 bundle packed from `models/LFM3-1.7B`'s own Q4 cache.

