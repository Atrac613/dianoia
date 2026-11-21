"""Compares ways to quantize Qwen3 to Q4_0 by fake quantization in transformers.

Every scheme keeps llama.cpp's Q4_0 layout, which dianoia's CPU, Metal and
Vulkan kernels already run: 32 weights per block, one scale each, values
-8..7. Only how the scale and the values are chosen differs, so a winner
ships without kernel work. Each scheme quantizes and dequantizes every layer
matrix in float32, and the model is compared with the unquantized one:

- KL(ref || got) and top-1 agreement over every position of the oracle's
  prompts teacher-forced with the reference's greedy continuation, and
- greedy agreement: tokens generated identical to the reference's
  continuation (tool/qwen3_oracle.py's greedy.i32) before the first
  difference, as docs/qwen3_support.md counts it.

As in dianoia, the tied embedding stays Q8 (it is both the token table and
the LM head), and the norms stay float.

Schemes:
    q8        dianoia's Q8: int8, max-abs scale per 32 (the baseline)
    rtn       llama.cpp's Q4_0 rule (what `pack --weights=q4` does)
    mse       per block, the scale with the least squared error
    imatrix   as mse, the error weighted per column by the mean squared
              activation entering it (llama.cpp's importance matrix),
              collected on calibration text unrelated to the prompts
    imatrix-ffn-q8 / imatrix-v-down-q8
              imatrix, with some matrices kept Q8, as Q4_K_M keeps
              ffn_down and attn_v at 6 bits
    q4_1, q5_0, q6
              other layouts, to see how quality follows the bits: Q4_1
              (asymmetric 4-bit, 5 bits a weight), Q5_0 (5.5) and a
              symmetric 6-bit (6.5), all round-to-nearest
    gptq      GPTQ: columns quantized in order, each one's rounding error
              pushed onto the columns not yet quantized through the
              inverse Hessian of the calibration inputs; a block's scale is
              chosen (as mse) when the block starts

Usage, with dart-llm-lab checked out next to this repository:
    ../dart-llm-lab/venv/bin/python -I tool/qwen3_q4_study.py \\
        --model ../dart-llm-lab/models/Qwen3-0.6B --runs build/qwen3_oracle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BLOCK = 32

# Scale candidates for mse and imatrix: the block's signed extreme divided
# by these. Q4_0's rule is -8; smaller magnitudes clip the extreme.
_DIVISORS = torch.tensor([-8.0 + 0.25 * i for i in range(13)] + [7.0 - 0.25 * i for i in range(9)])


def _blocks(w: torch.Tensor) -> torch.Tensor:
    rows, cols = w.shape
    assert cols % BLOCK == 0, cols
    return w.reshape(rows, cols // BLOCK, BLOCK)


def q8(w: torch.Tensor, _: torch.Tensor | None) -> torch.Tensor:
    """dianoia's Q8: round(x / s) with s = max|x| / 127 per block."""
    x = _blocks(w)
    s = x.abs().amax(-1, keepdim=True) / 127
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (torch.round(x / s).clamp(-127, 127) * s).reshape(w.shape)


def _signed_max(x: torch.Tensor) -> torch.Tensor:
    idx = x.abs().argmax(-1, keepdim=True)
    return torch.gather(x, -1, idx)


def _q4(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    d16 = d.half().float()  # Q4_0 stores the scale as a half
    inv = torch.where(d16 == 0, torch.zeros_like(d16), 1 / d16)
    return torch.round(x * inv).clamp(-8, 7) * d16


def rtn(w: torch.Tensor, _: torch.Tensor | None) -> torch.Tensor:
    """llama.cpp's quantize_row_q4_0_ref: d = signed extreme / -8."""
    x = _blocks(w)
    d = _signed_max(x) / -8
    return _q4(x, d).reshape(w.shape)


def _sym(bits: int):
    """llama.cpp's Q5_0-style rule for [bits]: values -2^(b-1)..2^(b-1)-1,
    d = signed extreme / -2^(b-1), a half scale per 32: bits + 0.5 bits."""
    lo = -(1 << (bits - 1))

    def scheme(w: torch.Tensor, _: torch.Tensor | None) -> torch.Tensor:
        x = _blocks(w)
        d = (_signed_max(x) / lo).half().float()
        inv = torch.where(d == 0, torch.zeros_like(d), 1 / d)
        return (torch.round(x * inv).clamp(lo, -lo - 1) * d).reshape(w.shape)

    return scheme


def asym4(w: torch.Tensor, _: torch.Tensor | None) -> torch.Tensor:
    """llama.cpp's Q4_1: values 0..15 between the block's min and max, a
    half scale and a half min per 32 (5 bits); an upper bound on Q4_K,
    which stores the same per-32 scale and min in 6 bits (4.5 bits)."""
    x = _blocks(w)
    lo = x.amin(-1, keepdim=True).half().float()
    d = ((x.amax(-1, keepdim=True) - lo) / 15).half().float()
    inv = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    return (torch.round((x - lo) * inv).clamp(0, 15) * d + lo).reshape(w.shape)


def _search(w: torch.Tensor, importance: torch.Tensor | None) -> torch.Tensor:
    """The candidate scale with the least (weighted) squared error."""
    x = _blocks(w)
    weight = (
        torch.ones_like(x)
        if importance is None
        else importance.reshape(1, -1, BLOCK).expand_as(x)
    )
    m = _signed_max(x)
    best_err = best = None
    for div in _DIVISORS:
        got = _q4(x, m / div)
        err = (weight * (got - x) ** 2).sum(-1, keepdim=True)
        if best is None:
            best_err, best = err, got
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best = torch.where(better, got, best)
    return best.reshape(w.shape)


def mse(w: torch.Tensor, _: torch.Tensor | None) -> torch.Tensor:
    return _search(w, None)


def imatrix(w: torch.Tensor, importance: torch.Tensor | None) -> torch.Tensor:
    return _search(w, importance)


def gptq(w: torch.Tensor, hessian: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """GPTQ (Frantar et al. 2022) in the Q4_0 layout, without act-order."""
    w = w.clone().double()
    h = hessian.double().clone()
    rows, cols = w.shape
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h += damp * torch.mean(torch.diag(h)) * torch.eye(cols, dtype=h.dtype)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    q = torch.zeros_like(w)
    step = 128
    for i1 in range(0, cols, step):
        i2 = min(i1 + step, cols)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        h1 = hinv[i1:i2, i1:i2]
        scale = None
        for i in range(i2 - i1):
            if i % BLOCK == 0:
                block = w1[:, i : i + BLOCK].float()
                scale = _best_scale(block)
            col = w1[:, i]
            qc = _q4(col.float()[:, None], scale).double()[:, 0]
            q[:, i1 + i] = qc
            e = (col - qc) / h1[i, i]
            w1[:, i:] -= e[:, None] @ h1[i, i:][None, :]
            err1[:, i] = e
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return q.float()


def _best_scale(block: torch.Tensor) -> torch.Tensor:
    """Per row of [rows, 32], the candidate scale with the least error."""
    m = _signed_max(block)
    best_err = best = None
    for div in _DIVISORS:
        d = m / div
        err = ((_q4(block, d) - block) ** 2).sum(-1, keepdim=True)
        if best is None:
            best_err, best = err, d
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best = torch.where(better, d, best)
    return best


def _keep_q8(kinds: tuple[str, ...]):
    def scheme(name: str):
        return q8 if any(k in name for k in kinds) else imatrix

    return scheme


# name: (quantizer per matrix name, needs the importance matrix)
SCHEMES = {
    "q8": (lambda _: q8, False),
    "rtn": (lambda _: rtn, False),
    "mse": (lambda _: mse, False),
    "imatrix": (lambda _: imatrix, True),
    "imatrix-v-down-q8": (_keep_q8(("v_proj", "down_proj")), True),
    "imatrix-ffn-q8": (_keep_q8(("mlp.",)), True),
    "gptq": (lambda _: "gptq", True),
    "q4_1": (lambda _: asym4, False),
    "q5_0": (lambda _: _sym(5), False),
    "q6": (lambda _: _sym(6), False),
}

# Calibration text for the importance matrix: generic prose and code, none
# of it from the oracle's prompts.
_CALIBRATION = [
    "The history of Kyoto spans more than a thousand years. Founded as "
    "Heian-kyo in 794, it served as the imperial capital until 1868, and its "
    "temples, shrines and gardens survived the Second World War largely "
    "intact. Today the city is home to universities, craft workshops and "
    "companies such as Nintendo and Kyocera.",
    "def merge_sort(items):\n    if len(items) <= 1:\n        return items\n"
    "    mid = len(items) // 2\n    left = merge_sort(items[:mid])\n"
    "    right = merge_sort(items[mid:])\n    out = []\n    while left and right:\n"
    "        out.append(left.pop(0) if left[0] <= right[0] else right.pop(0))\n"
    "    return out + left + right\n",
    "東京は日本の首都であり、世界でも有数の大都市圏を形成している。江戸時代には"
    "徳川幕府が置かれ、明治維新の後に天皇が京都から移った。",
    "Q: A train leaves at 9:40 and arrives at 13:15. How long is the trip?\n"
    "A: From 9:40 to 13:40 is 4 hours; 13:15 is 25 minutes earlier, so the "
    "trip takes 3 hours and 35 minutes.",
    "Photosynthesis is not the only way organisms capture energy: "
    "chemosynthetic bacteria near hydrothermal vents oxidize hydrogen "
    "sulfide, and their ecosystems thrive without sunlight.",
    '{"name": "search_flights", "arguments": {"from": "HND", "to": "SFO", '
    '"date": "2026-12-01", "passengers": 2}}',
]


def calibration_sequences(model, tokenizer, samples: int) -> list[torch.Tensor]:
    """The fixed texts as chats, then [samples] replies the model samples
    itself (seeded) to each text's first line: text in the model's own
    distribution, none of it from the oracle's prompts."""
    seqs = []
    torch.manual_seed(0)
    for text in _CALIBRATION:
        for prompt in [text] + [text.split("\n")[0][:80]] * samples:
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=prompt is not text, enable_thinking=False,
            )
            ids = tokenizer(chat, return_tensors="pt").input_ids
            if prompt is not text:
                ids = model.generate(
                    ids, max_new_tokens=192, do_sample=True, temperature=0.8,
                    top_p=0.95, pad_token_id=0,
                )
            seqs.append(ids)
    return seqs


def collect_statistics(model, linears, seqs, hessians: bool):
    """Per layer matrix: the mean squared input per column and, with
    [hessians], the inputs' second moment X^T X / n."""
    sums = {n: torch.zeros(m.in_features, dtype=torch.float64) for n, m in linears.items()}
    grams = {n: torch.zeros(m.in_features, m.in_features) for n, m in linears.items()} if hessians else {}
    counts = dict.fromkeys(linears, 0)
    hooks = []
    for n, m in linears.items():

        def hook(_, inputs, __, n=n):
            x = inputs[0].reshape(-1, inputs[0].shape[-1])
            sums[n] += (x.double() ** 2).sum(0)
            if hessians:
                grams[n] += x.T @ x
            counts[n] += x.shape[0]

        hooks.append(m.register_forward_hook(hook))
    for ids in seqs:
        model(ids)
    for h in hooks:
        h.remove()
    importance = {n: (sums[n] / counts[n]).float() for n in linears}
    return importance, {n: g / counts[n] for n, g in grams.items()}


def kl_top1(ref: torch.Tensor, got: torch.Tensor) -> tuple[float, float, float]:
    """Mean KL(ref || got), its 99th percentile, and top-1 agreement."""
    lp_ref = torch.log_softmax(ref.double(), dim=-1)
    lp_got = torch.log_softmax(got.double(), dim=-1)
    kl = (lp_ref.exp() * (lp_ref - lp_got)).sum(-1)
    top1 = (ref.argmax(-1) == got.argmax(-1)).double().mean()
    return kl.mean().item(), torch.quantile(kl, 0.99).item(), top1.item()


def greedy_agreement(model, ids: torch.Tensor, ref: np.ndarray) -> int:
    """Generated tokens equal to [ref] before the first difference."""
    out = model.generate(
        ids, max_new_tokens=len(ref), min_new_tokens=len(ref), do_sample=False,
        eos_token_id=None, pad_token_id=0,
    )[0, ids.shape[1]:].numpy()
    for i, (a, b) in enumerate(zip(out, ref)):
        if a != b:
            return i
    return len(ref)


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["short_chat", "long_chat"])
    parser.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    parser.add_argument("--samples", type=int, default=4,
                        help="self-generated calibration replies per text")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    embed = model.get_input_embeddings()
    embed_original = embed.weight.detach().clone()
    linears = {
        n: m
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and n.startswith("model.layers.")
    }
    originals = {n: m.weight.detach().clone() for n, m in linears.items()}

    cases = {}
    for case in args.cases:
        d = args.runs / f"ref_{case}"
        ids = np.fromfile(d / "ids.i32", dtype="<i4")
        gen = np.fromfile(d / "greedy.i32", dtype="<i4")
        seq = torch.tensor(np.concatenate([ids, gen[:-1]]), dtype=torch.long)[None]
        cases[case] = (torch.tensor(ids, dtype=torch.long)[None], gen, seq)
    reference = {c: model(seq).logits[0] for c, (_, _, seq) in cases.items()}
    importance, hessians = {}, {}
    if any(SCHEMES[s][1] for s in args.schemes):
        seqs = calibration_sequences(model, tokenizer, args.samples)
        print(f"calibration: {sum(s.shape[1] for s in seqs)} tokens", flush=True)
        importance, hessians = collect_statistics(
            model, linears, seqs, hessians="gptq" in args.schemes
        )

    print(f"{'scheme':18s}  " + "  ".join(
        f"{c + ' KL':>15s} {'p99':>6s} {'top1':>5s} {'greedy':>6s}" for c in cases))
    for name in args.schemes:
        pick, _ = SCHEMES[name]
        # The tied embedding: Q8 in every scheme, as in dianoia's caches.
        embed.weight.data = q8(embed_original, None)
        for n, m in linears.items():
            quantize = pick(n)
            m.weight.data = (
                gptq(originals[n], hessians[n])
                if quantize == "gptq"
                else quantize(originals[n], importance.get(n))
            )
        cells = []
        for case, (ids, gen, seq) in cases.items():
            kl, p99, top1 = kl_top1(reference[case], model(seq).logits[0])
            agree = greedy_agreement(model, ids, gen)
            cells.append(f"{kl:15.4f} {p99:6.3f} {top1 * 100:5.1f} {agree:3d}/{len(gen)}")
        print(f"{name:18s}  " + "  ".join(cells), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
