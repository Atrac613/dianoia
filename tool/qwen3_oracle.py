"""Writes the Qwen3 tokenizer, chat-template and forward oracles.

The text oracle (test/fixtures/qwen3_text_oracle.json, committed) needs only
the tokenizer and chat template, so `dart test` can check dianoia's BPE and
`renderChat` for Qwen3 without the weights. It records:

- `encode`: `{text, ids}` from `tokenizer(text, add_special_tokens=False)`
  (Qwen3 adds no BOS). Covers NFC normalization, the digit-by-digit split,
  contractions, Japanese, emoji, and added tokens spelled in plain text.
- `decode`: `{ids, skip_special_tokens, text}`, including ids that end in
  the middle of a UTF-8 sequence (decoded as U+FFFD).
- `templates`: `{name, messages, tools, enable_thinking,
  add_generation_prompt, prompt}` from `apply_chat_template(tokenize=False)`.
  `tools` is null when the case passes none; `enable_thinking` is null when
  the case leaves it undefined (which the template treats as true).
- `eos_token_ids`: from config.json and generation_config.json.

With `--forward`, it also runs the model in float32 on CPU and writes one
directory per prompt under `--forward-dir` (default build/qwen3_oracle,
never committed), e.g. build/qwen3_oracle/ref_short_chat/:

    ids.i32        int32 LE [n]            prompt token ids
    hidden.bin     float32 LE [n, hidden]  output of the final RMSNorm
                                           (model.norm), every position
    logits.bin     float32 LE [vocab]      lm_head logits, last position
    greedy.i32     int32 LE [32]           greedy continuation; EOS does not
                                           stop it, so all 32 are argmax
    meta.json      model, revision, prompt text, n, hidden, vocab

Weights are the official BF16 safetensors at the pinned revisions below,
downloaded into --models (default ../dart-llm-lab/models/Qwen3-<size>) when
missing. Everything needed (transformers, torch, safetensors, numpy,
huggingface_hub, jinja2) is already in dart-llm-lab's venv, see
tool/requirements-vision-validation.txt.

Usage, with dart-llm-lab checked out next to this repository:
    ../dart-llm-lab/venv/bin/python -I tool/qwen3_oracle.py \
        --text-out test/fixtures/qwen3_text_oracle.json
    ../dart-llm-lab/venv/bin/python -I tool/qwen3_oracle.py --forward
    ../dart-llm-lab/venv/bin/python -I tool/qwen3_oracle.py --size 1.7B \
        --forward --forward-dir build/qwen3_oracle_1.7b
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hugging Face commit of each official checkpoint the oracles come from.
REVISIONS = {
    "0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}

GREEDY_TOKENS = 32

CAFE_NFC = "café"
CAFE_NFD = unicodedata.normalize("NFD", CAFE_NFC)
assert CAFE_NFD == "café"

ENCODE = [
    "",
    " ",
    "  ",
    "\n",
    "\n\n\n",
    " \n \n",
    "\t",
    "Hello, world!",
    " Hello",
    "Hello  world   end",
    "line one\nline two\r\nline three\n\n",
    "trailing spaces   ",
    "1234567890",
    "Pi is 3.14159265358979 and 2^64 = 18446744073709551616.",
    "x123y 1,000,000 0.5",
    "I'm sure you'll see it's what we've done, they'd say. DON'T, I'LL",
    "Don't",
    "rock 'n' roll",
    "def f(x):\n    return x * 2  # double\n",
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Kobe\"}}",
    "日本語のテキストです。東京は晴れ。",
    "カタカナとひらがな、そして漢字",
    "한국어 텍스트",
    "Привет, мир",
    "emoji 😀👍🏽 family 👨‍👩‍👧 flag 🇯🇵",
    CAFE_NFD,
    CAFE_NFC,
    "Ａｂｃ　全角",
    "<|im_start|>user\nhi<|im_end|>\n",
    "<|endoftext|>",
    "<think>\nreasoning\n</think>\n\nanswer",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
    "<tool_response>\nok\n</tool_response>",
    "a<think>b</think>c",
    " <tool_call> spaced </tool_call> ",
    "<|im_start|",
    "<|fim_prefix|>x<|fim_suffix|>y<|fim_middle|>",
]


def tool(name: str, description: str, properties: dict) -> dict:
    """An OpenAI-style function tool."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(properties)},
        },
    }


WEATHER = tool(
    "get_weather",
    'Get the current weather for a city. Units are "metric" or "imperial".',
    {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["metric", "imperial"]}},
)
# Non-ASCII and nested values pin down how `tojson` escapes and spaces.
TIDES = tool(
    "get_tides",
    "潮汐表を取得する (tide table) — returns high/low times.",
    {"port": {"type": "string", "description": "港の名前"}, "days": {"type": "integer", "minimum": 1}},
)
TOOLS = [WEATHER, TIDES]

SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
USER = {"role": "user", "content": "What is the weather in Kobe?"}


def call(name: str, arguments: dict | str) -> dict:
    """A tool call as it appears in an assistant message."""
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


TEMPLATES = [
    {"name": "user_only", "messages": [USER]},
    {"name": "system_and_user", "messages": [SYSTEM, USER]},
    {"name": "thinking_on", "messages": [SYSTEM, USER], "enable_thinking": True},
    {"name": "thinking_undefined", "messages": [SYSTEM, USER], "enable_thinking": None},
    {"name": "no_generation_prompt", "messages": [SYSTEM, USER], "add_generation_prompt": False},
    {"name": "tools_without_system", "messages": [USER], "tools": TOOLS},
    {"name": "tools_with_system", "messages": [SYSTEM, USER], "tools": TOOLS},
    {"name": "tools_thinking_on", "messages": [SYSTEM, USER], "tools": TOOLS, "enable_thinking": True},
    {
        "name": "multi_turn_reasoning",
        # Reasoning before the last user query is dropped; after it, kept.
        "messages": [
            SYSTEM,
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "Greet back."},
            {"role": "user", "content": "And 2+2?"},
            {"role": "assistant", "content": "4.", "reasoning_content": "\nSimple sum.\n"},
        ],
        "add_generation_prompt": False,
        "enable_thinking": True,
    },
    {
        "name": "multi_turn_reasoning_then_user",
        "messages": [
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "Greet back."},
            {"role": "user", "content": "And 2+2?"},
        ],
        "enable_thinking": True,
    },
    {
        "name": "reasoning_inline_in_content",
        # No reasoning_content: the template splits <think>...</think> out.
        "messages": [
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "<think>\nOld thought.\n</think>\n\nHello!"},
            {"role": "user", "content": "Bye."},
            {"role": "assistant", "content": "<think>\nNew thought.\n</think>\n\nBye!"},
        ],
        "add_generation_prompt": False,
    },
    {
        "name": "assistant_empty_reasoning_after_query",
        "messages": [
            USER,
            {"role": "assistant", "content": "", "tool_calls": [call("get_weather", {"city": "Kobe"})]},
            {"role": "tool", "content": "Rain, 14 C"},
            {"role": "assistant", "content": "It is raining.", "reasoning_content": ""},
        ],
        "tools": TOOLS,
        "add_generation_prompt": False,
    },
    {
        "name": "tool_round_trip",
        "messages": [
            SYSTEM,
            USER,
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Need the weather tool.",
                "tool_calls": [call("get_weather", {"city": "Kobe", "unit": "metric"})],
            },
            {"role": "tool", "content": '{"temperature": 14, "condition": "rain"}'},
        ],
        "tools": TOOLS,
        "enable_thinking": True,
    },
    {
        "name": "parallel_tool_calls",
        "messages": [
            USER,
            {
                "role": "assistant",
                "content": "Checking both.",
                "tool_calls": [
                    call("get_weather", {"city": "Kobe"}),
                    call("get_tides", '{"port": "神戸", "days": 2}'),
                ],
            },
            {"role": "tool", "content": "Rain, 14 C"},
            {"role": "tool", "content": "High tide 06:12"},
            {"role": "assistant", "content": "Rain, and high tide at 06:12."},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": TOOLS,
    },
    {
        "name": "user_tool_response_text",
        # A user message wrapped in <tool_response> is not the last query.
        "messages": [
            {"role": "user", "content": "Start."},
            {"role": "assistant", "content": "A.", "reasoning_content": "kept"},
            {"role": "user", "content": "<tool_response>\nraw\n</tool_response>"},
        ],
    },
    {
        "name": "second_system_message",
        "messages": [SYSTEM, USER, {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Go."}],
    },
    {
        "name": "non_string_content",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "ignored"}]}],
    },
]

DECODE_EXTRA = [
    # Split mid code point by BPE: every prefix of the ids is a case, so the
    # ones ending inside a UTF-8 sequence decode with U+FFFD.
    " 👨‍👩‍👧 🇯🇵",
]

SHORT_CHAT = [SYSTEM, {"role": "user", "content": "Name three prime numbers and explain why they are prime."}]

LONG_TOPICS = [
    "The Seto Inland Sea separates Honshu, Shikoku and Kyushu and is dotted with hundreds of small islands.",
    "A binary search halves the remaining interval on every comparison, so it runs in logarithmic time.",
    "Photosynthesis converts light, water and carbon dioxide into glucose and oxygen inside chloroplasts.",
    "In 1543 Copernicus published a model that placed the Sun, not the Earth, at the center.",
    "Sourdough bread rises because wild yeast and lactic acid bacteria ferment the flour.",
    "TCP retransmits segments that are not acknowledged before a timeout derived from the round-trip time.",
    "The violin has four strings tuned in fifths: G, D, A and E.",
    "Matrix multiplication is associative but, in general, not commutative.",
    "Glaciers carve U-shaped valleys, while rivers usually cut V-shaped ones.",
    "A haiku traditionally has three phrases of five, seven and five morae.",
    "Rust's borrow checker rejects programs where a mutable reference aliases another reference.",
    "The speed of sound in dry air at 20 degrees Celsius is about 343 meters per second.",
    "Kintsugi repairs broken pottery with lacquer mixed with powdered gold.",
    "Entropy in an isolated system never decreases over time.",
    "Octopuses have three hearts and blue, copper-based blood.",
    "A Bloom filter can report false positives but never false negatives.",
]


def long_chat() -> list[dict]:
    """A user message of about 500 tokens: varied facts to summarize."""
    facts = "\n".join(f"{i + 1}. {fact}" for i, fact in enumerate(LONG_TOPICS + LONG_TOPICS[:8]))
    text = f"Here are some notes.\n\n{facts}\n\nGroup these notes by subject and summarize each group in one sentence."
    return [SYSTEM, {"role": "user", "content": text}]


def model_dir(args: argparse.Namespace) -> Path:
    """The local checkpoint, downloaded at the pinned revision if missing."""
    path = args.models or Path(f"../dart-llm-lab/models/Qwen3-{args.size}")
    if not (path / "config.json").exists():
        if not path.parent.is_dir():
            # E.g. run from a git worktree, where ../dart-llm-lab is elsewhere.
            sys.exit(f"{path.parent} does not exist; pass --models")
        snapshot_download(f"Qwen/Qwen3-{args.size}", revision=REVISIONS[args.size], local_dir=path)
    return path


def text_oracle(tokenizer, path: Path, size: str) -> dict:
    """Tokenizer and chat-template cases; no model forward."""
    encode = [{"text": text, "ids": tokenizer(text, add_special_tokens=False).input_ids} for text in ENCODE]

    decode = []
    for case in encode:
        if case["ids"] and any(t in case["text"] for t in ("<|", "<think>", "<tool", "😀", "日本語", "Hello,")):
            for skip in (False, True):
                decode.append(
                    {
                        "ids": case["ids"],
                        "skip_special_tokens": skip,
                        "text": tokenizer.decode(case["ids"], skip_special_tokens=skip),
                    }
                )
    for text in DECODE_EXTRA:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        for end in range(1, len(ids)):
            decode.append(
                {"ids": ids[:end], "skip_special_tokens": False, "text": tokenizer.decode(ids[:end])}
            )

    templates = []
    for case in TEMPLATES:
        tools = case.get("tools")
        thinking = case.get("enable_thinking", False)
        add_generation_prompt = case.get("add_generation_prompt", True)
        kwargs = {} if thinking is None else {"enable_thinking": thinking}
        prompt = tokenizer.apply_chat_template(
            case["messages"],
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **kwargs,
        )
        templates.append(
            {
                "name": case["name"],
                "messages": case["messages"],
                "tools": tools,
                "enable_thinking": thinking,
                "add_generation_prompt": add_generation_prompt,
                "prompt": prompt,
            }
        )

    config = json.loads((path / "config.json").read_text())
    generation = json.loads((path / "generation_config.json").read_text())
    return {
        "model": f"Qwen/Qwen3-{size}",
        "revision": REVISIONS[size],
        "eos_token_ids": {
            "config": config["eos_token_id"],
            "generation_config": generation["eos_token_id"],
        },
        "encode": encode,
        "decode": decode,
        "templates": templates,
    }


def write_forward(model, tokenizer, name: str, messages: list[dict], out: Path, size: str) -> None:
    """Runs one prompt in float32 and writes ids, hidden, logits and greedy."""
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    captured = {}
    hook = model.model.norm.register_forward_hook(lambda _m, _i, output: captured.__setitem__("hidden", output))
    with torch.no_grad():
        result = model(torch.tensor([ids]), use_cache=True)
    hook.remove()
    hidden = captured["hidden"][0].float().numpy()
    logits = result.logits[0, -1].float().numpy()

    greedy = []
    past = result.past_key_values
    next_id = int(np.argmax(logits))
    with torch.no_grad():
        for _ in range(GREEDY_TOKENS):
            greedy.append(next_id)
            step = model(torch.tensor([[next_id]]), past_key_values=past, use_cache=True)
            past = step.past_key_values
            next_id = int(torch.argmax(step.logits[0, -1]))

    out.mkdir(parents=True, exist_ok=True)
    np.asarray(ids, dtype="<i4").tofile(out / "ids.i32")
    hidden.astype("<f4").tofile(out / "hidden.bin")
    logits.astype("<f4").tofile(out / "logits.bin")
    np.asarray(greedy, dtype="<i4").tofile(out / "greedy.i32")
    meta = {
        "model": f"Qwen/Qwen3-{size}",
        "revision": REVISIONS[size],
        "dtype": "float32",
        "prompt": prompt,
        "n": len(ids),
        "hidden": hidden.shape[1],
        "vocab": logits.shape[0],
        "greedy_text": tokenizer.decode(greedy),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {out}: {len(ids)} tokens, greedy {meta['greedy_text']!r}")


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", choices=sorted(REVISIONS), default="0.6B")
    parser.add_argument("--models", type=Path, help="checkpoint directory (default ../dart-llm-lab/models/Qwen3-<size>)")
    parser.add_argument("--text-out", type=Path, help="text oracle JSON to write")
    parser.add_argument("--forward", action="store_true", help="also write float32 forward references")
    parser.add_argument("--forward-dir", type=Path, default=Path("build/qwen3_oracle"))
    args = parser.parse_args()
    if not args.text_out and not args.forward:
        parser.error("nothing to do: pass --text-out and/or --forward")

    path = model_dir(args)
    tokenizer = AutoTokenizer.from_pretrained(path)

    if args.text_out:
        oracle = text_oracle(tokenizer, path, args.size)
        args.text_out.write_text(json.dumps(oracle, ensure_ascii=False, indent=1) + "\n")
        counts = {key: len(oracle[key]) for key in ("encode", "decode", "templates")}
        print(f"wrote {args.text_out}: {counts}")

    if args.forward:
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
        for name, messages in (("short_chat", SHORT_CHAT), ("long_chat", long_chat())):
            write_forward(model, tokenizer, name, messages, args.forward_dir / f"ref_{name}", args.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
