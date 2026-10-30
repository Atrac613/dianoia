"""Writes the chat-template oracle for deferred tools and tool search.

dart-llm-lab's text oracle (test/fixtures/text_oracle.json) covers ordinary
tools; this one covers the parts of LFM3's chat_template.jinja a tool
search harness needs: tools marked `defer_loading`, `tool_reference` items
in tool results, `result.tools`, the `additional_tools` message field, and
consecutive tool messages merged by `tool_call_id`. It records only the
rendered prompt, so `dart test` can compare `renderChat` against it without
the model.

Usage, with dart-llm-lab checked out next to this repository:
    ../dart-llm-lab/venv/bin/python -I tool/template_oracle.py \
        --model ../dart-llm-lab/models/LFM3-1.7B \
        --out test/fixtures/template_oracle.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def tool(name: str, description: str, properties: dict, **extra) -> dict:
    """An OpenAI-style function tool; extra keys go on the function."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(properties)},
            **extra,
        },
    }


SEARCH = tool(
    "tool_search",
    "Search the deferred tools. Returns references to the matching tools.",
    {"query": {"type": "string"}},
)
WEATHER = tool("get_weather", "Get the current weather for a city.", {"city": {"type": "string"}})
FORECAST = tool(
    "get_forecast",
    'Get a multi-day forecast. Units are "metric" or "imperial".',
    {"city": {"type": "string"}, "days": {"type": "integer", "minimum": 1}},
    defer_loading=True,
)
TIDES = tool("get_tides", "Get the tide table for a port.", {"port": {"type": "string"}}, defer_loading=True)
# Deferred by the key outside "function", which the template also accepts.
STOCKS = {**tool("get_quote", "Get a stock quote.", {"symbol": {"type": "string"}}), "defer_loading": True}
# defer_loading false: listed up front, and the key never reaches the prompt.
NEWS = tool("get_news", "Get the headlines.", {"topic": {"type": "string"}}, defer_loading=False)

CATALOG = [SEARCH, WEATHER, FORECAST, TIDES, STOCKS, NEWS]


def call(name: str, **arguments) -> dict:
    """An assistant message that calls one tool."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def ref(name: str) -> dict:
    """A tool_reference item, as a tool search result returns it."""
    return {"type": "tool_reference", "name": name}


SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
ASK = {"role": "user", "content": "Will Kobe have high tide and rain tomorrow?"}

TEMPLATES = [
    {"name": "deferred_catalog", "messages": [SYSTEM, ASK], "tools": CATALOG},
    {"name": "all_deferred_with_system", "messages": [SYSTEM, ASK], "tools": [FORECAST, TIDES]},
    {"name": "all_deferred_without_system", "messages": [ASK], "tools": [FORECAST, STOCKS]},
    {
        "name": "tool_reference_round_trip",
        "messages": [
            SYSTEM,
            ASK,
            call("tool_search", query="tide forecast"),
            {"role": "tool", "content": [ref("get_tides"), ref("get_forecast")]},
            call("get_tides", port="Kobe"),
            {"role": "tool", "content": "High tide 06:12, 18:40"},
        ],
        "tools": CATALOG,
    },
    {
        "name": "tool_reference_mixed",
        "messages": [
            ASK,
            call("tool_search", query="tide"),
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "Found 3 tools."},
                    ref("get_tides"),
                    ref("get_quote"),
                    "Pick one.",
                    ref("get_forecast"),
                    {"score": 0.5, "note": "ranked"},
                ],
            },
        ],
        "tools": CATALOG,
    },
    {
        "name": "result_tools",
        "messages": [
            ASK,
            call("tool_search", query="tide"),
            {"role": "tool", "content": [{"result": {"tools": [TIDES, FORECAST["function"]]}}]},
            call("tool_search", query="quote"),
            {"role": "tool", "content": [{"result": {"tools": STOCKS}}]},
            call("tool_search", query="news"),
            {"role": "tool", "content": [{"result": {"tools": "get_news: the headlines"}}]},
        ],
        "tools": CATALOG,
    },
    {
        "name": "additional_tools_field",
        "messages": [
            ASK,
            call("tool_search", query="tide"),
            {"role": "tool", "content": "ignored", "additional_tools": [TIDES, FORECAST]},
            call("tool_search", query="quote"),
            {"role": "tool", "content": "unused", "additional_tools": STOCKS},
            call("tool_search", query="news"),
            {"role": "tool", "additional_tools": True, "content": '{"name": "get_news"}'},
            call("tool_search", query="more"),
            {"role": "tool", "additional_tools": True, "content": "<additional_tools>\nraw\n</additional_tools>"},
            call("tool_search", query="again"),
            {"role": "tool", "additional_tools": True, "content": "<tool_response>\nas is\n</tool_response>"},
            call("tool_search", query="list"),
            {"role": "tool", "additional_tools": True, "content": [TIDES]},
            call("tool_search", query="empty"),
            {"role": "tool", "additional_tools": True, "content": ""},
        ],
        "tools": CATALOG,
    },
    {
        "name": "merged_tool_messages",
        "messages": [
            ASK,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "tool_search", "arguments": {"query": "tide"}}},
                    {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Kobe"}}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "first"},
            {"role": "tool", "tool_call_id": "a", "content": ""},
            {"role": "tool", "tool_call_id": "a", "content": [ref("get_tides")]},
            {"role": "tool", "tool_call_id": "a", "content": {"type": "text", "text": "last"}},
            {"role": "tool", "tool_call_id": "b", "content": "Rain, 14 C"},
            {"role": "tool", "tool_call_id": "b", "content": "", "additional_tools": [FORECAST]},
        ],
        "tools": CATALOG,
    },
    {
        "name": "tool_content_shapes",
        "messages": [
            ASK,
            call("get_weather", city="Kobe"),
            {"role": "tool", "content": {"temperature": 14, "condition": "rain"}},
            call("get_weather", city="Osaka"),
            {"role": "tool", "content": {"type": "text", "text": "Sunny"}},
            call("get_weather", city="Nara"),
            {"role": "tool", "content": None},
            call("get_weather", city="Kyoto"),
            {"role": "tool", "content": ["a", {"type": "text", "text": "b"}, [1, 2]]},
            call("get_weather", city="Uji"),
            {"role": "tool", "content": 42},
        ],
        "tools": [WEATHER],
    },
]


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    cases = []
    for case in TEMPLATES:
        prompt = tokenizer.apply_chat_template(
            case["messages"],
            tools=case["tools"],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        cases.append({**case, "enable_thinking": False, "prompt": prompt})

    args.out.write_text(json.dumps({"templates": cases}, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(cases)} template cases to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
