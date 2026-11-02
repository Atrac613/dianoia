#!/usr/bin/env python3
"""Run the seven existing visual prompts directly against official QAD GGUFs.

Uses the supplied llama.cpp mtmd CLI, greedy decoding, and the GGUF's Jinja
chat template with thinking explicitly disabled. Timing is diagnostic only;
this reference's optimized vision encoder differs from dianoia's CPU encoder.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from prepare_vl_qad import read_gguf


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cli', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--mmproj', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('--out must be new')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    template = read_gguf(args.model)['metadata']['tokenizer.chat_template']
    old = '{%- set enable_thinking = enable_thinking | default(true) -%}'
    if old not in template:
        raise ValueError('unexpected thinking control in GGUF template')
    template = template.replace(old, '{%- set enable_thinking = false -%}')
    baseline = json.loads(args.baseline.read_text())
    cases = baseline['runs'][0]['cases']
    record = dict(settings=dict(temperature=0, seed=123, repeat_penalty=1.0,
        threads=4, context_length=512, max_new_tokens=32, language_backend='Metal',
        vision_backend='CPU', thinking=False),
        template_sha256=hashlib.sha256(template.encode()).hexdigest(),
        cli_sha256=hashlib.sha256(args.cli.read_bytes()).hexdigest(), cases=[])
    for case in cases:
        actual_hash = hashlib.sha256(Path(case['image']).read_bytes()).hexdigest()
        if actual_hash != case['image_sha256']:
            raise ValueError(f"image differs from baseline: {case['name']}")
        command = [str(args.cli), '-m', str(args.model), '--mmproj', str(args.mmproj),
            '--image', case['image'], '-p', case['prompt'], '--temp', '0', '-s', '123',
            '--repeat-penalty', '1', '-n', '32', '-c', '512', '-t', '4', '-ngl', '99',
            '--no-mmproj-offload', '--no-warmup', '--jinja', '--chat-template', template]
        start = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        log = args.out.parent / f"reference-{case['name']}.log"
        log.write_text(result.stderr)
        answer = result.stdout.strip()
        if not answer:
            raise ValueError(f"empty answer: {case['name']}")
        record['cases'].append(dict(name=case['name'], prompt=case['prompt'],
            image=case['image'], image_sha256=case['image_sha256'],
            text=answer, wall_s=time.monotonic()-start))
        args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        print(case['name'], repr(answer), flush=True)
    record['finished'] = True
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
