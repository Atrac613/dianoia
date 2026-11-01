#!/usr/bin/env python3
"""Render pinned VL template and image-token expansion with Hugging Face.

Requires transformers and torch. Uses the lab tokenizer only to render strings;
no weights or tokenizer file are saved in the fixture. Tool-search cases reuse
the existing oracle inputs, with an image added to the first user message.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from transformers import AutoTokenizer

REVISION = 'ba44bd51db20a2753f28c273a1571ccaa210d954'
BASE = f'https://huggingface.co/LiquidAI/LFM3-VL-2B/raw/{REVISION}/'


def main() -> None:
    """Renders and expands each case with the pinned template and processor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('../dart-llm-lab/models/LFM3-1.7B'))
    parser.add_argument('--template', type=Path)
    parser.add_argument('--processor', type=Path)
    parser.add_argument('--out', type=Path, default=Path('test/fixtures/vision/prompt_oracle.json'))
    args = parser.parse_args()
    template = args.template.read_text() if args.template else urllib.request.urlopen(BASE+'chat_template.jinja').read().decode()
    processor = args.processor.read_text() if args.processor else urllib.request.urlopen(BASE+'processing_lfm3_vl.py').read().decode()
    tree = ast.parse(processor)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Lfm3VlProcessor')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'replace_image_token')
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), BASE+'processing_lfm3_vl.py', 'exec'), scope)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    image = {'type': 'image'}

    def text(value: str) -> dict:
        """A text content part."""
        return {'type': 'text', 'text': value}

    cases = [
        dict(name='image_then_text', messages=[{'role': 'user', 'content': [image, text('何が見える？')]}], counts=[6]),
        dict(name='two_images', messages=[{'role': 'system', 'content': 'Compare.'},
            {'role': 'user', 'content': [text('First: '), image, text(' Next: '), image]}], counts=[1, 6]),
        dict(name='image_mapping', messages=[{'role': 'user', 'content': image}], counts=[1]),
        dict(name='followup', messages=[{'role': 'user', 'content': [image, text('Read.')]},
            {'role': 'assistant', 'content': '<think>old</think>Answer.'},
            {'role': 'user', 'content': 'Explain.'}], counts=[1]),
        dict(name='preserve_thinking', preserve_thinking=True,
            messages=[{'role': 'user', 'content': [image, text('Read.')]},
            {'role': 'assistant', 'thinking': 'Look.', 'content': 'Answer.'},
            {'role': 'user', 'content': 'Explain.'}], counts=[1]),
        dict(name='tool_image', messages=[{'role': 'user', 'content': 'Look.'},
            {'role': 'tool', 'content': [image, text('Screenshot.')]}], counts=[1]),
    ]
    for original in json.loads(Path('test/fixtures/template_oracle.json').read_text())['templates']:
        c = copy.deepcopy(original)
        c.pop('prompt', None)
        c['name'] = 'with_image_'+c['name']
        first = next(m for m in c['messages'] if m['role'] == 'user')
        first['content'] = [image, text(first['content'])]
        c['counts'] = [1]
        cases.append(c)
    output = []
    for case in cases:
        for thinking in [True, False]:
            c = dict(case, name=case['name']+('_think' if thinking else '_no_think'), enable_thinking=thinking)
            prompt = tokenizer.apply_chat_template(c['messages'], chat_template=template,
                tools=c.get('tools'), tokenize=False, add_generation_prompt=True,
                enable_thinking=thinking, preserve_thinking=c.get('preserve_thinking', False))
            pieces = prompt.split('<|image|>')
            assert len(pieces) == len(c['counts'])+1
            grids = torch.tensor([[1, 2, 2*count] for count in c['counts']])
            proxy = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2), image_token='<|image|>')
            expanded = ''.join(piece + scope['replace_image_token'](proxy, {'image_grid_thw': grids}, i)
                               for i, piece in enumerate(pieces[:-1])) + pieces[-1]
            output.append(dict(c, prompt=prompt, expanded=expanded))
    args.out.write_text(json.dumps(dict(revision=REVISION,
        template_sha256=hashlib.sha256(template.encode()).hexdigest(),
        processor_sha256=hashlib.sha256(processor.encode()).hexdigest(),
        transformers=transformers.__version__, cases=output), ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
