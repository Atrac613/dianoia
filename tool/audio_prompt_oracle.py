#!/usr/bin/env python3
"""Render the ASR subset with the pinned official processor's default template."""
import ast
import hashlib
import json
import urllib.request
from pathlib import Path

import jinja2

SOURCE = ('https://huggingface.co/LiquidAI/LFM3-Audio-2B/resolve/'
          '0295e4130594c06800fd25451d97139ac2da1aff/processing_lfm3_audio.py')


def main():
    source = urllib.request.urlopen(SOURCE).read()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == 'DEFAULT_CHAT_TEMPLATE' for t in n.targets))
    template = jinja2.Environment().from_string(ast.literal_eval(node.value))
    cases = []
    for language in (None, 'English', 'Japanese'):
        messages = [{'role': 'system', 'content': 'Perform ASR' +
                     (f' in {language}' if language else '') + '.'},
                    {'role': 'user', 'content': [{'type': 'audio'}]}]
        text = template.render(messages=messages, bos_token='<|startoftext|>', add_generation_prompt=True)
        cases.append(dict(language=language, text=text))
    Path('test/fixtures/audio/prompt_oracle.json').write_text(json.dumps(dict(
        source=SOURCE, source_sha256=hashlib.sha256(source).hexdigest(), cases=cases), indent=2) + '\n')


if __name__ == '__main__':
    main()
