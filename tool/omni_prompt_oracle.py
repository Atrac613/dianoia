#!/usr/bin/env python3
"""Render LFM3-Omni prompts with the pinned official chat template.

Writes test/fixtures/omni/prompt_oracle.json. The template is rendered as data
with jinja2 and tokenized with the pinned tokenizer.json; no repository code
is executed and no weights are downloaded. Placeholder expansion (repeating
<|audio|>/<|image|> per encoder row) is the processor's job and is tested in
Dart against processing_lfm3_omni.py's documented behavior.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

import jinja2
from tokenizers import Tokenizer

REVISION = '21d75123f86c1f7ffb4bef1ea202710c4e7ebe67'
BASE = f'https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/{REVISION}/'

CASES = {
    'asr_english': [
        {'role': 'system', 'content': 'Perform ASR in English.'},
        {'role': 'user', 'content': [{'type': 'audio'}]},
    ],
    'asr_auto': [
        {'role': 'system', 'content': 'Perform ASR.'},
        {'role': 'user', 'content': [{'type': 'audio'}]},
    ],
    'tts': [
        {'role': 'system',
         'content': 'Perform TTS in American English using a female voice.'},
        {'role': 'user', 'content': 'The quick brown fox jumps over the lazy dog'},
    ],
    'interleaved_audio': [
        {'role': 'system', 'content': 'Respond with interleaved text and audio.'},
        {'role': 'user', 'content': [{'type': 'audio'}]},
    ],
    'image_text': [
        {'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': 'Describe this image.'}]},
    ],
    'image_spoken_question': [
        {'role': 'system', 'content': 'Respond in text.'},
        {'role': 'user', 'content': [{'type': 'image'}, {'type': 'audio'}]},
    ],
}


def fetch(name):
    data = urllib.request.urlopen(BASE + name).read()
    return data, hashlib.sha256(data).hexdigest()


def main():
    template_bytes, template_sha = fetch('chat_template.jinja')
    tokenizer_bytes, tokenizer_sha = fetch('tokenizer.json')
    env = jinja2.Environment(extensions=['jinja2.ext.loopcontrols'])

    def raise_exception(message):
        raise jinja2.TemplateError(message)

    env.globals['raise_exception'] = raise_exception
    # `{% generation %}` is a transformers extension; it only marks spans.
    source = template_bytes.decode().replace('{%- generation -%}', '') \
        .replace('{%- endgeneration -%}', '')
    template = env.from_string(source)
    tok = Tokenizer.from_str(tokenizer_bytes.decode())
    cases = []
    for name, messages in CASES.items():
        text = template.render(messages=messages, bos_token='<|startoftext|>',
                               add_generation_prompt=True)
        cases.append(dict(name=name, messages=messages, text=text,
                          ids=tok.encode(text, add_special_tokens=False).ids))
    markers = {t: tok.token_to_id(t) for t in [
        '<|vision_start|>', '<|vision_end|>', '<|image|>', '<|video|>',
        '<|audio|>', '<|audio_out|>', '<|audio_start|>', '<|mixed_start|>',
        '<|text_end|>', '<|im_start|>', '<|im_end|>', '<|startoftext|>',
        '<think>', '</think>']}
    out = Path('test/fixtures/omni/prompt_oracle.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        source=BASE,
        source_sha256={'chat_template.jinja': template_sha,
                       'tokenizer.json': tokenizer_sha},
        markers=markers, cases=cases), indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
