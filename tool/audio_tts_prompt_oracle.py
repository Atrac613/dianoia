#!/usr/bin/env python3
"""Render TTS prompts with the pinned local language GGUF's own template."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from transformers.utils.chat_template_utils import _compile_jinja_template

REV = '02355c1a141e464a064f2ab2c276638dbcb7f38a'
SHA = '112e4acb79be79dcd8df5749495614817e51c064cd9ca576aead02b64b2b7584'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--gguf-python', type=Path, required=True)
    parser.add_argument('--out', type=Path,
                        default=Path('test/fixtures/audio/tts_prompt_oracle.json'))
    args = parser.parse_args()
    with args.gguf.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != SHA:
        raise ValueError('Expected the pinned official Q4_0 language GGUF')
    sys.path.insert(0, str(args.gguf_python))
    import gguf
    reader = gguf.GGUFReader(str(args.gguf))
    source = reader.fields['tokenizer.chat_template'].contents()
    template = _compile_jinja_template(source)
    cases = []
    for language, name, voice in [('en', 'American English', 'female'),
                                  ('ja', 'Japanese', 'female'),
                                  ('en-gb', 'British English', 'male')]:
        text = 'Hello world.'
        prompt = template.render(
            messages=[{'role': 'system',
                       'content': f'Perform TTS in {name} using a {voice} voice.'},
                      {'role': 'user', 'content': text}],
            bos_token='<|startoftext|>', add_generation_prompt=True)
        cases.append(dict(text=text, language=language, voice=voice,
                          prompt=prompt + '<|audio_start|>'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        source='LiquidAI/LFM3-Audio-2B-GGUF', revision=REV,
        file_sha256=SHA, template_sha256=hashlib.sha256(source.encode()).hexdigest(),
        cases=cases), indent=2) + '\n')


if __name__ == '__main__':
    main()
