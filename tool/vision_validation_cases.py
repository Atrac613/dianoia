#!/usr/bin/env python3
"""Prepare reproducible natural-image and OCR cases without storing image assets.

Requires Pillow. Supply the pinned scikit-image astronaut PNG and a local font
with Japanese glyphs. Font files are never copied. The manifest records source,
font and decoded image hashes; Pillow is pinned by requirements-vision-validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

SOURCE = 'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/astronaut.png'
SOURCE_SHA256 = '88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5'


def main() -> None:
    """Writes the photo and OCR images with their cases.json manifest."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--astronaut', type=Path, required=True)
    p.add_argument('--font', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('--out must be a new directory')
    source_hash = hashlib.sha256(args.astronaut.read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA256:
        p.error('astronaut source differs from the pinned asset')
    from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
    font36 = ImageFont.truetype(str(args.font), 36)
    font56 = ImageFont.truetype(str(args.font), 56)
    args.out.mkdir(parents=True)
    Image.open(args.astronaut).convert('RGB').resize((256, 256), Image.Resampling.BICUBIC).save(args.out/'astronaut.png')
    image = Image.new('RGB', (384, 192), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((24, 28), 'CODE', font=font36, fill='black')
    draw.text((24, 88), 'AB42', font=font56, fill='black')
    image.save(args.out/'ocr-latin.png')
    image = Image.new('RGB', (384, 192), 'white')
    ImageDraw.Draw(image).text((20, 40), '東京 123', font=font56, fill='black')
    image.save(args.out/'ocr-ja.png')
    cases = [
        dict(name='photo-en', image='astronaut.png',
             prompt='What is the person wearing? Answer in a short phrase.',
             expected_concept='astronaut clothing / spacesuit; assess specificity manually'),
        dict(name='photo-ja', image='astronaut.png',
             prompt='この人物は何を着ていますか。日本語で短く答えてください。',
             expected_concept='astronaut clothing / spacesuit; assess Japanese wording manually'),
        dict(name='ocr-latin', image='ocr-latin.png',
             prompt='Read the code after CODE. Return only the code.', expected_text='AB42'),
        dict(name='ocr-ja', image='ocr-ja.png',
             prompt='画像に書かれた文字を、そのまま書き起こしてください。', expected_text='東京 123'),
    ]
    for case in cases:
        image = Image.open(args.out/case['image']).convert('RGB')
        case.update(height=image.height, width=image.width,
                    rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest())
    manifest = dict(source=SOURCE, source_sha256=source_hash, pillow=pillow_version,
        font_name=unicodedata.normalize('NFC', args.font.name),
        font_sha256=hashlib.sha256(args.font.read_bytes()).hexdigest(), cases=cases)
    (args.out/'cases.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(f'Prepared {len(cases)} cases in {args.out}')


if __name__ == '__main__':
    main()
