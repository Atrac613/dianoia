#!/usr/bin/env python3
"""Generate deterministic RGB preprocessing hashes from upstream processor code.

Requires torch and torchvision, installed outside the Dart package. Downloads
only the pinned processor source; never downloads weights. Uses the actual
smart_resize and patchify definitions, torchvision uint8 antialiased bicubic
resize, and the HF backend's fused (pixel - 127.5) / 127.5 normalization.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import urllib.request
from pathlib import Path

import torch
import torchvision
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvf

REVISION = 'ba44bd51db20a2753f28c273a1571ccaa210d954'
SOURCE = f'https://huggingface.co/LiquidAI/LFM3-VL-2B/raw/{REVISION}/image_processing_lfm3_vl.py'


def main() -> None:
    """Preprocesses the synthetic cases upstream's way and writes their hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='optional local upstream processor source')
    parser.add_argument('--out', type=Path, default=Path('test/fixtures/vision/image_oracle.json'))
    args = parser.parse_args()
    source = args.source.read_text() if args.source else urllib.request.urlopen(SOURCE).read().decode()
    tree = ast.parse(source)
    resize = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'smart_resize')
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Lfm3VlImageProcessor')
    patchify = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'patchify')
    scope = {'math': math}
    exec(compile(ast.Module(body=[resize, patchify], type_ignores=[]), SOURCE, 'exec'), scope)
    cases = []
    specs = [
        ('unchanged', 8, 12, 2, 16, 256),
        ('upsample', 3, 5, 2, 64, 256),
        ('downsample', 37, 53, 2, 16, 256),
        ('mixed_axes', 6, 10, 2, 16, 256),
        ('singleton', 1, 1, 2, 16, 256),
        ('width_only', 8, 11, 2, 16, 256),
        ('height_only', 11, 8, 2, 16, 256),
        ('round_even', 10, 18, 2, 16, 256),
        ('official_defaults', 257, 129, 16, 65536, 16777216),
    ]
    for name, h, w, patch, minimum, maximum in specs:
        rgb = torch.tensor([(i * 37 + (i // 7) * 13) % 256 for i in range(h * w * 3)], dtype=torch.uint8)
        image = rgb.reshape(h, w, 3).permute(2, 0, 1).unsqueeze(0)
        oh, ow = scope['smart_resize'](h, w, factor=patch * 2, min_pixels=minimum, max_pixels=maximum)
        resized = tvf.resize(image, [oh, ow], interpolation=InterpolationMode.BICUBIC, antialias=True)
        normalized = tvf.normalize(resized.float(), [127.5]*3, [127.5]*3)
        patches, gh, gw = scope['patchify'](None, normalized, patch, 2, 2)
        cases.append(dict(name=name, height=h, width=w, patch_size=patch,
                          min_pixels=minimum, max_pixels=maximum,
                          resized_height=oh, resized_width=ow, grid=[1, gh, gw],
                          sha256=hashlib.sha256(patches.numpy().astype('<f4').tobytes()).hexdigest()))
    args.out.write_text(json.dumps(dict(source=SOURCE, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                      torch=torch.__version__, torchvision=torchvision.__version__, cases=cases), indent=2)+'\n')


if __name__ == '__main__':
    main()
