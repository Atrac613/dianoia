#!/usr/bin/env python3
"""Generate LFM3-Audio frontend fixtures using pinned official Python code.

Use the environment in requirements-vision-validation.txt. Downloads source
only, never weights. The remote module is imported only by this explicit tool.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch
import transformers

REVISION = '0295e4130594c06800fd25451d97139ac2da1aff'
SOURCE = f'https://huggingface.co/LiquidAI/LFM3-Audio-2B/resolve/{REVISION}/feature_extraction_lfm3_audio.py'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('test/fixtures/audio/features_oracle.json'))
    args = parser.parse_args()
    source = urllib.request.urlopen(SOURCE).read()
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'official_audio_features.py'
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location('official_audio_features', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        extractor = module.Lfm3AudioFeatureExtractor()
        cases = []
        for name, size in [('empty', 0), ('one', 1), ('short', 399),
                           ('window', 400), ('tail', 401), ('before_hop', 559),
                           ('hop', 560), ('after_hop', 561), ('long', 2000),
                           ('silence', 2000)]:
            pcm = torch.tensor([0.0 if name == 'silence' else
                                (((i * 73 + (i // 7) * 19) % 1024) - 512) / 1024
                                for i in range(size)], dtype=torch.float32)
            result = extractor(pcm, sampling_rate=16000, return_tensors='pt')
            features = result['input_features'][0]
            cases.append(dict(name=name, samples=size, frames=features.shape[-1],
                              encoder_tokens=extractor.num_encoder_tokens(features.shape[-1]),
                              values=features.flatten().tolist()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(source=SOURCE,
        source_sha256=hashlib.sha256(source).hexdigest(), torch=torch.__version__,
        transformers=transformers.__version__, cases=cases), indent=2) + '\n')


if __name__ == '__main__':
    main()
