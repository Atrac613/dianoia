#!/usr/bin/env python3
"""Reference outputs from pinned upstream vision operators with synthetic weights.

Requires torch, numpy and torchvision. No model weights are downloaded or saved.
The fixture contains shapes and outputs; test weights are generated at runtime
from tensor names and indices by the exact rational formula below.
"""
import argparse
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import torch
from torch import nn
from torch.nn import functional as F

REVISION = 'ba44bd51db20a2753f28c273a1571ccaa210d954'
SOURCE = f'https://huggingface.co/LiquidAI/LFM3-VL-2B/raw/{REVISION}/modeling_lfm3_vl.py'
CLASSES = ['Lfm3VlRMSNorm', 'Lfm3VlMLP', 'Lfm3VlVisionRotaryEmbedding',
           'Lfm3VlVisionAttention', 'Lfm3VlVisionConv2d', 'Lfm3VlVisionLayer',
           'Lfm3VlMultiModalProjector']
FUNCTIONS = ['rotate_half', 'apply_rotary_pos_emb_vision', 'repeat_kv', 'eager_attention_forward']


def fill(name, shape):
    count = 1
    for size in shape:
        count *= size
    seed = sum((i + 1) * ord(c) for i, c in enumerate(name)) % 997
    norm_weight = 'norm' in name and name.endswith('.weight')
    return torch.tensor([(1 if norm_weight else 0) +
                         ((seed + i * 7 + (i // 3) * 11) % 37 - 18) /
                         (256 if norm_weight else 64) for i in range(count)], dtype=torch.float32).reshape(shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--out', type=Path, default=Path('test/fixtures/vision/encoder_oracle.json'))
    args = parser.parse_args()
    source = args.source.read_text() if args.source else urllib.request.urlopen(SOURCE).read().decode()
    tree = ast.parse(source)
    nodes = []
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name in CLASSES:
            n.decorator_list = []
            nodes.append(n)
        elif isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS:
            nodes.append(n)
    scope = dict(torch=torch, nn=nn, F=F, GradientCheckpointingLayer=nn.Module,
                 ACT2FN={'gelu': F.gelu}, maybe_autocast=lambda *a, **k: nullcontext(),
                 is_flash_attention_requested=lambda c: False,
                 ALL_ATTENTION_FUNCTIONS=SimpleNamespace(get_interface=lambda name, default: default))
    # Avoid runtime resolution of HF typing-only annotations.
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes, type_ignores=[])), SOURCE, 'exec'), scope)
    json_config = json.loads(Path('test/fixtures/vision/config.json').read_text())
    json_config['vision_config'].update(hidden_size=16, intermediate_size=24,
        num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=3,
        layer_types=['conv', 'full_attention', 'conv'], patch_size=2)
    json_config['text_config'].update(hidden_size=8)
    json_config['projector_hidden_size'] = 12
    v = SimpleNamespace(**json_config['vision_config'], _attn_implementation='eager')
    config = SimpleNamespace(**{k: val for k, val in json_config.items() if k not in ['vision_config', 'text_config']},
                             vision_config=v, text_config=SimpleNamespace(**json_config['text_config']))
    modules = nn.ModuleDict({
        'patchifier': nn.Linear(24, 16, bias=False),
        'embedding_norm': scope['Lfm3VlRMSNorm'](16, v.norm_eps),
        'layers': nn.ModuleList([scope['Lfm3VlVisionLayer'](v, i) for i in range(3)]),
    })
    projector = scope['Lfm3VlMultiModalProjector'](config)
    shapes = {}
    for prefix, module in [('model.vision_tower.', modules), ('model.multi_modal_projector.', projector)]:
        for name, tensor in module.named_parameters():
            full_name = prefix + name
            shapes[full_name] = list(tensor.shape)
            tensor.data.copy_(fill(full_name, tensor.shape))
    rope = scope['Lfm3VlVisionRotaryEmbedding'](v)
    cases = []
    with torch.no_grad():
        for gh, gw in [(2, 2), (2, 4), (4, 6), (6, 4)]:
            h, w = gh * 2, gw * 2
            rgb = torch.tensor([(i * 37 + (i // 7) * 13) % 256 for i in range(h * w * 3)], dtype=torch.uint8)
            image = (rgb.reshape(h, w, 3).permute(2, 0, 1).float() - 127.5) / 127.5
            # The patch layout is independently validated by vision_image_oracle.py.
            patches = image.reshape(3, gh//2, 2, 2, gw//2, 2, 2).permute(1, 4, 2, 5, 0, 3, 6)
            patches = patches.unsqueeze(5).expand(-1, -1, -1, -1, -1, 2, -1, -1).reshape(gh*gw, 24)
            positions = torch.tensor([(by+dy, bx+dx) for by in range(0, gh, 2)
                                      for bx in range(0, gw, 2) for dy in range(2) for dx in range(2)])
            hidden = modules['embedding_norm'](modules['patchifier'](patches))
            captures = {'embedding': hidden.flatten().tolist()}
            position_embeddings = rope(hidden, positions)
            for i, layer in enumerate(modules['layers']):
                hidden = layer(hidden, torch.tensor([0, gh*gw], dtype=torch.int32),
                               [[gh, gw]], position_embeddings)
                captures[f'layer.{i}'] = hidden.flatten().tolist()
            captures['projected'] = projector(hidden).flatten().tolist()
            cases.append(dict(grid=[gh, gw], stages=captures))
    args.out.write_text(json.dumps(dict(source=SOURCE, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                      torch=torch.__version__, config=json_config, shapes=shapes, cases=cases), indent=2)+'\n')


if __name__ == '__main__':
    main()
