#!/usr/bin/env python3
"""Validate the Q4_0 audio encoder with pinned official Torch graph code.

Default: generate a tiny synthetic GGUF tensor fixture and upstream stage
outputs, without checkpoint weights. With --gguf, dequantize a local official
mmproj using llama.cpp's gguf-py (--gguf-python) and write real stage outputs.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import transformers
from transformers import AutoModel, Lfm2Config
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.cache_utils import DynamicCache

REVISION = '0295e4130594c06800fd25451d97139ac2da1aff'
SOURCE = f'https://huggingface.co/LiquidAI/LFM3-Audio-2B/resolve/{REVISION}/modeling_lfm3_audio.py'


def gguf_name(name: str) -> str:
    if name.startswith('downsample.'):
        name = name.removeprefix('downsample.')
        for old, new in {'conv_in': 'conv1d.0', 'conv_mid': 'conv1d.2',
                         'lin_mid': 'conv1d.3', 'conv_end': 'conv1d.5',
                         'lin_end': 'conv1d.6', 'lin_out': 'pre_encode.out'}.items():
            if name.startswith(old + '.'):
                return 'a.' + new + name[len(old):]
    if name == 'lfm.embedding_norm.weight':
        return 'a.post_ln.weight'
    if name.startswith('lfm.layers.'):
        parts = name.split('.')
        suffix = '.'.join(parts[3:])
        for old, new in {'operator_norm': 'ln1', 'ffn_norm': 'ln2',
                'conv.in_proj': 'conv_pw1', 'conv.conv': 'conv_dw', 'conv.out_proj': 'conv_pw2',
                'self_attn.q_proj': 'attn_q', 'self_attn.k_proj': 'attn_k',
                'self_attn.v_proj': 'attn_v', 'self_attn.out_proj': 'attn_out',
                'self_attn.q_layernorm': 'attn_q_norm', 'self_attn.k_layernorm': 'attn_k_norm',
                'feed_forward.w1': 'ffn_gate', 'feed_forward.w3': 'ffn_up',
                'feed_forward.w2': 'ffn_down'}.items():
            if suffix.startswith(old + '.'):
                return f'a.blk.{parts[2]}.{new}{suffix[len(old):]}'
    for old, new in {'layer_norm': '0', 'linear_1': '1', 'linear_2': '3'}.items():
        if name.startswith(old + '.'):
            return 'mm.a.mlp.' + new + name[len(old):]
    raise ValueError(name)


def quantized(name: str) -> bool:
    return name.endswith('.weight') and (
        name in ('a.pre_encode.out.weight', 'mm.a.mlp.1.weight', 'mm.a.mlp.3.weight') or
        name.startswith('a.blk.') and not any(x in name for x in ('norm', '.ln', 'conv_dw')))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('test/fixtures/audio/encoder_oracle.json'))
    parser.add_argument('--gguf', type=Path)
    parser.add_argument('--gguf-python', type=Path)
    parser.add_argument('--stream-out', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    source = urllib.request.urlopen(SOURCE).read()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in
             ('Lfm3AudioDownsampler', 'Lfm3AudioEncoder', 'Lfm3AudioProjector',
              'Lfm3AudioEncoderStreamingCache')]
    scope = dict(torch=torch, nn=nn, AutoModel=AutoModel, copy=copy, DynamicCache=DynamicCache,
                 create_sliding_window_causal_mask=create_sliding_window_causal_mask)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[])), SOURCE, 'exec'), scope)
    reader = None
    if args.gguf:
        sys.path.insert(0, str(args.gguf_python))
        import gguf
        reader = gguf.GGUFReader(str(args.gguf))
        tensors = {t.name: t for t in reader.tensors}
        field = lambda k: reader.fields['clip.audio.' + k].contents()
        hidden, inter, heads, kv, proj, ch, window = (field('embedding_length'),
            field('feed_forward_length'), field('attention.head_count'), field('attention.head_count_kv'),
            field('projection_dim'), tensors['a.conv1d.0.weight'].shape[3].item(), field('window_size'))
        layers = ['conv' if f'a.blk.{i}.conv_dw.weight' in tensors else 'full_attention'
                  for i in range(field('block_count'))]
    else:
        hidden, inter, heads, kv, proj, ch, window = 32, 64, 4, 2, 32, 4, 2
        layers = ['conv', 'full_attention', 'conv']
    text = Lfm2Config(hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=len(layers), num_attention_heads=heads, num_key_value_heads=kv,
        layer_types=layers, block_auto_adjust_ff_dim=False, conv_L_cache=3,
        rope_parameters={'rope_type': 'default', 'rope_theta': 1000000.0}, norm_eps=1e-5)
    text._attn_implementation = 'eager'
    config = SimpleNamespace(num_mel_bins=128, conv_channels=ch, hidden_size=hidden,
        text_config=text, sliding_window=window)
    encoder = scope['Lfm3AudioEncoder'](config).eval()
    projector = scope['Lfm3AudioProjector'](SimpleNamespace(audio_encoder_config=config,
        text_config=SimpleNamespace(hidden_size=proj))).eval()
    fixture_tensors = []
    for module in (encoder, projector):
        for name, param in module.named_parameters():
            g = gguf_name(name)
            shape = list(param.shape)
            if 'conv_dw' in g: shape = [hidden, 3]
            if g.startswith('a.conv1d.') and g.endswith('.bias'): shape = [ch, 1, 1]
            if g == 'a.pre_encode.out.weight': shape = [hidden, ch * 16]
            if reader:
                tensor = tensors[g]
                values = gguf.dequantize(tensor.data, tensor.tensor_type).reshape(param.shape)
            else:
                count = math.prod(shape)
                seed = sum((i + 1) * ord(c) for i, c in enumerate(g)) % 997
                if quantized(g):
                    blocks = bytearray()
                    values = []
                    for b in range(count // 32):
                        scale = (b % 3 + 1) / 256
                        blocks += np.float16(scale).tobytes()
                        packed = [(seed + b * 11 + i * 37) % 256 for i in range(16)]
                        blocks += bytes(packed)
                        values.extend([scale * ((x & 15) - 8) for x in packed])
                        values.extend([scale * ((x >> 4) - 8) for x in packed])
                    raw = bytes(blocks)
                    kind = 'q4_0'
                else:
                    norm = ('norm' in g or '.ln' in g or g == 'mm.a.mlp.0.weight') and g.endswith('.weight')
                    values = [(1 if norm else 0) + ((seed + i * 7 + i // 3 * 11) % 37 - 18) /
                              (256 if norm else 1024) for i in range(count)]
                    raw = np.array(values, dtype='<f4').tobytes()
                    kind = 'f32'
                values = np.array(values, dtype=np.float32).reshape(param.shape)
                fixture_tensors.append(dict(name=g, shape=shape, type=kind,
                    bytes=base64.b64encode(raw).decode()))
            param.data.copy_(torch.from_numpy(np.array(values, copy=True)))
    cases = []
    with torch.no_grad():
        for frames in ([1, 8, 9, 33, 265] if reader else [1, 8, 9, 33, 65]):
            features = torch.tensor([((i * 17 + i // 5 * 3) % 127 - 63) / 64
                for i in range(128 * frames)], dtype=torch.float32).reshape(1, 128, frames)
            captures = {}
            handles = [encoder.downsample.register_forward_hook(lambda m, a, o:
                captures.update(downsample=o.transpose(1, 2).flatten().tolist()))]
            for i, layer in enumerate(encoder.lfm.layers):
                handles.append(layer.register_forward_hook(lambda m, a, o, i=i:
                    captures.update({f'layer.{i}': o.flatten().tolist()})))
            encoded, _ = encoder(features, torch.tensor([frames]))
            captures['embedding'] = encoded.flatten().tolist()
            captures['projected'] = projector(encoded).flatten().tolist()
            for handle in handles: handle.remove()
            cases.append(dict(frames=frames, stages=captures))
    metadata = {'general.architecture': 'clip', 'clip.audio.projector_type': 'lfm3a',
        'clip.has_audio_encoder': 1, 'clip.audio.embedding_length': hidden,
        'clip.audio.feed_forward_length': inter, 'clip.audio.block_count': len(layers),
        'clip.audio.attention.head_count': heads, 'clip.audio.attention.head_count_kv': kv,
        'clip.audio.attention.layer_norm_epsilon': 1e-5, 'clip.audio.num_mel_bins': 128,
        'clip.audio.projection_dim': proj, 'clip.audio.window_size': window}
    result = dict(source=SOURCE, source_sha256=hashlib.sha256(source).hexdigest(),
        torch=torch.__version__, transformers=transformers.__version__, metadata=metadata,
        tensors=fixture_tensors, cases=cases)
    if args.gguf:
        result['gguf_sha256'] = hashlib.file_digest(args.gguf.open('rb'), 'sha256').hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    stream_cases = []
    with torch.no_grad():
        for frames in [0, 1, 7, 8, 9, 16, 33, 65] + ([265] if reader else []):
            features = torch.tensor([((i * 17 + i // 5 * 3) % 127 - 63) / 64
                for i in range(128 * frames)], dtype=torch.float32).reshape(1, 128, frames)
            cache = None
            chunks = []
            at = 0
            sizes = [0]
            pattern = [1, 7, 8, 3, 17]
            while at < frames:
                count = min(pattern[(len(sizes) - 1) % len(pattern)], frames - at)
                sizes.append(count)
                at += count
            at = 0
            joined = []
            for count, final in [(size, False) for size in sizes] + [(0, True)]:
                encoded, cache = encoder.stream(features[..., at:at + count], cache, is_final=final)
                projected = projector(encoded)
                chunks.append(dict(input_frames=count, is_final=final,
                    output_tokens=projected.shape[1], projected=projected.flatten().tolist()))
                joined.append(projected)
                at += count
            if frames:
                batch, _ = encoder(features, torch.tensor([frames]))
                batch = projector(batch)
                torch.testing.assert_close(torch.cat(joined, dim=1), batch, atol=2e-4, rtol=2e-4)
            stream_cases.append(dict(frames=frames, chunks=chunks))
    stream_result = {k: result[k] for k in ('source', 'source_sha256', 'torch', 'transformers', 'metadata')}
    stream_result['cases'] = stream_cases
    if reader: stream_result['gguf_sha256'] = result['gguf_sha256']
    stream_out = args.stream_out or args.out.with_name('encoder_stream_oracle.json')
    stream_out.parent.mkdir(parents=True, exist_ok=True)
    stream_out.write_text(json.dumps(stream_result, indent=2) + '\n')


if __name__ == '__main__':
    main()
