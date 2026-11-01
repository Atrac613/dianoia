#!/usr/bin/env python3
"""Unpack the official BF16 LFM3-VL mmproj for dianoia QAD validation.

Inverse of llama.cpp conversion/lfm3.py: preserve BF16/F32 payloads and
reassemble the two F16 temporal patchifier slices as exact F32 values.
No training or matrix requantization is performed here.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import struct


def read_gguf(path: Path) -> dict:
    with path.open('rb') as f:
        def read(fmt: str):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]
        def string() -> str:
            return f.read(read('Q')).decode()
        def value(t: int):
            if t == 8:
                return string()
            if t == 9:
                typ, count = read('I'), read('Q')
                for _ in range(count):
                    value(typ)
                return {'array_length': count}
            return read({0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f',
                         7:'?', 10:'Q', 11:'q', 12:'d'}[t])
        if f.read(4) != b'GGUF' or read('I') != 3:
            raise ValueError('expected GGUF v3')
        nt, nk = read('Q'), read('Q')
        metadata = {}
        for _ in range(nk):
            key = string()
            metadata[key] = value(read('I'))
        tensors = []
        for _ in range(nt):
            name = string()
            dims = [read('Q') for _ in range(read('I'))]
            tensors.append(dict(name=name, dims=dims, type=read('I'), offset=read('Q')))
        alignment = metadata.get('general.alignment', 32)
        start = (f.tell() + alignment - 1) // alignment * alignment
    return dict(metadata=metadata, data_start=start, tensors=tensors)


def canonical(name: str) -> str:
    projector = {'mm.0': 'linear_1', 'mm.2': 'linear_2', 'mm.input_norm': 'layer_norm'}
    for prefix, target in projector.items():
        if name.startswith(prefix + '.'):
            return 'model.multi_modal_projector.' + target + name[len(prefix):]
    if name == 'v.embedding_norm.weight':
        return 'model.vision_tower.embedding_norm.weight'
    match = re.fullmatch(r'v\.blk\.(\d+)\.(.+)\.weight', name)
    if not match:
        raise ValueError(f'unsupported tensor {name}')
    target = {'ln1':'operator_norm', 'ln2':'ffn_norm',
              'ffn_gate':'feed_forward.w1', 'ffn_up':'feed_forward.w3', 'ffn_down':'feed_forward.w2',
              'conv2d_in_proj':'conv.in_proj', 'conv2d_out_proj':'conv.out_proj',
              'conv2d_spatial':'conv.spatial_conv',
              'attn_q':'self_attn.q_proj', 'attn_k':'self_attn.k_proj',
              'attn_v':'self_attn.v_proj', 'attn_out':'self_attn.out_proj',
              'attn_q_norm':'self_attn.q_layernorm', 'attn_k_norm':'self_attn.k_layernorm'}[match[2]]
    return f'model.vision_tower.layers.{match[1]}.{target}.weight'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mmproj', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must be a new directory')
    gguf = read_gguf(args.mmproj)
    if gguf['metadata'].get('clip.vision.projector_type') != 'lfm3vl':
        raise ValueError('not an LFM3-VL projector')
    entries = []
    patch = [next(t for t in gguf['tensors'] if t['name'] == name)
             for name in ['v.patch_embd.weight', 'v.patch_embd.weight.1']]
    if any(t['type'] != 1 or t['dims'] != [16, 16, 3, 1024] for t in patch):
        raise ValueError('unsupported patchifier shape/type')
    for t in gguf['tensors']:
        if t in patch:
            continue
        dtype, size = {0:('F32', 4), 30:('BF16', 2)}[t['type']]
        entries.append((canonical(t['name']), list(reversed(t['dims'])), dtype,
                        gguf['data_start'] + t['offset'], math.prod(t['dims']) * size))
    args.out.mkdir(parents=True)
    for name in ['config.json', 'processor_config.json', 'tokenizer.json', 'LICENSE']:
        shutil.copyfile(args.config / name, args.out / name)
    patch_name = 'model.vision_tower.patchifier.weight'
    header, offset = {}, 0
    for name, shape, dtype, _, size in entries:
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset+size])
        offset += size
    patch_size = 1024 * 1536 * 4
    header[patch_name] = dict(dtype='F32', shape=[1024, 1536], data_offsets=[offset, offset+patch_size])
    data = json.dumps(header, separators=(',', ':')).encode()
    padded = (len(data) + 7) // 8 * 8
    with args.mmproj.open('rb') as src, (args.out/'model.safetensors').open('wb') as dst:
        dst.write(struct.pack('<Q', padded) + data + b' ' * (padded-len(data)))
        for _, _, _, start, size in entries:
            src.seek(start)
            remaining = size
            while remaining:
                chunk = src.read(min(4*1024*1024, remaining))
                if not chunk:
                    raise ValueError('truncated GGUF payload')
                dst.write(chunk)
                remaining -= len(chunk)
        for row in range(1024):
            for channel in range(3):
                for t in patch:
                    src.seek(gguf['data_start'] + t['offset'] + (row*3+channel)*256*2)
                    dst.write(struct.pack('<256f', *struct.unpack('<256e', src.read(512))))
    print(f'Wrote {len(header)} vision/projector tensors to {args.out}')


if __name__ == '__main__':
    main()
