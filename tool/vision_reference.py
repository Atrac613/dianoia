#!/usr/bin/env python3
"""Dump real-checkpoint CPU float32 VL inference for comparison with vision-run.

Requires transformers==5.21.0, torch, torchvision and Pillow. Loads only local
checkpoint files and executes their custom code. RGB decoding includes EXIF
orientation. Outputs and weights belong in ignored build/ directories.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-new', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--think', action='store_true')
    args = parser.parse_args()
    if args.out.exists() or args.max_new < 0:
        parser.error('--out must be new and --max-new nonnegative')
    import torch
    import transformers
    from PIL import Image, ImageOps
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.out.mkdir(parents=True)
    image = ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
    (args.out/'input.rgb').write_bytes(image.tobytes())
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image}, {'type': 'text', 'text': args.prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
        enable_thinking=args.think, tokenize=True, return_dict=True, return_tensors='pt')
    model = AutoModelForImageTextToText.from_pretrained(args.model, trust_remote_code=True,
        local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval()
    def dump(name, tensor):
        tensor.detach().float().cpu().contiguous().numpy().astype('<f4').tofile(args.out/name)
    stages = []
    def hook(stage):
        def save(module, inputs, output):
            stages.append(stage)
            dump(f'vision_{stage}.f32', output)
        return save
    tower = model.model.vision_tower
    hooks = [tower.embedding_norm.register_forward_hook(hook('embedding')),
        model.model.multi_modal_projector.register_forward_hook(hook('projected'))]
    hooks.extend(layer.register_forward_hook(hook(f'layer.{i}')) for i, layer in enumerate(tower.layers))
    dump('pixels.f32', inputs['pixel_values'])
    inputs['input_ids'].cpu().numpy().astype('<i4').tofile(args.out/'ids.i32')
    started = time.perf_counter()
    with torch.inference_mode():
        result = model(**inputs, use_cache=True)
        prefill_seconds = time.perf_counter()-started
        for handle in hooks:
            handle.remove()
        logits = result.logits[:, -1, :]
        dump('logits.f32', logits)
        generated = []
        all_logits = []
        cache = result.past_key_values
        started = time.perf_counter()
        eos = model.config.eos_token_id
        stop = set(eos if isinstance(eos, list) else [eos])
        for step in range(args.max_new):
            all_logits.append(logits.clone())
            token = int(logits.argmax(-1).item())
            if token in stop:
                break
            generated.append(token)
            result = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            logits = result.logits[:, -1, :]
        decode_seconds = time.perf_counter()-started
        if all_logits:
            dump('gen_logits.f32', torch.cat(all_logits, dim=0))
        else:
            (args.out/'gen_logits.f32').write_bytes(b'')
    import numpy as np
    np.asarray(generated, dtype='<i4').tofile(args.out/'gen.i32')
    text = processor.tokenizer.decode(generated, skip_special_tokens=False)
    (args.out/'meta.json').write_text(json.dumps(dict(complete=True, prompt=args.prompt,
        thinking=args.think, dtype='float32', torch=torch.__version__, transformers=transformers.__version__,
        height=image.height, width=image.width, grid_thw=inputs['image_grid_thw'][0].tolist(),
        prompt_tokens=inputs['input_ids'].shape[-1], stages=stages,
        generated_ids=generated, text=text, prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds), indent=2)+'\n')
    print(text, flush=True)
    print(f'wrote upstream evidence to {args.out}', flush=True)


if __name__ == '__main__':
    main()
