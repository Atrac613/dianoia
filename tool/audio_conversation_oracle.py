#!/usr/bin/env python3
"""Execute the pinned official interleaving loop with scripted tokens/frames.

No model weights: the AST of the actual per-row generation block is executed,
with sampling/embedding arithmetic replaced by deterministic inputs/zero rows.
"""
import ast
import hashlib
import json
import urllib.request
from collections import deque
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import jinja2
import torch

BASE = 'https://huggingface.co/LiquidAI/LFM3-Audio-2B/resolve/0295e4130594c06800fd25451d97139ac2da1aff/'


class Mode(Enum):
    TEXT = 'text'
    AUDIO = 'audio'
    INTERLEAVED = 'mixed'


def main():
    sources = {n: urllib.request.urlopen(BASE + n).read() for n in
               ['modeling_lfm3_audio.py', 'processing_lfm3_audio.py']}
    tree = ast.parse(sources['modeling_lfm3_audio.py'])
    generate = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_generate')
    row = next(n for n in ast.walk(generate) if isinstance(n, ast.For) and
               isinstance(n.target, ast.Name) and n.target.id == 'row' and
               any(isinstance(x, ast.If) and ast.unparse(x.test) == 'row in sampled_frames' for x in n.body))
    block = compile(ast.fix_missing_locations(ast.Module(body=[row], type_ignores=[])), '<upstream row>', 'exec')
    config = SimpleNamespace(audio_eos_code=9, audio_out_token_id=105,
                             text_end_token_id=102, audio_start_token_id=104,
                             mixed_start_token_id=103, interleaved_n_text=6, interleaved_n_audio=12)
    schedules = [
        ('alternating_and_tail', [7] * 6 + ['audio'] * 12 + [102, 'audio', 'eos', 101]),
        ('early_text_end', [7, 102, 'audio', 'audio', 'eos', 101]),
        ('eos_mid_burst_and_reopen', [7] * 6 + ['eos', 103] + [7] * 6 + ['audio', 'eos', 102]),
        ('turn_end_inside_mixed', [101]),
        ('audio_only_after_eos', [102, 'eos', 104, 'audio', 'eos', 101]),
    ]
    cases = []
    for name, script in schedules:
        env = dict(torch=torch, Lfm3AudioGenerationMode=Mode, config=config,
                   batch_size=1, modes=[Mode.INTERLEAVED], interleaving=[True],
                   slots_left=[6], speaking=[False], text_done=[False], finished=[False],
                   text_history=[deque(maxlen=4096)], audio_history=[deque(maxlen=4096)],
                   frames=[[]], text_out=[[]], audio_streams=[None], text_streams=[None],
                   text_only=[False], text_only_suppress=[], suppress=[], eos_ids=[101],
                   text_temperature=None, text_top_k=None, text_top_p=None, text_min_p=None,
                   generation_config=SimpleNamespace(repetition_penalty=1.0), text_warpers=None,
                   return_audio_eos=True, embed_tokens=lambda t: torch.zeros(8))
        rows = []
        for item in script:
            is_audio = env['modes'][0] is Mode.AUDIO or (env['modes'][0] is Mode.INTERLEAVED and env['speaking'][0])
            assert is_audio == isinstance(item, str), (name, item)
            env['interleaving'] = [env['modes'][0] is Mode.INTERLEAVED]
            env['sampled_frames'] = {0: torch.tensor([9 if item == 'eos' else 0] + [0]*11)} if is_audio else {}
            env['next_embeds'] = torch.zeros(1, 1, 8)
            env['hidden'] = torch.zeros(1, 8)
            env['self'] = SimpleNamespace(lm_head=lambda x: torch.zeros(128), _sample=lambda *a, **kw: torch.tensor(item),
                                         model=SimpleNamespace(get_audio_out_embeds=lambda x: torch.zeros(8)))
            exec(block, env)
            rows.append(dict(input=item, mode=env['modes'][0].value, slots_left=env['slots_left'][0],
                             speaking=env['speaking'][0], text_done=env['text_done'][0], finished=env['finished'][0]))
            if item == 'eos': assert env['frames'][0][-1].tolist() == [9]*12
        cases.append(dict(name=name, rows=rows))
    template_tree = ast.parse(sources['processing_lfm3_audio.py'])
    template_node = next(n for n in template_tree.body if isinstance(n, ast.Assign) and
                         any(isinstance(t, ast.Name) and t.id == 'DEFAULT_CHAT_TEMPLATE' for t in n.targets))
    template = jinja2.Environment().from_string(ast.literal_eval(template_node.value))
    prompt = template.render(messages=[{'role': 'system', 'content': 'Respond with interleaved text and audio.'},
                                       {'role': 'user', 'content': [{'type': 'audio'}]}],
                             bos_token='<|startoftext|>', add_generation_prompt=True) + '<|mixed_start|>'
    result = dict(sources={n: dict(url=BASE+n, sha256=hashlib.sha256(s).hexdigest()) for n,s in sources.items()},
                  torch=torch.__version__, first_prompt=prompt, cases=cases)
    Path('test/fixtures/audio/conversation_oracle.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
