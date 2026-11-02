#!/usr/bin/env python3
"""Execute pinned transcribe_live with scripted logits/encoder rows; no weights."""
import ast
import hashlib
import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace
import torch

URL = 'https://huggingface.co/LiquidAI/LFM3-Audio-2B/resolve/0295e4130594c06800fd25451d97139ac2da1aff/modeling_lfm3_audio.py'


def main():
    source = urllib.request.urlopen(URL).read()
    tree = ast.parse(source)
    state_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Lfm3AudioLiveTranscriptionState')
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'transcribe_live')
    method.decorator_list = []; method.name = 'run'
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    env = dict(torch=torch, DynamicCache=lambda **kw: object(),
               Lfm3AudioLiveTranscriptionOutput=lambda **kw: SimpleNamespace(**kw))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, state_cls, method], type_ignores=[])), '<upstream>', 'exec'), env)
    markers = {'<|audio|>': 15, '<|audio_out|>': 16, '<|audio_start|>': 12,
               '<|mixed_start|>': 13, '<|text_end|>': 14, '<|mixed_end|>': 17,
               '<|im_end|>': 11, '<|endoftext|>': 18, '<|im_start|>': 19}
    embed = torch.nn.Embedding(32, 2)
    embed.weight.data.copy_(torch.tensor([[i, 0] for i in range(32)], dtype=torch.float32))
    script = [
        dict(rows=0, final=False, limit=48, prompt=[1, 2], logits=[]),
        dict(rows=12, final=False, limit=48, logits=[[15,16,12,13,11,17,18,7],[14]]),
        dict(rows=12, final=False, limit=2, logits=[[8],[9],[10]]),
        dict(rows=0, final=False, limit=48, logits=[]),
        dict(rows=1, final=True, limit=96, logits=[[15,16,17],[10],[11]]),
    ]
    cases = []
    for name, chunks in [('blocks_and_final_tail', script),
                         ('final_without_new_audio', [script[1] | dict(prompt=[1,2]), dict(rows=0, final=True, limit=96, logits=[[18]])]),
                         ('final_turn_start_stop', [dict(rows=1, final=True, limit=96, prompt=[1,2], logits=[[19]])])]:
        state = None; outputs = []
        for chunk in chunks:
            inputs = []; scores = iter(chunk['logits'])
            def features(x, cache, final):
                cache = cache or SimpleNamespace(calls=0)
                cache.calls += 1
                return torch.tensor([[[-1., -2.]]*chunk['rows']]).reshape(1,chunk['rows'],2), cache
            def forward(state, embeddings):
                inputs.append(embeddings[0, :, 0].tolist())
                state.attention_mask = torch.cat([state.attention_mask, torch.ones(1,embeddings.shape[1],dtype=torch.long)], dim=1)
                return torch.zeros(1,2)
            def head(x):
                ranked = next(scores)
                logits = torch.full((1,32), -1000.)
                for i,token in enumerate(ranked): logits[0,token] = 100-i
                return logits
            config = SimpleNamespace(audio_encoder_config=SimpleNamespace(num_mel_bins=128),
                text_config=SimpleNamespace(eos_token_id=11,vocab_size=32),
                text_end_token_id=14,audio_start_token_id=12,mixed_start_token_id=13)
            model = SimpleNamespace(training=False, config=config, _placeholder_token_ids=[15,16],
                model=SimpleNamespace(check_speech_placeholders=lambda *a,**kw:None,get_audio_features_stream=features),
                get_input_embeddings=lambda:embed, _live_forward=forward, lm_head=head)
            result = env['run'](model, torch.zeros(1,128,chunk['rows']*8), state,
                input_ids=torch.tensor([chunk['prompt']]) if 'prompt' in chunk else None,
                is_final=chunk['final'],max_new_tokens=chunk['limit'],suppress_token_ids=[17,18],stop_token_ids=[18,19])
            state = result.state
            outputs.append(dict(**chunk, ids=result.sequences[0].tolist(),audio_tokens=result.num_audio_tokens,
                closed=result.closed,pending=None if state.pending is None else state.pending[0,:,0].tolist(),
                finished=state.finished,forward_inputs=inputs,context_tokens=state.attention_mask.shape[1]))
        cases.append(dict(name=name,chunks=outputs))
    Path('test/fixtures/audio/live_oracle.json').write_text(json.dumps(dict(source=URL,
        source_sha256=hashlib.sha256(source).hexdigest(),torch=torch.__version__,markers=markers,vocab_size=32,cases=cases),indent=2)+'\n')


if __name__ == '__main__':main()
