#!/usr/bin/env python3
"""Pinned upstream Depthformer/LiquidCodec oracle; synthetic weights by default.

--gguf uses only the existing local official Q4 mmproj, dequantized with gguf-py.
No checkpoint weights are written into committed fixtures.
"""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import importlib
import json
import math
import sys
import tempfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers.utils import torch_compilable_check

REV = '0295e4130594c06800fd25451d97139ac2da1aff'
BASE = f'https://huggingface.co/LiquidAI/LFM3-Audio-2B/resolve/{REV}/'


def extract(source, classes, scope, extra=()):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in classes]
    for cls, method, rename in extra:
        parent = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls)
        node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == method)
        node.name = rename
        node.decorator_list = []
        nodes.append(node)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes, type_ignores=[])), '<upstream>', 'exec'), scope)


def layer_name(name, prefix):
    if name == 'embedding_norm.weight': return prefix+'output_norm.weight'
    parts = name.split('.')
    assert parts[0] == 'layers', name
    rest = '.'.join(parts[2:])
    mapping = {'operator_norm':'ln1', 'post_operator_norm':'attn_post_norm',
        'ffn_norm':'ln2', 'post_ffn_norm':'ffn_post_norm',
        'self_attn.q_proj':'attn_q', 'self_attn.k_proj':'attn_k', 'self_attn.v_proj':'attn_v',
        'self_attn.o_proj':'attn_out', 'self_attn.gate_proj':'attn_gate',
        'self_attn.q_norm':'attn_q_norm', 'self_attn.k_norm':'attn_k_norm',
        'conv.in_proj':'conv_pw1', 'conv.out_proj':'conv_pw2', 'conv.conv':'conv_dw',
        'feed_forward.w1':'ffn_gate', 'feed_forward.w3':'ffn_up', 'feed_forward.w2':'ffn_down'}
    for old,new in mapping.items():
        if rest.startswith(old+'.'): return f'{prefix}blk.{parts[1]}.{new}{rest[len(old):]}'
    raise ValueError(name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf',type=Path)
    p.add_argument('--gguf-python',type=Path)
    p.add_argument('--out',type=Path,default=Path('test/fixtures/audio/generation_oracle.json'))
    args=p.parse_args()
    torch.set_num_threads(2)
    sources={n:urllib.request.urlopen(BASE+n).read() for n in ['modeling_lfm3.py','configuration_lfm3.py',
        'lfm3_conv.py','modeling_lfm3_audio.py','modeling_liquid_codec.py']}
    with tempfile.TemporaryDirectory() as temp:
        package=Path(temp)/'tts_reference';package.mkdir();(package/'__init__.py').write_text('')
        for name in ['configuration_lfm3.py','modeling_lfm3.py','lfm3_conv.py']: (package/name).write_bytes(sources[name])
        sys.path.insert(0,temp)
        lm=importlib.import_module('tts_reference.modeling_lfm3')
        Config=lm.Lfm3Config
        scope=dict(torch=torch,nn=nn,F=F,math=math,Lfm3Model=lm.Lfm3Model,
            DynamicCache=DynamicCache,torch_compilable_check=torch_compilable_check,
            LiquidCodecDecoderOutput=lambda **kw:SimpleNamespace(**kw))
        extract(sources['modeling_lfm3_audio.py'], ['Lfm3AudioRMSNorm','Lfm3AudioCodebookEmbedding','Lfm3AudioDepthDecoder'],scope)
        extract(sources['modeling_liquid_codec.py'], ['LiquidCodecLfm','LiquidCodecQuantizer','LiquidCodecISTFT',
            'LiquidCodecDecoder','LiquidCodecStreamingCache'],scope, [('LiquidCodecModel','decode','codec_decode')])
        reader=None
        if args.gguf:
            sys.path.insert(0,str(args.gguf_python));import gguf
            reader=gguf.GGUFReader(str(args.gguf));tensors={t.name:t for t in reader.tensors}
            dh=1024;hidden=2048;vocab=2017;heads=32;kv=8;df=2816;ch=512;cd=64;cf=2304;qdim=256;fft=1280
            levels=[8,7,6,6]
        else:
            dh=32;hidden=32;vocab=33;heads=2;kv=1;df=64;ch=32;cd=16;cf=64;qdim=32;fft=80
            levels=[2,2,2,4]
        depth_config=Config(hidden_size=dh,intermediate_size=df,num_hidden_layers=3,
            num_attention_heads=heads,num_key_value_heads=kv,head_dim=dh//heads,
            layer_types=['full_attention']*3,use_sandwich_norm=False,conv_L_cache=7,
            rope_parameters={'full_attention':{'rope_type':'default','rope_theta':1000000.0}},
            vocab_size=1,tie_word_embeddings=True,pad_token_id=None,bos_token_id=None,eos_token_id=None)
        depth_config._attn_implementation='eager'
        depth=scope['Lfm3AudioDepthDecoder'](SimpleNamespace(depth_config=depth_config,
            num_codebooks=12,codebook_size=vocab,text_config=SimpleNamespace(hidden_size=hidden))).eval()
        def codec_lfm(window):
            c=Config(hidden_size=ch,intermediate_size=cf,num_hidden_layers=8,num_attention_heads=ch//cd,
                num_key_value_heads=(ch//cd)//4 if reader else 1,head_dim=cd,
                layer_types=['sliding_attention' if i in (2,4,6) else 'conv' for i in range(8)],
                use_sandwich_norm=True,conv_L_cache=7,sliding_window=window,
                rope_parameters={'sliding_attention':{'rope_type':'default','rope_theta':1000000.0}},
                vocab_size=1,tie_word_embeddings=False,pad_token_id=None,bos_token_id=None,eos_token_id=None)
            c._attn_implementation='eager';return c
        config=SimpleNamespace(n_q=12,fsq_levels=levels,quantizer_dim=qdim,codebook_size=vocab-1,
            encoder_dim=ch,decoder_window=30,stft_n_fft=fft,stft_win_length=fft,
            stft_hop_length=fft//4,stft_frame_stack=6,lfm_config=codec_lfm)
        codec=SimpleNamespace(config=config,quantizer=scope['LiquidCodecQuantizer'](config),
            decoder=scope['LiquidCodecDecoder'](config).eval(),from_quantizer=nn.Linear(qdim,ch))
        feedback=nn.Embedding(12*vocab,hidden)
        store={}
        def data(name,shape,vector=False):
            if name in store:return store[name]['values']
            if reader:
                t=tensors[name];values=gguf.dequantize(t.data,t.tensor_type).reshape(shape)
                store[name]=dict(values=values);return values
            count=math.prod(shape);seed=sum((i+1)*ord(c) for i,c in enumerate(name))%997
            if vector:
                norm=('norm' in name or '.ln' in name) and name.endswith('weight')
                values=np.array([(1 if norm else 0)+((seed+i*7+i//3*11)%37-18)/(256 if norm else 1024)
                    for i in range(count)],dtype=np.float32).reshape(shape)
                raw=values.astype('<f4').tobytes();kind='f32'
            else:
                raw=bytearray();flat=[]
                for b in range(count//32):
                    scale=(b%3+1)/256;raw+=np.float16(scale).tobytes()
                    packed=[(seed+b*11+i*37)%256 for i in range(16)];raw+=bytes(packed)
                    flat.extend([scale*((x&15)-8) for x in packed]+[scale*((x>>4)-8) for x in packed])
                values=np.array(flat,dtype=np.float32).reshape(shape);kind='q4_0'
            store[name]=dict(values=values,shape=shape,type=kind,bytes=base64.b64encode(raw).decode())
            return values
        def assign(param,name,shape=None,vector=False,index=None):
            value=data(name,shape or list(param.shape),vector)
            if index is not None:value=value[index]
            param.data.copy_(torch.from_numpy(np.array(value,copy=True)).reshape(param.shape))
        for name,param in depth.named_parameters():
            if name=='mask_embedding':g='a.gen.code.mask_embd';shape=[dh];idx=None
            elif name.startswith('depth_linear.'):g='a.gen.code.proj_in.'+name.split('.')[-1];shape=None;idx=None
            elif name.startswith('model.'):g=layer_name(name[6:],'a.gen.code.');shape=None;idx=None
            else:
                parts=name.split('.');idx=int(parts[1]);g='a.gen.code.embd.weight' if parts[2]=='embedding' else 'a.gen.code.head_norm.weight'
                shape=[12,vocab,dh] if parts[2]=='embedding' else [12,dh]
            assign(param,g,shape,vector=('norm' in g or '.ln' in g or g.endswith('bias') or 'mask_embd' in g),index=idx)
        assign(feedback.weight,'a.gen.code.out_embd.weight')
        for name,param in codec.quantizer.named_parameters():
            if name.startswith('project_out.'):assign(param,'a.gen.wav.quant.first.out_proj.'+name.split('.')[-1],vector=True)
        for name,param in codec.from_quantizer.named_parameters():assign(param,'a.gen.wav.tfm.in_proj.'+name,vector=name.endswith('bias'))
        for name,param in codec.decoder.named_parameters():
            if name.startswith('lfm.'):
                g=layer_name(name[4:],'a.gen.wav.tfm.')
                shape=[ch,7] if 'conv_dw' in g else None
            else:g='a.gen.wav.tfm.out_proj.'+name.split('.')[-1];shape=None
            assign(param,g,shape,vector=('norm' in g or '.ln' in g or 'conv_dw' in g or g.endswith('bias')))
        table=np.array([[(code//math.prod(levels[:j])%levels[j]-levels[j]//2)/(levels[j]//2)
            for j in range(4)] for code in range(vocab-1)],dtype=np.float32)
        table_name='a.gen.wav.quant.first.codebook.weight'
        if reader:np.testing.assert_equal(data(table_name,list(table.shape),True),table)
        else:store[table_name]=dict(values=table,shape=list(table.shape),type='f32',bytes=base64.b64encode(table.astype('<f4').tobytes()).decode())
        depth_cases=[];codec_cases=[]
        with torch.no_grad():
            for partial in [False,True]:
                hidden_state=torch.tensor([((i*17+i//5)%127-63)/64 for i in range(hidden)],dtype=torch.float32)
                codes=torch.tensor([(i*13+17)%(vocab-1) if partial and i%3==0 else 0 for i in range(12)])
                masked=torch.tensor([not(partial and i%3==0) for i in range(12)])
                captures={}
                handles=[depth.depth_linear.register_forward_hook(lambda m,a,o:None)]
                # Record the exact input consumed by the upstream LFM3 trunk.
                handles.append(depth.model.register_forward_pre_hook(lambda m,a,kw:captures.update(input=kw['inputs_embeds'].flatten().tolist()),with_kwargs=True))
                for i,layer in enumerate(depth.model.layers):handles.append(layer.register_forward_hook(lambda m,a,o,i=i:captures.update({f'layer.{i}':o.flatten().tolist()})))
                handles.append(depth.model.register_forward_hook(lambda m,a,o:captures.update(embedding=o.last_hidden_state.flatten().tolist())))
                logits=depth(hidden_state[None],codes[None],masked[None])
                captures['logits']=logits.flatten().tolist()
                for h in handles:h.remove()
                actual_codes=[(i*13+17)%(vocab-1) for i in range(12)]
                fb=feedback(torch.tensor(actual_codes)+torch.arange(12)*vocab).sum(0)
                depth_cases.append(dict(hidden=hidden_state.tolist(),codes=[-1 if masked[i] else codes[i].item() for i in range(12)],stages=captures,feedback_codes=actual_codes,feedback=fb.tolist()))
            for frames in [1,2,6]:
                codes=torch.tensor([[(t*31+cb*13+17)%(vocab-1) for t in range(frames)] for cb in range(12)])[None]
                full=scope['codec_decode'](codec,codes).audio_values.flatten().tolist()
                cache=scope['LiquidCodecStreamingCache'](config);chunks=[]
                for t in range(frames):
                    samples=scope['codec_decode'](codec,codes[...,t:t+1],streaming_cache=cache,finalize=False).audio_values
                    chunks.append(samples.flatten().tolist())
                tail=scope['codec_decode'](codec,codes[...,:0],streaming_cache=cache,finalize=True).audio_values.flatten().tolist()
                torch.testing.assert_close(torch.tensor(sum(chunks,[])+tail),torch.tensor(full),atol=2e-4,rtol=2e-4)
                codec_cases.append(dict(codes=codes.transpose(1,2).flatten().tolist(),samples=full,chunks=chunks,tail=tail))
        result=dict(sources={n:dict(url=BASE+n,sha256=hashlib.sha256(s).hexdigest()) for n,s in sources.items()},torch=torch.__version__,
            metadata={'general.architecture':'clip','clip.gen.audio.projector_type':'lfm3a_gen',
            'clip.gen.audio.projection_dim':hidden,'clip.gen.audio.embedding_length':dh,'clip.gen.audio.feed_forward_length':df,
            'clip.gen.audio.attention.head_count':heads,'clip.gen.audio.attention.head_count_kv':kv,'clip.gen.audio.block_count':3},
            tensors=[] if reader else [dict(name=n,**{k:v for k,v in d.items() if k!='values'}) for n,d in store.items()],
            depth=depth_cases,codec=codec_cases)
        if reader:result['gguf_sha256']=hashlib.file_digest(args.gguf.open('rb'),'sha256').hexdigest()
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
