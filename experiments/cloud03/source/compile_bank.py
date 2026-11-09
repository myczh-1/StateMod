"""Compile original static rule with frozen 8B weights, never target-task tokens."""
import time
import numpy as np
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from common import forward,save,write,sha
from backend import clean

def compile_bank(model,tokenizer,rule,path,layers):
    ids=tokenizer.apply_chat_template([{'role':'system','content':rule['text']}],tokenize=True,
                                      add_generation_prompt=False,enable_thinking=False)
    assert ids==rule['ids'],'STOP_ENGINEERING compile tokenization'
    lo,hi=rule['body_range'];assert hi-lo==36
    assert tokenizer.encode(rule['text'],add_special_tokens=False)==ids[lo:hi]
    captures={};handles=[]
    for i,b in enumerate(model.model.layers):
        def hook(m,args,kw,i=i):
            x=kw.get('hidden_states',args[0] if args else None);view=(*x.shape[:-1],-1,m.head_dim)
            with torch.inference_mode():
                k=m.k_norm(m.k_proj(x).view(view)).transpose(1,2)
                v=m.v_proj(x).view(view).transpose(1,2)
            captures[i]=(k.detach().cpu().numpy().copy(),v.detach().cpu().numpy().copy())
        handles.append(b.self_attn.register_forward_pre_hook(hook,with_kwargs=True))
    try:z,cache=forward(model,ids)
    finally:
        for h in handles:h.remove()
    cfg=model.config
    a={name:np.stack([captures[i][j][0,:,lo:hi] for i in range(cfg.num_hidden_layers)])
       for j,name in enumerate(['K','V'])}
    path.mkdir(parents=True,exist_ok=False)
    save(path/'bank.npz',**a)
    raw={f'L{i}_{kind}':captures[i][j] for i in captures for j,kind in enumerate(['K_pre','V'])}
    raw.update({f'L{i}_K_post':l.keys.cpu().numpy().copy() for i,l in enumerate(cache.layers)})
    save(path/'compilation_raw.npz',logits=z,**raw)
    device=next(model.parameters()).device
    with torch.inference_mode():
        positions=torch.arange(len(ids),device=device)[None]
        cos,sin=model.model.rotary_emb(torch.zeros((1,len(ids),cfg.hidden_size),device=device),positions)
        checks=[]
        for i in captures:
            k=torch.tensor(captures[i][0],device=device)
            _,rot=apply_rotary_pos_emb(k,k,cos,sin)
            checks.append(dict(layer=i,rope_max=float((rot-cache.layers[i].keys).abs().max().item()),
                               value_exact=bool(np.array_equal(captures[i][1],cache.layers[i].values.cpu().numpy()))))
    metadata=dict(state_id='r2_static_qwen3_8b_cloud03',version='1.0.0',source_task_ids=[],rule=rule,
        model_revision='b968826d9c46dd6066d109eabc6255188de91218',shape=list(a['K'].shape),
        active_layers=layers,weight=1,window='FIRST_DECISION_ONLY',native_extra_positions=0,
        active_side_slots_per_layer=36,coordinate='post-k_norm pre-RoPE K; unrotated V; virtual relative phase zero',
        normalization='none; raw rule-source K/V',sha256=sha(path/'bank.npz'),checks=checks)
    write(path/'bank.json',metadata)
    assert clean(model),'STOP_ENGINEERING compile hooks'
    assert all(c['rope_max']<=3e-5 and c['value_exact'] for c in checks),'STOP_NUMERICAL compiler'
    assert a['K'].shape==a['V'].shape==(cfg.num_hidden_layers,cfg.num_key_value_heads,36,cfg.head_dim)
    assert all(np.isfinite(v).all() for v in a.values())
    write(path/'PAYLOAD_FREEZE.json',dict(time=time.time(),before_behavior=True,files={n:sha(path/n) for n in ['bank.npz','bank.json','compilation_raw.npz']}))
    return a
