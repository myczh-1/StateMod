"""Qwen3 static side-KV, adapted from L22; no runtime task-aware policy here."""
import contextlib, hashlib, math, types
import numpy as np
import torch
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
from .runtime_helpers import forward, save, write, metrics, logp, cache_hash

def clone(model, cache):
    result = DynamicCache(config=model.config)
    for i, layer in enumerate(cache.layers):
        result.update(layer.keys.clone(), layer.values.clone(), i)
    return result

def disjoint(a, b):
    return all(getattr(x,k).data_ptr() != getattr(y,k).data_ptr()
               for x,y in zip(a.layers,b.layers) for k in ('keys','values'))

def clean(model):
    return not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()) and all(
        'forward' not in b.self_attn.__dict__ for b in model.model.layers)

def arrays(cache):
    return {f'{i}_{k}': getattr(l,k).detach().cpu().numpy().copy()
            for i,l in enumerate(cache.layers) for k in ('keys','values')}

def capture(model, ids, cache=None):
    states=[]
    def hook(m,args,out):
        states.append(out[0,-1].detach().cpu().numpy().copy())
    handle=model.model.layers[13].register_forward_hook(hook)
    try: z,c=forward(model,ids,cache)
    finally: handle.remove()
    assert len(states)==1
    return z,c,states[0]

class Bank:
    def __init__(self, model, payload, layers, weight=1.):
        self.model=model; self.layers=tuple(layers); self.weight=weight
        device=next(model.parameters()).device
        self.K=torch.tensor(payload['K'].copy(),device=device)
        self.V=torch.tensor(payload['V'].copy(),device=device)
        cfg=model.config
        assert self.K.shape==self.V.shape==(cfg.num_hidden_layers,cfg.num_key_value_heads,36,cfg.head_dim)
        assert torch.isfinite(self.K).all() and torch.isfinite(self.V).all()
        self.records=[];self.raw={}

    @contextlib.contextmanager
    def context(self):
        handles=[];self.records=[];self.raw={}
        try:
            for i in self.layers:
                m=self.model.model.layers[i].self_attn
                assert 'forward' not in m.__dict__
                original=m.forward; owner=self
                def attention(module,hidden_states,position_embeddings,attention_mask,
                              past_key_values=None,cache_position=None,_old=original,_i=i,**kw):
                    # Zero bypasses the full side path, including its Key competition.
                    if owner.weight==0:
                        return _old(hidden_states=hidden_states,position_embeddings=position_embeddings,
                                    attention_mask=attention_mask,past_key_values=past_key_values,
                                    cache_position=cache_position,**kw)
                    shape=hidden_states.shape[:-1]; view=(*shape,-1,module.head_dim)
                    assert shape==(1,1), 'STOP_ENGINEERING not single prediction'
                    q0=module.q_norm(module.q_proj(hidden_states).view(view)).transpose(1,2)
                    k0=module.k_norm(module.k_proj(hidden_states).view(view)).transpose(1,2)
                    v=module.v_proj(hidden_states).view(view).transpose(1,2)
                    cos,sin=position_embeddings; q,k=apply_rotary_pos_emb(q0,k0,cos,sin)
                    before=past_key_values.layers[_i].get_seq_length()
                    assert cache_position.tolist()==[before]
                    k,v=past_key_values.update(k,v,module.layer_idx,{'sin':sin,'cos':cos,'cache_position':cache_position})
                    assert k.shape[-2]==before+1
                    groups=module.num_key_value_groups
                    nk,nv=repeat_kv(k,groups),repeat_kv(v,groups)
                    bk,bv=repeat_kv(owner.K[_i][None],groups),repeat_kv(owner.V[_i][None],groups)
                    assert attention_mask.shape==(1,1,1,k.shape[-2])
                    native=q@nk.transpose(2,3)*module.scaling+attention_mask
                    side=q0@bk.transpose(2,3)*module.scaling+math.log(owner.weight)
                    probs=torch.softmax(torch.cat([native,side],-1),dim=-1,dtype=torch.float32).to(q.dtype)
                    value=torch.cat([nv,bv],2)
                    result=(probs@value).transpose(1,2).contiguous().reshape(*shape,-1)
                    raw=dict(q_pre=q0,q_rotated=q,native_key=k,native_value=v,mask=attention_mask,
                             position_ids=cache_position,cos=cos,sin=sin,probabilities=probs,attention_output=result)
                    for key,a in raw.items(): owner.raw[f'L{_i}_{key}']=a.detach().cpu().numpy().copy()
                    owner.records.append(dict(layer=_i,positions=cache_position.tolist(),native_before=before,
                        native_after=k.shape[-2],kv_heads=k.shape[1],query_heads=q.shape[1],slots=36,
                        weight=owner.weight,bank_mass=probs[...,k.shape[-2]:].sum(-1).detach().cpu().tolist()))
                    return module.o_proj(result),probs
                m.forward=types.MethodType(attention,m);handles.append(m)
            yield
        finally:
            for m in handles: del m.forward

    def check_attention(self, path):
        # Persist raw evidence before independent float64 arithmetic and thresholds.
        save(path.with_suffix('.npz'),**self.raw)
        checks=[]
        for r in self.records:
            i=r['layer'];get=lambda k:self.raw[f'L{i}_{k}'].astype(np.float64)
            groups=r['query_heads']//r['kv_heads']
            k=np.repeat(get('native_key'),groups,axis=1);v=np.repeat(get('native_value'),groups,axis=1)
            bk=np.repeat(self.K[i].cpu().numpy()[None],groups,axis=1).astype(np.float64)
            bv=np.repeat(self.V[i].cpu().numpy()[None],groups,axis=1).astype(np.float64)
            z=np.concatenate([get('q_rotated')@k.swapaxes(-1,-2)/math.sqrt(k.shape[-1])+get('mask'),
                 get('q_pre')@bk.swapaxes(-1,-2)/math.sqrt(k.shape[-1])+math.log(self.weight)],axis=-1)
            z-=z.max(-1,keepdims=True);p=np.exp(z);p/=p.sum(-1,keepdims=True)
            y=(p@np.concatenate([v,bv],2)).transpose(0,2,1,3).reshape(1,1,-1)
            checks.append(dict(layer=i,output=metrics(y,get('attention_output')),
                probability_max_error=float(abs(p-get('probabilities')).max())))
        write(path.with_suffix('.json'),dict(records=self.records,checks=checks))
        assert [r['layer'] for r in self.records]==list(self.layers),'STOP_ENGINEERING layer scope'
        assert all(r['output']['rms']<=.001 and r['probability_max_error']<=1e-5 for r in checks),'STOP_NUMERICAL Attention'

def call(model, seq, cache=None, bank=None):
    if bank is None: return forward(model,seq,cache)
    with bank.context(): return forward(model,seq,cache)

def replay(model, history, bank, onset):
    if bank is None or onset is None or len(history)<=onset:
        return forward(model,history)
    _,c=forward(model,history[:onset])
    z,c=call(model,history[onset:onset+1],c,bank)
    if len(history)>onset+1: z,c=forward(model,history[onset+1:],c)
    return z,c

def checked_step(model, seq, cache, history, bank, onset, path, active):
    z,c=call(model,seq,cache,bank if active else None)
    trace=list(bank.records) if active else []
    if active: bank.check_attention(path.with_name(path.name+'_attention'))
    fresh,fc=replay(model,history,bank,onset)
    save(path.with_suffix('.npz'),cached=z,fresh=fresh)
    row=dict(numerical=metrics(z,fresh),selected_logp_error=abs(logp(z,int(z.argmax()))-logp(fresh,int(z.argmax()))),
             token=int(z.argmax()),fresh_token=int(fresh.argmax()),bank_calls=len(trace),trace=trace,
             native_length=c.get_seq_length(),fresh_length=fc.get_seq_length())
    write(path.with_suffix('.json'),row)
    if row['numerical']['rms']>.001 or row['selected_logp_error']>.01:
        save(path.with_name(path.name+'_failed_cached.npz'),**arrays(c));save(path.with_name(path.name+'_failed_fresh.npz'),**arrays(fc))
        raise AssertionError('STOP_NUMERICAL cached/fresh')
    assert row['native_length']==row['fresh_length']==len(history) and clean(model),'STOP_ENGINEERING history/hooks'
    assert len(trace)==(len(bank.layers) if active else 0),'STOP_ENGINEERING access count'
    return z,c,row
