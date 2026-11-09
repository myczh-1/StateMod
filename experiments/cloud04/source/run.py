"""One-shot new-family/wording confirmation; load frozen Bank/readout, never compile or train."""
import argparse,gc,sys,time,traceback
import numpy as np
from common import *
from backend import Bank,call,clone,disjoint,clean,arrays,capture,checked_step
from gate import inspect_return
from readout import Judgment

def require_versions(cfg):
    import importlib.metadata
    for name,v in cfg['runtime_versions'].items():
        assert importlib.metadata.version(name)==v, 'STOP_ENGINEERING version '+name

def engineering(model,payload,layers,ids,path):
    path.mkdir(parents=True)
    z,a=forward(model,ids)
    bank=Bank(model,payload,layers,0.)
    zz,b=call(model,ids,None,bank)
    save(path/'zero.npz',no_hook=z,zero=zz)
    result=dict(logits_exact=np.array_equal(z,zz),cache_exact=cache_hash(a)==cache_hash(b),
        cache_disjoint=disjoint(a,b),hooks_clean=clean(model),zero_calls=len(bank.records))
    write(path/'zero.json',{k:bool(v) if isinstance(v,np.bool_) else v for k,v in result.items()})
    assert all(result[k] for k in ['logits_exact','cache_exact','cache_disjoint','hooks_clean']) and result['zero_calls']==0,'STOP_ENGINEERING Zero'

def probe(model,tok,ids,tokens,checkpoint,judge,path):
    path.mkdir(parents=True);before=cache_hash(checkpoint)
    z,c,h=capture(model,[tokens[-1]],clone(model,checkpoint))
    plain,pc=forward(model,[tokens[-1]],clone(model,checkpoint))
    fresh,fc,hf=capture(model,ids+tokens)
    save(path/'states.npz',hidden=h,hidden_fresh=hf,logits=z,plain=plain,fresh=fresh)
    text=tok.decode(ids+tokens,skip_special_tokens=False)
    boundary=all(tok.encode(text+s,add_special_tokens=False)[:len(ids+tokens)]==ids+tokens for s in [' set',' frozenset'])
    # The detector only sees h. Fresh-path agreement is checked after prediction; no expected labels are read.
    judgment=judge.predict(h);fresh_judgment=judge.predict(hf)
    result=dict(judgment=judgment,readonly_logits_exact=bool(np.array_equal(z,plain)),
        readonly_cache_exact=cache_hash(c)==cache_hash(pc),checkpoint_unchanged=cache_hash(checkpoint)==before,
        disjoint=disjoint(c,pc) and disjoint(c,checkpoint),clean=clean(model),
        logits=metrics(z,fresh),hidden=metrics(h,hf),
        p_error=abs(judgment['p']-fresh_judgment['p']),same_action=judgment['enabled']==fresh_judgment['enabled'],fresh_judgment=fresh_judgment,boundary=boundary)
    write(path/'audit.json',result)
    assert all(result[k] for k in ['readonly_logits_exact','readonly_cache_exact','checkpoint_unchanged','disjoint','clean']), 'STOP_ENGINEERING probe'
    assert max(result[k]['rms'] for k in ['logits','hidden'])<=.001 and result['p_error']<=.001 and result['same_action'],'STOP_NUMERICAL frozen readout'
    return judgment,boundary

def task(model,tok,t,manual_flag,judge,payload,cfg,out,eos):
    path=out/t['id'];path.mkdir();ids=tok.apply_chat_template(t['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    assert ids==t['input_ids'],'STOP_ENGINEERING prompt'
    tokens=[];logits=[];trace=[];cache=None;gate=None
    for j in range(cfg['max_new_tokens']):
        found=inspect_return(tok.decode(tokens,skip_special_tokens=True)) if tokens else None
        if found:
            gate=dict(**found,prediction_step=j,absolute_position=len(ids)+len(tokens)-1);break
        z,cache=forward(model,ids if j==0 else [tokens[-1]],cache)
        logits.append(z);tokens.append(int(z.argmax()))
        trace.append(dict(step=j,bank_calls=0,native_length=cache.get_seq_length()))
        if tokens[-1] in eos:break
    before=cache_hash(cache);onset=cache.get_seq_length()
    shared=path/'shared';shared.mkdir()
    save(shared/'cache.npz',**arrays(cache));save(shared/'logits.npz',logits=np.stack(logits))
    write(shared/'checkpoint.json',dict(input_ids=ids,generated_ids=tokens,gate=gate,cache_hash=before,onset=onset))
    judgment=None;boundary=False
    if gate:judgment,boundary=probe(model,tok,ids,tokens,cache,judge,path/'probe')
    decisions={'BASE':False,'MANUAL_V':bool(manual_flag and boundary),'MODULE_V':bool(boundary and judgment['enabled'])}
    write(path/'routing.json',dict(judgment=judgment,manual_is_label_based_upper_bound=True,
        module_input='numeric Block13 h only',decisions=decisions,coverage=bool(gate and boundary)))
    results=[];generated={};raw={};caches={}
    for name,enabled in decisions.items():
        p=path/name;p.mkdir();(p/'steps').mkdir();c=clone(model,cache)
        assert disjoint(c,cache) and cache_hash(c)==before
        gen=tokens.copy();zs=logits.copy();steps=trace.copy()
        bank=Bank(model,payload,cfg['layers'],1.) if enabled else None
        start=time.monotonic()
        for j in range(len(gen),cfg['max_new_tokens']):
            if gen[-1] in eos:break
            active=enabled and j==len(tokens)
            z,c,row=checked_step(model,[gen[-1]],c,ids+gen,bank,onset,p/'steps'/f'{j:03}',active)
            row.update(step=j,state='INJECTED_ONCE_THEN_CONSUMED' if active else 'CONSUMED' if enabled else 'OFF')
            zs.append(z);steps.append(row);gen.append(int(z.argmax()))
        stack=np.stack(zs);save(p/'logits.npz',logits=stack)
        result=dict(id=t['id'],condition=name,input_ids=ids,output_ids=gen,text=tok.decode(gen,skip_special_tokens=True),
            stop='EOS' if gen[-1] in eos else 'TRUNCATED',gate=gate,coverage=bool(gate and boundary),enabled=enabled,
            judgment=judgment if name=='MODULE_V' else None,bank_calls=sum(r['bank_calls'] for r in steps),
            active_prediction_steps=sum(r['bank_calls']>0 for r in steps),steps=steps,seconds=time.monotonic()-start,
            final_cache_sha=cache_hash(c),prompt_unchanged=True)
        write(p/'generation.json',result)
        assert result['bank_calls']==(len(cfg['layers']) if enabled else 0),'STOP_ENGINEERING single access'
        assert cache_hash(cache)==before and clean(model),'STOP_ENGINEERING branch pollution'
        equal_route='BASE' if not enabled and name!='BASE' else 'MANUAL_V' if name=='MODULE_V' and decisions['MANUAL_V'] else None
        if equal_route:
            same=dict(tokens=gen==generated[equal_route],logits=bool(np.array_equal(stack,raw[equal_route])),cache=result['final_cache_sha']==caches[equal_route])
            write(p/'identical_route.json',same);assert all(same.values()),'STOP_ENGINEERING same route'
        results.append(result);generated[name]=gen;raw[name]=stack;caches[name]=result['final_cache_sha']
        print(t['id'],name,'enabled',enabled,'calls',result['bank_calls'],'stop',result['stop'],flush=True)
        del c,bank;gc.collect()
    return results

def text_task(model,tok,t,cfg,out,eos):
    p=out/t['id']/'TEXT_CODE';p.mkdir();(p/'steps').mkdir()
    ids=tok.apply_chat_template(t['text_messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    assert ids==t['text_input_ids']
    tokens=[];zs=[];steps=[];cache=None;gate=None;start=time.monotonic()
    for j in range(cfg['max_new_tokens']):
        found=inspect_return(tok.decode(tokens,skip_special_tokens=True)) if tokens and gate is None else None
        if found:gate=dict(**found,prediction_step=j,absolute_position=len(ids)+j-1)
        z,cache,row=checked_step(model,ids if j==0 else [tokens[-1]],cache,ids+tokens,None,None,p/'steps'/f'{j:03}',False)
        row.update(step=j,state='TEXT_REFERENCE_NO_BANK');steps.append(row);zs.append(z);tokens.append(int(z.argmax()))
        if tokens[-1] in eos:break
    save(p/'logits.npz',logits=np.stack(zs))
    r=dict(id=t['id'],condition='TEXT_CODE',input_ids=ids,output_ids=tokens,text=tok.decode(tokens,skip_special_tokens=True),
        stop='EOS' if tokens[-1] in eos else 'TRUNCATED',gate=gate,coverage=gate is not None,enabled=False,
        bank_calls=0,active_prediction_steps=0,steps=steps,seconds=time.monotonic()-start,rule_text_visible=True)
    write(p/'generation.json',r);print(t['id'],'TEXT_CODE',r['stop'],flush=True)
    return r

def main(args):
    verify_package();cfg=read(ROOT/'data/config.json');require_versions(cfg)
    assert sha(ROOT/'data/judgment.npz')==cfg['judgment_sha'] and sha(ROOT/'data/bank.npz')==cfg['bank_sha']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.time()
    write(out/'START.json',dict(time=start,argv=sys.argv,source_freeze_sha=sha(ROOT/'freeze.json')))
    try:
        import torch
        from transformers import AutoTokenizer
        expected=verify_model(args.model);tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        model=load_model(args.model);versions=parameter_versions(model)
        assert model.config.hidden_size==4096 and model.config.num_hidden_layers==36
        assert model.config.num_attention_heads==32 and model.config.num_key_value_heads==8 and model.config.head_dim==128
        with np.load(ROOT/'data/bank.npz') as a:payload={k:a[k].copy() for k in a.files}
        assert payload['K'].shape==payload['V'].shape==(36,8,36,128)
        tasks=read(ROOT/'data/inputs.json');judge=Judgment(ROOT/'data/judgment.npz')
        engineering(model,payload,cfg['layers'],read(ROOT/'data/engineering_input.json')['input_ids'],out/'engineering')
        manual=read(ROOT/'data/manual_schedule.json');results=[]
        eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
        for t in tasks:
            results+=task(model,tok,t,manual[t['id']],judge,payload,cfg,out,eos)
            results.append(text_task(model,tok,t,cfg,out,eos))
        write(out/'RESULTS.json',results)
        check_parameters(model,versions);assert clean(model)
        write(out/'COST.json',dict(wall_seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),condition_output_tokens=sum(len(r['output_ids']) for r in results),
            condition_records=len(results),archived_text_reference_count=0,payload_bytes=(ROOT/'data/bank.npz').stat().st_size,
            active_side_slots_per_layer=36,active_layers=14,judgment_bytes=(ROOT/'data/judgment.npz').stat().st_size))
        del model;gc.collect();torch.cuda.empty_cache()
        assert verify_model(args.model)==expected;verify_package()
        write(out/'DECISION.json',dict(status='ENGINEERING_COMPLETE_PENDING_EXECUTION_SCORING',model_sha_unchanged=True,
            judgment_sha_unchanged=sha(ROOT/'data/judgment.npz')==cfg['judgment_sha'],bank_sha_unchanged=sha(ROOT/'data/bank.npz')==cfg['bank_sha'],
            bank_compiled=False,judge_trained=False,planned_conditions=128,completed_conditions=len(results),
            covered=sum(r['coverage'] for r in results if r['condition']=='MODULE_V')))
    except BaseException as ex:
        write(out/'FAILURE.json',dict(error=repr(ex),traceback=traceback.format_exc(),
            status='STOP_NUMERICAL' if 'STOP_NUMERICAL' in str(ex) else 'STOP_ENGINEERING'))
        raise
    finally:seal_output(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);main(p.parse_args())
