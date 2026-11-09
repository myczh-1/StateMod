"""Three bounded Cloud07 diagnostics; immutable Cloud06 numerical/Bank/readout backend."""
import argparse, gc, sys, time, traceback
import numpy as np
from common import *
from backend import Bank, clone, disjoint, clean, arrays, checked_step
from legacy_run import engineering, probe, require_versions
from readout import Judgment
from gate import inspect_return

def selected_token(j, start, forced, z):
    offset=j-start
    forced_now=0<=offset<len(forced)
    return (int(forced[offset]) if forced_now else int(z.argmax())), forced_now

def capture_gate(model,tok,ids,tokens,cache,judge,path,gate):
    shared=path/'shared';shared.mkdir()
    save(shared/'cache.npz',**arrays(cache))
    write(shared/'checkpoint.json',dict(input_ids=ids,generated_ids=tokens,gate=gate,cache_hash=cache_hash(cache)))
    return probe(model,tok,ids,tokens,cache,judge,path/'probe')

def natural(model,tok,t,cond,judge,cfg,out,eos):
    p=out/t['id']/cond;p.mkdir(parents=True);(p/'steps').mkdir()
    kind='text' if cond=='TEXT_CODE' else 'base'
    ids=tok.apply_chat_template(t[kind+'_messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    assert ids==t[kind+'_ids'],'STOP_ENGINEERING input tokens'
    tokens=[];zs=[];steps=[];cache=None;gate=None;judgment=None;boundary=None;start=time.monotonic()
    for j in range(cfg['max_new_tokens']):
        found=inspect_return(tok.decode(tokens,skip_special_tokens=True)) if tokens and gate is None else None
        if found:
            gate=dict(**found,prediction_step=j,absolute_position=len(ids)+j-1)
            if cond=='BASE':judgment,boundary=capture_gate(model,tok,ids,tokens,cache,judge,p,gate)
        z,cache,row=checked_step(model,ids if j==0 else [tokens[-1]],cache,ids+tokens,None,None,p/'steps'/f'{j:03}',False)
        row.update(step=j,state='NO_BANK');steps.append(row);zs.append(z);tokens.append(int(z.argmax()))
        if tokens[-1] in eos:break
    save(p/'logits.npz',logits=np.stack(zs))
    r=dict(id=t['id'],condition=cond,input_ids=ids,output_ids=tokens,text=tok.decode(tokens,skip_special_tokens=True),stop='EOS' if tokens[-1] in eos else 'TRUNCATED',gate=gate,coverage=gate is not None,judgment=judgment,boundary=boundary,bank_calls=0,active_prediction_steps=0,steps=steps,seconds=time.monotonic()-start,final_cache_sha=cache_hash(cache))
    write(p/'generation.json',r);print('AB',t['id'],cond,r['stop'],flush=True)
    return r

def controlled_probe(model,tok,t,item,judge,out):
    p=out/'controlled_readouts'/t['id'];p.mkdir(parents=True)
    ids=t['base_ids'];tokens=item['prefix_ids']
    assert tok.decode(tokens)==item['prefix_text'] and inspect_return(tok.decode(tokens,skip_special_tokens=True))
    # Prefill then sequential teacher forcing, holding the actual saved prefix fixed.
    z,cache=forward(model,ids)
    for token in tokens[:-1]:z,cache=forward(model,[token],cache)
    j,b=probe(model,tok,ids,tokens,cache,judge,p/'probe')
    write(p/'READOUT.json',dict(task_id=t['id'],input_ids=ids,prefix_ids=tokens,judgment=j,boundary=b,bank=False,natural_generation=False))
    print('READOUT',t['id'],j['p'],flush=True)

def c_checkpoint(model,tok,c,out):
    ids=c['input_ids'];tokens=c['prefix_ids'];path=out/'C/shared';path.mkdir(parents=True)
    zs=[];cache=None
    for j,expected in enumerate(tokens):
        z,cache=forward(model,ids if j==0 else [tokens[j-1]],cache)
        zs.append(z)
        if int(z.argmax())!=expected:
            save(path/'prefix_mismatch.npz',logits=z)
            write(path/'REFERENCE_NOT_REPRODUCED.json',dict(step=j,expected=expected,actual=int(z.argmax())))
            return None
    assert tok.decode(tokens)==c['prefix_text'] and inspect_return(tok.decode(tokens,skip_special_tokens=True))
    save(path/'cache.npz',**arrays(cache));save(path/'logits.npz',logits=np.stack(zs))
    write(path/'checkpoint.json',dict(input_ids=ids,prefix_ids=tokens,cache_hash=cache_hash(cache),onset=cache.get_seq_length()))
    return cache,zs

def c_branch(model,tok,c,name,checkpoint,prefix_logits,payload,cfg,out,eos):
    p=out/'C'/name;p.mkdir();(p/'steps').mkdir()
    ids=c['input_ids'];prefix=c['prefix_ids'];gen=prefix.copy();zs=prefix_logits.copy()
    enabled=name in ['NATURAL_ON','ON_SET','ON_FRO']
    forced=[] if name.startswith('NATURAL') else c['constructors']['frozenset' if name.endswith('FRO') else 'set']['ids']
    bank=Bank(model,payload,cfg['layers'],1.) if enabled else None
    cache=clone(model,checkpoint);before=cache_hash(checkpoint);onset=checkpoint.get_seq_length()
    assert disjoint(cache,checkpoint) and cache_hash(cache)==before
    steps=[dict(step=j,bank_calls=0,forced=False,state='ORIGINAL_PREFIX') for j in range(len(prefix))]
    begin=time.monotonic()
    for j in range(len(prefix),cfg['max_new_tokens']):
        active=enabled and j==len(prefix)
        z,cache,row=checked_step(model,[gen[-1]],cache,ids+gen,bank,onset,p/'steps'/f'{j:03}',active)
        token,was_forced=selected_token(j,len(prefix),forced,z)
        with np.load(p/'steps'/f'{j:03}.npz') as a:fresh=a['fresh']
        forced_error=abs(logp(z,token)-logp(fresh,token))
        row.update(step=j,actual_token=token,forced=was_forced,actual_token_logp_error=forced_error,state='INJECTED_ONCE' if active else 'BANK_OFF')
        write(p/'steps'/f'{j:03}_selection.json',row)
        assert forced_error<=.01,'STOP_NUMERICAL forced selected token'
        if forced and j==len(prefix)+len(forced):
            save(p/'constructor_boundary_cache.npz',**arrays(cache));save(p/'constructor_boundary_logits.npz',logits=z)
            write(p/'constructor_boundary.json',dict(history_ids=ids+gen,cache_hash=cache_hash(cache),prediction_step=j,bank_calls_this_step=row['bank_calls']))
        zs.append(z);steps.append(row);gen.append(token)
        if token in eos:break
    save(p/'logits.npz',logits=np.stack(zs))
    r=dict(id=c['task']['id'],condition=name,input_ids=ids,output_ids=gen,text=tok.decode(gen,skip_special_tokens=True),stop='EOS' if gen[-1] in eos else 'TRUNCATED',forced_constructor=bool(forced),forced_ids=forced,natural_behavior_endpoint=not forced,bank_calls=sum(s['bank_calls'] for s in steps),active_prediction_steps=sum(s['bank_calls']>0 for s in steps),steps=steps,seconds=time.monotonic()-begin,final_cache_sha=cache_hash(cache),gate_step=len(prefix))
    write(p/'generation.json',r)
    assert r['bank_calls']==(14 if enabled else 0) and r['active_prediction_steps']==int(enabled),'STOP_ENGINEERING once'
    assert cache_hash(checkpoint)==before and clean(model),'STOP_ENGINEERING checkpoint/hook'
    print('C',name,r['stop'],r['bank_calls'],flush=True)
    return r

def c_diagnostic(model,tok,c,payload,cfg,out,eos,refs):
    prepared=c_checkpoint(model,tok,c,out)
    if prepared is None:return [],'REFERENCE_NOT_REPRODUCED'
    cache,zs=prepared;results=[]
    for name in ['NATURAL_OFF','NATURAL_ON']:
        results.append(c_branch(model,tok,c,name,cache,zs,payload,cfg,out,eos))
    refcheck={name:results[i]['output_ids']==refs[old]['output_ids'] for i,(name,old) in enumerate([('NATURAL_OFF','C_BASE'),('NATURAL_ON','C_MANUAL_V')])}
    write(out/'C/reference_check.json',refcheck)
    if not all(refcheck.values()):return results,'REFERENCE_NOT_REPRODUCED'
    for name in ['OFF_SET','OFF_FRO','ON_SET','ON_FRO']:
        results.append(c_branch(model,tok,c,name,cache,zs,payload,cfg,out,eos))
    by={r['condition']:r for r in results}
    checks={}
    for a,b in [('NATURAL_OFF','OFF_SET'),('NATURAL_ON','ON_FRO')]:
        with np.load(out/'C'/a/'logits.npz') as x,np.load(out/'C'/b/'logits.npz') as y:
            checks[a+'_'+b]=dict(tokens=by[a]['output_ids']==by[b]['output_ids'],logits_exact=bool(np.array_equal(x['logits'],y['logits'])),cache=by[a]['final_cache_sha']==by[b]['final_cache_sha'])
    write(out/'C/forced_replay_check.json',checks)
    assert all(all(v.values()) for v in checks.values()),'STOP_ENGINEERING forced replay'
    return results,'C_MATRIX_COMPLETE'

def main(args):
    verify_package();cfg=read(ROOT/'data/config.json');require_versions(cfg)
    out=Path(args.output);out.mkdir(exist_ok=False);started=time.time()
    write(out/'START.json',dict(time=started,argv=sys.argv,freeze_sha=sha(ROOT/'freeze.json')))
    try:
        import torch
        from transformers import AutoTokenizer
        expected=verify_model(args.model);tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        model=load_model(args.model);versions=parameter_versions(model)
        with np.load(ROOT/'data/bank.npz') as a:payload={k:a[k].copy() for k in a.files}
        judge=Judgment(ROOT/'data/judgment.npz');matrix=read(ROOT/'MATRIX.json');tasks={t['id']:t for t in read(ROOT/'TASKS.json')}
        refs=read(ROOT/'data/references.json');eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
        engineering(model,payload,cfg['layers'],read(ROOT/'data/engineering_input.json')['input_ids'],out/'engineering')
        results=[];seen=set();reference_checks={}
        for item in matrix['natural_ab']:
            key=(item['task_id'],item['condition'])
            if key in seen:assert item['reuse'];continue
            seen.add(key);r=natural(model,tok,tasks[key[0]],key[1],judge,cfg,out,eos);results.append(r)
            if key[1]=='BASE' and key[0] in refs:reference_checks[key[0]]=r['output_ids']==refs[key[0]]['output_ids']
        write(out/'A_REFERENCE.json',reference_checks)
        for item in matrix['controlled_readouts']:controlled_probe(model,tok,tasks[item['task_id']],item,judge,out)
        c=matrix['c'];c['input_ids']=read(ROOT/'data/c_input.json')['input_ids']
        cr,cs=c_diagnostic(model,tok,c,payload,cfg,out,eos,refs);results+=cr
        write(out/'RESULTS.json',results)
        check_parameters(model,versions);assert clean(model)
        write(out/'COST.json',dict(wall_seconds=time.time()-started,peak_allocated=torch.cuda.max_memory_allocated(),condition_records=len(results),output_tokens=sum(len(r['output_ids']) for r in results),controlled_readouts=8))
        del model;gc.collect();torch.cuda.empty_cache();assert verify_model(args.model)==expected;verify_package()
        write(out/'DECISION.json',dict(status='ENGINEERING_COMPLETE_PENDING_SCORING',c_status=cs,planned_trajectories=28,completed_trajectories=len(results),module_trained=False,bank_compiled=False,threshold_changed=False,model_sha_unchanged=True))
    except BaseException as ex:
        write(out/'FAILURE.json',dict(status='STOP_NUMERICAL' if 'STOP_NUMERICAL' in str(ex) else 'STOP_ENGINEERING',error=repr(ex),traceback=traceback.format_exc()));raise
    finally:seal_output(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);main(p.parse_args())
