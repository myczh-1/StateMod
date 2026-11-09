"""Cached natural behavior first; Fresh is an independent, non-gating replay."""
import argparse,gc,importlib.metadata,time,traceback
import numpy as np
from common import *
from gate import inspect_return


def cache_arrays(cache):
    return {f'{i}_{k}':getattr(layer,k).detach().float().cpu().numpy().copy()
            for i,layer in enumerate(cache.layers) for k in ('keys','values')}


def checked(model,seq,cache,history,path,target=None):
    """Only the real Cached path; never call Fresh here."""
    z,c=forward(model,seq,cache)
    save(path.with_suffix('.npz'),cached=z)
    if not np.isfinite(z).all():
        write(path.with_suffix('.json'),dict(status='STOP_NUMERICAL_PRIMARY',reason='nonfinite Cached logits'))
        raise AssertionError('STOP_NUMERICAL_PRIMARY nonfinite Cached logits')
    token=int(z.argmax()) if target is None else int(target)
    row=dict(token=token,cached_top1=int(z.argmax()),cached_logp=logp(z,token),
             native_length=c.get_seq_length(),input_length=len(history),bank_calls=0)
    write(path.with_suffix('.json'),row)
    assert c.get_seq_length()==len(history),'STOP_ENGINEERING cache history'
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()),'STOP_ENGINEERING unexpected hook'
    return z,c,row


def engineering(model,record,out):
    out.mkdir();ids=record['input_ids'];tail=record['forced_diagnostic_tokens'];a=b=None;steps=[]
    for j in range(len(tail)+1):
        seq=ids if j==0 else [tail[j-1]]
        za,a=forward(model,seq,a);zb,b=forward(model,seq,b)
        save(out/f'{j:03}.npz',repeat_a=za,repeat_b=zb)
        assert np.isfinite(za).all() and np.isfinite(zb).all(),'STOP_NUMERICAL_PRIMARY engineering nonfinite'
        row=dict(step=j,repeat_logits_exact=bool(np.array_equal(za,zb)),repeat_cache_exact=cache_hash(a)==cache_hash(b),
                 independent_cache=all(getattr(x,k).data_ptr()!=getattr(y,k).data_ptr() for x,y in zip(a.layers,b.layers) for k in ('keys','values')),
                 length=a.get_seq_length(),expected_length=len(ids)+j)
        write(out/f'{j:03}.json',row);steps.append(row)
        if j==1:save(out/'001_cached_kv.npz',**cache_arrays(a))
        assert row['repeat_logits_exact'] and row['repeat_cache_exact'] and row['independent_cache'],'STOP_ENGINEERING repeated Cached path'
        assert a.get_seq_length()==b.get_seq_length()==len(ids)+j,'STOP_ENGINEERING repeated history'
    write(out/'DECISION.json',dict(status='CACHED_ENGINEERING_PASSED',steps=steps))


def summarize_candidate(steps,ids):
    return dict(token_ids=ids,name_token_count=len(ids)-1,total_token_count=len(ids),
                type_logp=sum(s['cached_logp'] for s in steps[:-1]),
                type_eos_logp=sum(s['cached_logp'] for s in steps),steps=steps)


def measure(model,tok,record,out,eos):
    p=out/record['id'];p.mkdir();(p/'natural_steps').mkdir()
    ids=tok.apply_chat_template(record['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    assert ids==record['input_ids'],'STOP_ENGINEERING Tokenizer or input'
    gen=[];steps=[];cache=None;gate=None;start=time.monotonic()
    for j in range(record['max_new_tokens']):
        if record['condition']!='TEXT_SELECT' and gen and gate is None:
            found=inspect_return(tok.decode(gen,skip_special_tokens=True))
            if found:gate=dict(**found,prediction_step=j,absolute_position=len(ids)+j-1)
        z,cache,row=checked(model,ids if j==0 else [gen[-1]],cache,ids+gen,p/'natural_steps'/f'{j:03}')
        token=int(z.argmax());steps.append(row);gen.append(token)
        if token in eos:break
    result=dict(id=record['id'],task_id=record['task_id'],condition=record['condition'],input_ids=ids,output_ids=gen,
                text=tok.decode(gen,skip_special_tokens=True),stop='EOS' if gen[-1] in eos else 'TRUNCATED',gate=gate,
                coverage=gate is not None,steps=steps,bank_calls=0,judgment_called=False,seconds=time.monotonic()-start)
    if record['condition']=='TEXT_SELECT':
        result['valid']=result['stop']=='EOS' and result['text'].strip() in ('frozenset','set')
        result['decision']=result['text'].strip() if result['valid'] else 'INVALID'
    write(p/'generation.json',result);write(p/'result.json',result)
    print(record['id'],result['stop'],repr(result['text'][:90]),flush=True)
    return result


def measure_candidates(model,tok,record,out):
    """Auxiliary scoring starts only after all 24 natural results are persisted."""
    ids=record['input_ids'];result={}
    rendered=tok.apply_chat_template(record['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    for name in ('frozenset','set'):
        seq=record['candidates'][name]['ids'];name_ids=tok.encode(name,add_special_tokens=False)
        assert seq==name_ids+[151645] and tok.encode(rendered+name,add_special_tokens=False)==ids+name_ids
        p=out/record['id']/name;p.mkdir();history=[];cache=None;steps=[]
        for j,token in enumerate(seq):
            z,cache,row=checked(model,ids if j==0 else [history[-1]],cache,ids+history,p/f'{j:03}',target=token)
            row['rank']=int(np.sum(z>z[token])+1);steps.append(row);history.append(token)
        result[name]=summarize_candidate(steps,seq);write(p/'summary.json',result[name])
    write(out/record['id']/'candidates.json',result)
    return result


def compare_fresh(model,history,cached,token,path):
    """Same observed history, independent empty cache. Magnitude/argmax are descriptive."""
    fresh,fc=forward(model,history)
    save(path.with_suffix('.npz'),cached=cached,fresh=fresh)
    assert fc.get_seq_length()==len(history),'STOP_ENGINEERING Fresh history'
    if not np.isfinite(fresh).all():
        write(path.with_suffix('.json'),dict(status='DIAGNOSTIC_NONFINITE',input_length=len(history)))
        raise RuntimeError('DIAGNOSTIC_NONFINITE Fresh logits')
    cfg=read(ROOT/'data/config.json');a=logp(cached,token);b=logp(fresh,token);m=metrics(cached,fresh)
    row=dict(status='RECORDED',token=int(token),cached_top1=int(cached.argmax()),fresh_top1=int(fresh.argmax()),
             cached_logp=a,fresh_logp=b,logp_error=abs(a-b),metrics=m,input_length=len(history),fresh_length=fc.get_seq_length(),
             reference_exceeded=m['rms']>cfg['cached_fresh_logits_rms_max'] or abs(a-b)>cfg['selected_logp_error_max'],
             top1_changed=bool(cached.argmax()!=fresh.argmax()),behavior_gate=False)
    write(path.with_suffix('.json'),row)
    return row,fc


def diagnostics(model,records,inputs,out):
    root=out/'diagnostics';root.mkdir();er=read(ROOT/'data/engineering_input.json');eng=[]
    for j in range(len(er['forced_diagnostic_tokens'])+1):
        p=root/'engineering'/f'{j:03}';p.parent.mkdir(exist_ok=True)
        with np.load(out/'engineering'/f'{j:03}.npz') as a:z=a['repeat_a'].copy()
        row,cache=compare_fresh(model,er['input_ids']+er['forced_diagnostic_tokens'][:j],z,int(z.argmax()),p)
        if j==1:save(root/'engineering/001_fresh_kv.npz',**cache_arrays(cache))
        eng.append(row);del cache
    natural=[];candidate=[]
    for g in records:
        for j,token in enumerate(g['output_ids']):
            p=root/g['id']/'natural_steps'/f'{j:03}';p.parent.mkdir(parents=True,exist_ok=True)
            with np.load(out/g['id']/'natural_steps'/f'{j:03}.npz') as a:z=a['cached'].copy()
            row,cache=compare_fresh(model,g['input_ids']+g['output_ids'][:j],z,token,p)
            natural.append(dict(id=g['id'],step=j,**row));del cache
    for rec in inputs:
        if rec['condition']!='TEXT_SELECT':continue
        summaries=read(out/rec['id']/'candidates.json')
        for name,c in summaries.items():
            fresh_steps=[]
            for j,token in enumerate(c['token_ids']):
                p=root/rec['id']/name/f'{j:03}';p.parent.mkdir(parents=True,exist_ok=True)
                with np.load(out/rec['id']/name/f'{j:03}.npz') as a:z=a['cached'].copy()
                row,cache=compare_fresh(model,rec['input_ids']+c['token_ids'][:j],z,token,p)
                fresh_steps.append(row);del cache
            fs=sum(s['fresh_logp'] for s in fresh_steps[:-1]);fe=sum(s['fresh_logp'] for s in fresh_steps)
            cr=dict(id=rec['id'],candidate=name,cached_type_logp=c['type_logp'],fresh_type_logp=fs,
                    cached_type_eos_logp=c['type_eos_logp'],fresh_type_eos_logp=fe,
                    type_error=abs(c['type_logp']-fs),type_eos_error=abs(c['type_eos_logp']-fe))
            cr['reference_exceeded']=max(cr['type_error'],cr['type_eos_error'])>read(ROOT/'data/config.json')['candidate_logp_error_max']
            candidate.append(cr)
    write(root/'SUMMARY.json',dict(engineering=eng,natural=natural,candidates=candidate))
    write(root/'DECISION.json',dict(status='DIAGNOSTICS_COMPLETE',behavior_gate=False))


def main(args):
    verify_package();cfg=read(ROOT/'data/config.json')
    for name,expected_version in cfg['runtime_versions'].items():assert importlib.metadata.version(name)==expected_version,'STOP_ENGINEERING version '+name
    assert not (ROOT/'data/bank.npz').exists() and not (ROOT/'data/judgment.npz').exists()
    out=Path(args.output);out.mkdir(exist_ok=False);started=time.time();model=None;versions=None;expected=None;records=[]
    write(out/'START.json',dict(time=started,freeze_sha=sha(ROOT/'freeze.json')))
    try:
        import torch
        from transformers import AutoTokenizer
        expected=verify_model(args.model);tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        model=load_model(args.model);versions=parameter_versions(model)
        eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos;assert 151645 in eos
        engineering(model,read(ROOT/'data/engineering_input.json'),out/'engineering')
        inputs=read(ROOT/'data/inputs.json')
        for rec in inputs:records.append(measure(model,tok,rec,out,eos))
        write(out/'RESULTS.json',records);check_parameters(model,versions)
        write(out/'BEHAVIOR_COMPLETE.json',dict(status='CACHED_BEHAVIOR_COMPLETE_PENDING_SCORING',completed=24,time=time.time()))
        for rec in inputs:
            if rec['condition']=='TEXT_SELECT':measure_candidates(model,tok,rec,out)
        try:diagnostics(model,records,inputs,out)
        except Exception as ex:
            write(out/'DIAGNOSTIC_FAILURE.json',dict(status='DIAGNOSTIC_INCOMPLETE',error=repr(ex),traceback=traceback.format_exc(),behavior_gate=False))
            print('DIAGNOSTIC_INCOMPLETE: saved behavior retained',flush=True)
        write(out/'COST.json',dict(model_wall_seconds=time.time()-started,peak_allocated=torch.cuda.max_memory_allocated(),natural_records=len(records),
              natural_output_tokens=sum(len(r['output_ids']) for r in records),bank_loaded=False,judgment_loaded=False))
    except BaseException as ex:
        write(out/'FAILURE.json',dict(status='STOP_NUMERICAL_PRIMARY' if 'STOP_NUMERICAL_PRIMARY' in str(ex) else 'STOP_ENGINEERING',error=repr(ex),traceback=traceback.format_exc()));raise
    finally:
        try:
            if model is not None and versions is not None:check_parameters(model,versions)
            if model is not None:del model;gc.collect();torch.cuda.empty_cache()
            if expected is not None:assert verify_model(args.model)==expected,'Model file SHA changed'
            verify_package()
            write(out/'INTEGRITY.json',dict(passed=True,model_sha_verified=expected is not None,model_parameters_checked=versions is not None,bank_loaded=False,judgment_loaded=False))
        except BaseException as ex:
            write(out/'INTEGRITY_FAILURE.json',dict(status='STOP_ENGINEERING',error=repr(ex)));raise
        finally:seal_output(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);main(p.parse_args())
