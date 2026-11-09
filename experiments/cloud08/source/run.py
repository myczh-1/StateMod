"""Text-only baselines; no judgment/Bank imports or access."""
import argparse,gc,importlib.metadata,sys,time,traceback
import numpy as np
from common import *
from gate import inspect_return

def cache_arrays(cache):
    return {f'{i}_{k}':getattr(layer,k).detach().cpu().numpy().copy() for i,layer in enumerate(cache.layers) for k in ('keys','values')}

def checked(model,seq,cache,history,path,target=None):
    z,c=forward(model,seq,cache);fresh,fc=forward(model,history)
    save(path.with_suffix('.npz'),cached=z,fresh=fresh)
    token=int(z.argmax()) if target is None else int(target)
    row=dict(token=token,cached_top1=int(z.argmax()),fresh_top1=int(fresh.argmax()),cached_logp=logp(z,token),fresh_logp=logp(fresh,token),metrics=metrics(z,fresh),native_length=c.get_seq_length(),fresh_length=fc.get_seq_length(),bank_calls=0)
    row['logp_error']=abs(row['cached_logp']-row['fresh_logp']);write(path.with_suffix('.json'),row)
    if row['metrics']['rms']>.001 or row['logp_error']>.01:
        save(path.with_name(path.name+'_failed_cached.npz'),**cache_arrays(c));save(path.with_name(path.name+'_failed_fresh.npz'),**cache_arrays(fc))
        raise AssertionError('STOP_NUMERICAL Cached/Fresh')
    assert c.get_seq_length()==fc.get_seq_length()==len(history),'STOP_ENGINEERING cache history'
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()),'STOP_ENGINEERING unexpected hook'
    return z,c,row

def engineering(model,record,out):
    out.mkdir();ids=record['input_ids'];tail=record['forced_diagnostic_tokens'];a=b=None;steps=[]
    for j in range(len(tail)+1):
        seq=ids if j==0 else [tail[j-1]]
        za,a=forward(model,seq,a);zb,b=forward(model,seq,b)
        f,fc=forward(model,ids+tail[:j]);save(out/f'{j:03}.npz',repeat_a=za,repeat_b=zb,fresh=f)
        row=dict(step=j,repeat_logits_exact=bool(np.array_equal(za,zb)),repeat_cache_exact=cache_hash(a)==cache_hash(b),independent_cache=all(getattr(x,k).data_ptr()!=getattr(y,k).data_ptr() for x,y in zip(a.layers,b.layers) for k in ['keys','values']),fresh=metrics(za,f))
        steps.append(row);write(out/f'{j:03}.json',row)
        assert row['repeat_logits_exact'] and row['repeat_cache_exact'] and row['independent_cache'],'STOP_ENGINEERING repeated cached path'
        if row['fresh']['rms']>.001:
            save(out/f'{j:03}_failed_cached.npz',**cache_arrays(a));save(out/f'{j:03}_failed_fresh.npz',**cache_arrays(fc));raise AssertionError('STOP_NUMERICAL engineering')
    write(out/'DECISION.json',dict(status='TEXT_ONLY_ENGINEERING_PASSED',steps=steps,bank_loaded=False,judge_loaded=False))

def summarize_candidate(steps,ids):
    return dict(token_ids=ids,name_token_count=len(ids)-1,total_token_count=len(ids),type_logp=sum(s['cached_logp'] for s in steps[:-1]),fresh_type_logp=sum(s['fresh_logp'] for s in steps[:-1]),type_eos_logp=sum(s['cached_logp'] for s in steps),fresh_type_eos_logp=sum(s['fresh_logp'] for s in steps),steps=steps)

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
    # Full cached/fresh vocabulary arrays are already persisted per prediction.
    r=dict(id=record['id'],task_id=record['task_id'],condition=record['condition'],input_ids=ids,output_ids=gen,text=tok.decode(gen,skip_special_tokens=True),stop='EOS' if gen[-1] in eos else 'TRUNCATED',gate=gate,coverage=gate is not None,steps=steps,bank_calls=0,judgment_called=False,seconds=time.monotonic()-start)
    write(p/'generation.json',r);del cache
    if record['condition']=='TEXT_SELECT':
        r['valid']=r['stop']=='EOS' and r['text'].strip() in ['frozenset','set']
        r['decision']=r['text'].strip() if r['valid'] else 'INVALID';r['candidates']={}
        rendered=tok.apply_chat_template(record['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        for name in ['frozenset','set']:
            seq=record['candidates'][name]['ids'];name_ids=tok.encode(name,add_special_tokens=False)
            assert seq==name_ids+[151645] and tok.encode(rendered+name,add_special_tokens=False)==ids+name_ids
            history=[];cache=None;cs=[];cp=p/name;cp.mkdir()
            for j,token in enumerate(seq):
                z,cache,row=checked(model,ids if j==0 else [history[-1]],cache,ids+history,cp/f'{j:03}',target=token)
                row['rank']=int(np.sum(z>z[token])+1);cs.append(row);history.append(token)
            summary=summarize_candidate(cs,seq);write(cp/'summary.json',summary)
            if any(abs(summary[k]-summary['fresh_'+k])>.01 for k in ['type_logp','type_eos_logp']):
                save(cp/'failed_final_cache.npz',**cache_arrays(cache));raise AssertionError('STOP_NUMERICAL full candidate')
            r['candidates'][name]=summary;del cache
    write(p/'result.json',r);print(record['id'],r['stop'],repr(r['text'][:90]),flush=True)
    return r

def main(args):
    verify_package();cfg=read(ROOT/'data/config.json')
    for name,expected in cfg['runtime_versions'].items():assert importlib.metadata.version(name)==expected,'STOP_ENGINEERING version '+name
    assert not (ROOT/'data/bank.npz').exists() and not (ROOT/'data/judgment.npz').exists()
    out=Path(args.output);out.mkdir(exist_ok=False);started=time.time();write(out/'START.json',dict(time=started,freeze_sha=sha(ROOT/'freeze.json')))
    try:
        import torch
        from transformers import AutoTokenizer
        expected=verify_model(args.model);tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        model=load_model(args.model);versions=parameter_versions(model)
        eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
        assert 151645 in eos
        engineering(model,read(ROOT/'data/engineering_input.json'),out/'engineering')
        records=[]
        for record in read(ROOT/'data/inputs.json'):records.append(measure(model,tok,record,out,eos))
        write(out/'RESULTS.json',records);check_parameters(model,versions)
        write(out/'COST.json',dict(model_wall_seconds=time.time()-started,peak_allocated=torch.cuda.max_memory_allocated(),natural_records=len(records),code_records=16,select_records=8,candidate_paths=16,natural_output_tokens=sum(len(r['output_ids']) for r in records),bank_loaded=False,judgment_loaded=False))
        del model;gc.collect();torch.cuda.empty_cache();assert verify_model(args.model)==expected;verify_package()
        write(out/'DECISION.json',dict(status='TEXT_PRECHECK_COMPLETE_PENDING_SCORING',completed=24,model_sha_unchanged=True,bank_loaded=False,judgment_loaded=False,independent_confirmation=False))
    except BaseException as ex:
        write(out/'FAILURE.json',dict(status='STOP_NUMERICAL' if 'STOP_NUMERICAL' in str(ex) else 'STOP_ENGINEERING',error=repr(ex),traceback=traceback.format_exc()));raise
    finally:seal_output(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);main(p.parse_args())
