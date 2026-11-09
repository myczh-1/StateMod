"""Cloud11 collection only. No Bank, classifier, optimizer or remote fitting."""
import argparse,gc,importlib.metadata,time,traceback
import numpy as np
from common import *
from gate import inspect_return
from probe import capture,checkpoint_probe,prefix_valid,selection_valid
from teacher import engineering,measure,measure_candidates,compare_fresh,cache_arrays

def generate_base(model,tok,rec,out,eos,reference=None):
    p=out/rec['id'];p.mkdir();(p/'natural_steps').mkdir()
    ids=tok.apply_chat_template(rec['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    assert ids==rec['input_ids'],'STOP_ENGINEERING tokens'
    gen=[];steps=[];cache=None;gate=None;hidden=None;seen=False;started=time.monotonic()
    for j in range(rec['max_new_tokens']):
        found=inspect_return(tok.decode(gen,skip_special_tokens=True)) if gen and not seen else None
        seq=ids if j==0 else [gen[-1]]
        audited=None
        if found:
            seen=True;gate=dict(**found,prediction_step=j,absolute_position=len(ids)+j-1,
                                prefix_valid=prefix_valid(found['generated_prefix'],rec['function_name']))
            write(p/'gate.json',gate)
            if gate['prefix_valid']:
                save(p/'checkpoint_kv.npz',**cache_arrays(cache))
                zp,pc,hidden,original_hash=checkpoint_probe(model,seq,cache,p)
                audited=(zp,pc)
        # The actual primary trajectory is always the no-hook native forward.
        z,cache=forward(model,seq,cache)
        save(p/'natural_steps'/f'{j:03}.npz',cached=z)
        assert np.isfinite(z).all(),'STOP_NUMERICAL_PRIMARY generation'
        if audited:
            zp,pc=audited;actual_hash=cache_hash(cache)
            audit=dict(logits_exact=bool(np.array_equal(z,zp)),kv_exact=actual_hash==cache_hash(pc),
                       old_checkpoint_sha=original_hash,actual_cache_sha=actual_hash,
                       absolute_position=len(ids)+j-1,cache_position=list(range(len(ids)+j-len(seq),len(ids)+j)),
                       mask='all ones; no padding; causal attention',native_length=cache.get_seq_length(),
                       no_bank=True,hidden_dimension=len(hidden))
            write(p/'probe_native.json',audit)
            save(p/'decision.npz',hidden_cached=hidden,logits_cached=z)
            save(p/'decision_kv.npz',**cache_arrays(cache))
            assert audit['logits_exact'] and audit['kv_exact'],'STOP_ENGINEERING probe changed computation'
            del pc,zp,audited
        assert cache.get_seq_length()==len(ids)+j,'STOP_ENGINEERING history'
        token=int(z.argmax());row=dict(step=j,token=token,logp=logp(z,token),native_length=cache.get_seq_length(),bank_calls=0)
        write(p/'natural_steps'/f'{j:03}.json',row);steps.append(row);gen.append(token)
        if token in eos:break
    r=dict(id=rec['id'],task_id=rec['task_id'],condition='BASE',split=rec['split'],input_ids=ids,output_ids=gen,
           text=tok.decode(gen,skip_special_tokens=True),stop='EOS' if gen[-1] in eos else 'TRUNCATED',gate=gate,
           steps=steps,bank_calls=0,judgment_called=False,seconds=time.monotonic()-started)
    r['coverage']=hidden is not None and selection_valid(r['text'])
    r['coverage_reason']='CAPTURED' if r['coverage'] else ('GATE_NOT_REACHED' if not gate else 'INVALID_TYPE_DECISION_BOUNDARY')
    r['reference_token_match']=gen==reference['output_ids'] if reference else None
    write(p/'generation.json',r)
    print(rec['id'],r['stop'],r['coverage_reason'],flush=True)
    return r,hidden

def diagnostics(model,records,inputs,out):
    root=out/'diagnostics';root.mkdir();rows=[];cand=[]
    for rec in records:
        if rec['condition']!='BASE' or not rec.get('coverage'):continue
        p=root/rec['id'];p.mkdir();gate=rec['gate'];j=gate['prediction_step'];history=rec['input_ids']+rec['output_ids'][:j]
        with np.load(out/rec['id']/'decision.npz') as a:h=a['hidden_cached'];z=a['logits_cached']
        fresh,fc,hf=capture(model,history)
        save(p/'decision.npz',hidden_cached=h,hidden_fresh=hf,logits_cached=z,logits_fresh=fresh)
        row=dict(id=rec['id'],hidden=metrics(h,hf),logits=metrics(z,fresh),top1_changed=bool(z.argmax()!=fresh.argmax()),
                 selected_logp_error=abs(logp(z,int(z.argmax()))-logp(fresh,int(z.argmax()))),behavior_gate=False,
                 finite=bool(np.isfinite(hf).all() and np.isfinite(fresh).all()))
        write(p/'decision.json',row);rows.append(row);del fc
    for rec in inputs:
        if rec['condition']!='TEXT_SELECT':continue
        summaries=read(out/rec['id']/'candidates.json')
        for name,c in summaries.items():
            fresh_total=0;cr=[]
            for j,token in enumerate(c['token_ids']):
                p=root/rec['id']/name/f'{j:03}';p.parent.mkdir(parents=True,exist_ok=True)
                with np.load(out/rec['id']/name/f'{j:03}.npz') as a:z=a['cached']
                row,cache=compare_fresh(model,rec['input_ids']+c['token_ids'][:j],z,token,p)
                fresh_total+=row['fresh_logp'];cr.append(row);del cache
            cand.append(dict(id=rec['id'],candidate=name,cached=c['type_eos_logp'],fresh=fresh_total,
                             error=abs(c['type_eos_logp']-fresh_total),steps=cr))
    write(root/'SUMMARY.json',dict(decisions=rows,candidates=cand,behavior_gate=False))

def main(args):
    verify_package();cfg=read(ROOT/'data/config.json')
    for name,version in cfg['runtime_versions'].items():assert importlib.metadata.version(name)==version,'STOP_ENGINEERING runtime version '+name
    assert not (ROOT/'data/bank.npz').exists() and not (ROOT/'data/judgment.npz').exists()
    out=Path(args.output);out.mkdir(exist_ok=False);started=time.time();model=None;versions=None;expected=None
    records=[];states=[];covered=[];state_ids=[]
    write(out/'START.json',dict(time=started,freeze_sha=sha(ROOT/'freeze.json'),mode='COLLECTION_ONLY'))
    try:
        import torch
        from transformers import AutoTokenizer
        expected=verify_model(args.model);tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        model=load_model(args.model);versions=parameter_versions(model)
        assert model.config.hidden_size==5120 and model.config.num_hidden_layers==40
        eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
        engineering(model,read(ROOT/'data/engineering_input.json'),out/'engineering')
        inputs=read(ROOT/'data/inputs.json');refs={r['task_id']:r for r in read(ROOT/'data/reference.json')}
        for rec in inputs:
            if rec['condition']=='BASE':
                r,h=generate_base(model,tok,rec,out,eos,refs.get(rec['task_id'],{}).get('base'))
                states.append(h if h is not None else np.zeros(5120,dtype=np.float32));covered.append(r['coverage']);state_ids.append(rec['task_id'])
            else:r=measure(model,tok,rec,out,eos)
            records.append(r)
        write(out/'RESULTS.json',records)
        save(out/'states.npz',hidden=np.stack(states),valid=np.array(covered),task_ids=np.array(state_ids))
        for rec in inputs:
            if rec['condition']=='TEXT_SELECT':measure_candidates(model,tok,rec,out)
        write(out/'CACHED_COMPLETE.json',dict(status='CACHED_COLLECTION_COMPLETE',records=len(records),states=len(states),covered=sum(covered)))
        check_parameters(model,versions)
        try:diagnostics(model,records,inputs,out)
        except Exception as ex:
            check_parameters(model,versions)
            write(out/'DIAGNOSTIC_FAILURE.json',dict(error=repr(ex),traceback=traceback.format_exc(),behavior_gate=False))
        mismatch=[r['id'] for r in records if r.get('reference_token_match') is False]
        status='REFERENCE_NOT_REPRODUCED' if mismatch else ('CAPTURE_COMPLETE' if all(covered) else 'COVERAGE_LIMITED')
        write(out/'DECISION.json',dict(status=status,reference_mismatches=mismatch,covered=sum(covered),planned_states=16,
                                       bank_calls=0,module_fitted=False,base_functional_gate=False))
        write(out/'COST.json',dict(seconds=time.time()-started,peak_allocated=torch.cuda.max_memory_allocated(),
              natural_records=len(records),output_tokens=sum(len(r['output_ids']) for r in records),bank_loaded=False,judgment_loaded=False))
    except BaseException as ex:
        write(out/'FAILURE.json',dict(status='STOP_NUMERICAL_PRIMARY' if 'STOP_NUMERICAL_PRIMARY' in str(ex) else 'STOP_ENGINEERING',error=repr(ex),traceback=traceback.format_exc()));raise
    finally:
        try:
            if model is not None and versions is not None:check_parameters(model,versions)
            if model is not None:del model;gc.collect();torch.cuda.empty_cache()
            if expected is not None:assert verify_model(args.model)==expected
            verify_package();write(out/'INTEGRITY.json',dict(passed=True,model_sha_verified=expected is not None,model_parameters_checked=versions is not None,bank_loaded=False,judgment_loaded=False))
        except BaseException as ex:write(out/'INTEGRITY_FAILURE.json',dict(error=repr(ex)));raise
        finally:seal_output(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);main(p.parse_args())
