"""Eight fixed-prefix Cached replays; no free generation, Bank, or module execution."""
import argparse,gc,time,traceback
import numpy as np
from common import *
from probe import capture,checkpoint_probe,prefix_valid

def cache_arrays(cache):
    return {f'{i}_{part}':getattr(layer,part).detach().float().cpu().numpy().copy()
            for i,layer in enumerate(cache.layers) for part in ('keys','values')}

def replay_checkpoint(model,record):
    # Exactly the historical online segmentation: full prompt, then one prefix token at a time.
    _,cache=forward(model,record['input_ids'])
    for token in record['prefix_ids'][:-1]:_,cache=forward(model,[token],cache)
    assert cache.get_seq_length()==record['absolute_position']
    return cache

def measure(model,record,out):
    p=out/record['id'];p.mkdir();write(p/'INPUT.json',record)
    checkpoint=replay_checkpoint(model,record);last=record['prefix_ids'][-1:]
    save(p/'checkpoint_kv.npz',**cache_arrays(checkpoint))
    zp,pc,h,before=checkpoint_probe(model,last,checkpoint,p)
    # Native no-hook call owns the original checkpoint; probe branches own independent clones.
    z,native=forward(model,last,checkpoint)
    save(p/'decision.npz',hidden_cached=h,logits_cached=z)
    save(p/'decision_kv.npz',**cache_arrays(native))
    native_sha=cache_hash(native)
    audit=dict(probe_logits_exact=bool(np.array_equal(z,zp)),probe_kv_exact=native_sha==cache_hash(pc),
        hooks_removed=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()),
        native_length=native.get_seq_length(),absolute_position=record['absolute_position'],
        expected_length=len(record['history_ids']),checkpoint_sha=before,decision_cache_sha=native_sha,
        position_ids=[record['absolute_position']],cache_position=[record['absolute_position']],
        attention_mask='all ones; causal eager; no padding',bank_calls=0,forced_prefix=True)
    write(p/'PRIMARY_AUDIT.json',audit)
    assert audit['probe_logits_exact'] and audit['probe_kv_exact'] and audit['hooks_removed'],'STOP_ENGINEERING probe'
    assert audit['native_length']==audit['expected_length'],'STOP_ENGINEERING position'
    assert np.isfinite(h).all() and np.isfinite(z).all(),'STOP_NUMERICAL_PRIMARY nonfinite'
    ref=record['historical_reference'];reference=None
    if ref:
        d=ROOT/ref['directory']
        with np.load(d/'decision.npz',allow_pickle=False) as a:oldh=a['hidden_cached'];oldz=a['logits_cached']
        reference=dict(hidden=metrics(h,oldh),logits=metrics(z,oldz),kv_exact=native_sha==read(d/'probe_native.json')['actual_cache_sha'])
        save(p/'reference_comparison.npz',old_hidden=oldh,new_hidden=h,old_logits=oldz,new_logits=z)
        write(p/'REFERENCE_AUDIT.json',reference)
        assert reference['hidden']['exact'] and reference['logits']['exact'] and reference['kv_exact'],'REFERENCE_NOT_REPRODUCED'
    del checkpoint,native,pc
    return dict(id=record['id'],argmax=int(z.argmax()),reference=reference,primary_passed=True)

def fresh_diagnostic(model,record,out):
    p=out/record['id']
    with np.load(p/'decision.npz',allow_pickle=False) as a:h=a['hidden_cached'];z=a['logits_cached']
    zf,fc,hf=capture(model,record['history_ids'])
    save(p/'fresh.npz',hidden_fresh=hf,logits_fresh=zf)
    row=dict(hidden=metrics(h,hf),logits=metrics(z,zf),argmax_flip=bool(z.argmax()!=zf.argmax()),
        primary_gate=False,finite=bool(np.isfinite(hf).all() and np.isfinite(zf).all()))
    write(p/'FRESH_AUDIT.json',row);del fc
    return row

def main(args):
    verify_package();out=Path(args.output);out.mkdir(exist_ok=False)
    start=time.time();model=None;versions=None;expected=None
    write(out/'START.json',dict(scope='fixed_prefix_diagnostic',model_revision=read(ROOT/'data/config.json')['revision'],freeze_sha=sha(ROOT/'freeze.json')))
    try:
        import torch
        from transformers import AutoTokenizer
        expected=verify_model(args.model);tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        records=read(ROOT/'data/inputs.json')
        for r in records:
            assert tok.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)==r['input_ids']
            assert tok.encode(r['prefix_text'],add_special_tokens=False)==r['prefix_ids']
            assert prefix_valid(r['prefix_text'],r['function_name'])
        model=load_model(args.model);versions=parameter_versions(model)
        assert model.config.num_hidden_layers==40 and model.config.hidden_size==5120
        rows=[]
        for r in sorted(records,key=lambda r:r['historical_reference'] is None):
            rows.append(measure(model,r,out));print(r['id'],'CACHED_OK',flush=True)
        write(out/'RESULTS.json',rows)
        write(out/'CACHED_COMPLETE.json',dict(conditions=len(rows),historical_reference_count=4))
        diagnostic=[]
        for r in records:
            try:diagnostic.append(dict(id=r['id'],**fresh_diagnostic(model,r,out)))
            except Exception as ex:
                write(out/r['id']/'FRESH_FAILURE.json',dict(error=repr(ex),traceback=traceback.format_exc(),primary_gate=False))
        write(out/'FRESH_SUMMARY.json',diagnostic)
        write(out/'DECISION.json',dict(status='DIAGNOSTIC_CAPTURE_COMPLETE',conditions=8,bank_calls=0,training_steps=0,natural_generations=0))
        write(out/'COST.json',dict(seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(),conditions=8,source='forced existing code prefix',free_generation_tokens=0))
    except BaseException as ex:
        status='REFERENCE_NOT_REPRODUCED' if 'REFERENCE_NOT_REPRODUCED' in str(ex) else ('STOP_NUMERICAL_PRIMARY' if 'STOP_NUMERICAL_PRIMARY' in str(ex) else 'STOP_ENGINEERING')
        write(out/'FAILURE.json',dict(status=status,error=repr(ex),traceback=traceback.format_exc()));raise
    finally:
        try:
            if model is not None and versions is not None:check_parameters(model,versions)
            if model is not None:del model;gc.collect();torch.cuda.empty_cache()
            if expected is not None:assert verify_model(args.model)==expected
            verify_package();write(out/'INTEGRITY.json',dict(passed=True,model_sha_verified=expected is not None,parameters_checked=versions is not None,module_loaded=False,bank_loaded=False))
        except BaseException as ex:write(out/'INTEGRITY_FAILURE.json',dict(error=repr(ex)));raise
        finally:seal_output(out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);main(p.parse_args())
