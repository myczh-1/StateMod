"""Verified evidence -> descriptive code scoring -> fixed local CPU fit/validation."""
import argparse,importlib.metadata,time,traceback
import numpy as np
from common import *
from scoring import score
from student import teacher_target,validate_teachers,fit,infer,reconstruction_error,fidelity,passes

def verify_capture(run):
    mf=read(run/'manifest.json')['files']
    for n,h in mf.items():
        p=Path(n);assert not p.is_absolute() and '..' not in p.parts
        assert sha(run/p)==h,n
    assert read(run/'START.json')['freeze_sha']==sha(ROOT/'freeze.json'),'different experiment'
    return mf

def recompute_candidates(run,rec):
    c=read(run/rec['id']/'candidates.json')
    for name in ('frozenset','set'):
        ids=c[name]['token_ids'];assert ids==rec['candidates'][name]['ids']
        values=[]
        for j,token in enumerate(ids):
            with np.load(run/rec['id']/name/f'{j:03}.npz') as a:z=a['cached']
            values.append(logp(z,token))
        assert abs(sum(values)-c[name]['type_eos_logp'])<=1e-10,'candidate total mismatch'
    return c

def teacher_kl(target,logits):
    p=np.asarray(target);q=np.asarray(logits)
    entropy=np.zeros_like(p);mask=p>0;entropy[mask]+=p[mask]*np.log(p[mask]);mask=p<1;entropy[mask]+=(1-p[mask])*np.log1p(-p[mask])
    return float(np.mean(entropy+p*np.logaddexp(0,-q)+(1-p)*np.logaddexp(0,q)))

def finish(out,status,**detail):
    write(out/'DECISION.json',dict(status=status,independent_confirmation=False,V_executed=False,**detail))
    text=f'# Cloud11 判断迁移\n\n状态：`{status}`。\n\n'+json.dumps(detail,ensure_ascii=False,indent=2)+'\n\nBASE 功能成绩不作为拟合准入；Fresh差异单列。未编译或运行V。全部为已见四族的开发验证。\n'
    (out/'REPORT.md').write_text(text)

def main(args):
    verify_package();cfg=read(ROOT/'data/config.json');run=Path(args.run);out=Path(args.output);out.mkdir(exist_ok=False)
    started=time.time();training_executed=False
    try:
        for p,h in read(ROOT/'data/source_sha.json').items():assert sha(p)==h,p
        for name,version in cfg['local_fit_versions'].items():assert importlib.metadata.version(name)==version,'local fit version '+name
        verify_capture(run)
        if (run/'FAILURE.json').exists() or (run/'INTEGRITY_FAILURE.json').exists():
            finish(out,'CAPTURE_FAILED',training_executed=False,reason=read(run/'FAILURE.json') if (run/'FAILURE.json').exists() else read(run/'INTEGRITY_FAILURE.json'));return
        assert read(run/'INTEGRITY.json')['passed']
        results=read(run/'RESULTS.json');byid={r['id']:r for r in results}
        tasks=read(ROOT/'data/TRAIN_TASKS.json')+read(ROOT/'data/REFERENCE_TASKS.json');refs=read(ROOT/'data/reference.json');inputs=read(ROOT/'data/inputs.json')
        assert len(results)==24 and len(byid)==24 and set(byid)=={r['id'] for r in inputs}
        for rec in inputs:
            assert byid[rec['id']]['input_ids']==rec['input_ids'],'collected input differs from freeze'
            assert byid[rec['id']]['condition']==rec['condition']
        # Score all actual code, including baseline failures, without selection.
        scores=[]
        for task in tasks:
            r=byid[task['id']+'__BASE'];scores.append(dict(task_id=task['id'],condition='BASE',result=score(r['text'],task,r['stop']=='EOS')))
        for task,ref in zip(tasks[8:],refs):
            assert task['id']==ref['task_id'];r=ref['text_code'];scores.append(dict(task_id=task['id'],condition='ARCHIVED_TEXT_CODE',result=score(r['text'],task,r['stop']=='EOS')))
        write(out/'CODE_SCORES.json',scores)
        teachers=[]
        for rec in inputs:
            if rec['condition']=='TEXT_SELECT':teachers.append(dict(task_id=rec['task_id'],**teacher_target(byid[rec['id']],recompute_candidates(run,rec))))
        reference_teachers=[dict(task_id=r['task_id'],**teacher_target(r['teacher'],r['candidates'])) for r in refs]
        write(out/'TEACHER_TARGETS.json',dict(train=teachers,reference=reference_teachers))
        status=read(run/'DECISION.json')['status']
        if status!='CAPTURE_COMPLETE':finish(out,status,training_executed=False);return
        if not validate_teachers(teachers):finish(out,'TEACHER_INTERFACE_LIMITED',training_executed=False);return
        assert all(t['valid'] for t in reference_teachers),'archive teacher inconsistency'
        with np.load(run/'states.npz',allow_pickle=False) as a:raw=a['hidden'].astype(np.float64);valid=a['valid'];ids=a['task_ids'].tolist()
        assert ids==[t['id'] for t in tasks] and raw.shape==(16,5120) and valid.shape==(16,)
        assert valid.all() and np.isfinite(raw).all()
        for i,t in enumerate(tasks):
            p=run/(t['id']+'__BASE')
            with np.load(p/'decision.npz') as a:assert np.array_equal(raw[i],a['hidden_cached'])
            assert byid[t['id']+'__BASE']['coverage']
        assert [t['task_id'] for t in teachers]==ids[:8]
        target=np.array([t['p'] for t in teachers]);write(out/'TRAINING_INPUT_FREEZE.json',dict(train_ids=ids[:8],teacher_targets=target.tolist(),states_sha=sha(run/'states.npz'),source_manifest_sha=sha(run/'manifest.json'),heldout_used_for_fit=False,code_score_used_for_fit=False))
        w,losses=fit(raw[:8].copy(),target,cfg['module']);training_executed=True
        save(out/'judgment_14b.npz',**w)
        write(out/'CHECKPOINT_FREEZE.json',dict(sha256=sha(out/'judgment_14b.npz'),threshold=.5,train_ids=ids[:8],reference_evaluated=False,steps=256,seed=42916))
        # No reference inference or reference-dependent choice occurs above.
        logits,prob=infer(raw,w);err=reconstruction_error(raw,w)
        assert np.isfinite(logits).all() and np.isfinite(prob).all() and err<=1e-8,'STOP_NUMERICAL reconstruction'
        train_metrics=fidelity(prob[:8],teachers);ref_metrics=fidelity(prob[8:],reference_teachers)
        train_metrics['KL']=teacher_kl(target,logits[:8]);ref_metrics['KL']=teacher_kl([t['p'] for t in reference_teachers],logits[8:])
        predictions=[];allteachers=teachers+reference_teachers
        for i,t in enumerate(tasks):
            predictions.append(dict(task_id=t['id'],split='train' if i<8 else 'reference',prob=float(prob[i]),logit=float(logits[i]),threshold_margin=float(prob[i]-.5),action='frozenset' if prob[i]>.5 else 'set',teacher=allteachers[i],teacher_agreement=bool((prob[i]>.5)==(allteachers[i]['action']=='frozenset')),expected_choice=t['expected_choice'],contract_correct=bool((prob[i]>.5)==(t['expected_choice']=='frozenset'))))
        write(out/'PREDICTIONS.json',predictions);write(out/'FIDELITY_METRICS.json',dict(train=train_metrics,reference=ref_metrics));write(out/'TRAINING.json',dict(losses=losses,steps=256,local_versions=cfg['local_fit_versions'],optimizer_uses_teacher_targets_only=True))
        fresh_rows=[]
        for i,t in enumerate(tasks):
            p=run/'diagnostics'/(t['id']+'__BASE')/'decision.npz'
            if not p.exists():fresh_rows.append(dict(task_id=t['id'],status='NOT_AVAILABLE'));continue
            with np.load(p) as a:fh=a['hidden_fresh'].astype(np.float64)
            if not np.isfinite(fh).all():fresh_rows.append(dict(task_id=t['id'],status='NONFINITE_DIAGNOSTIC'));continue
            fl,fp=infer(fh[None],w)
            fresh_rows.append(dict(task_id=t['id'],status='RECORDED',cached=float(prob[i]),fresh=float(fp[0]),absolute_shift=abs(float(fp[0]-prob[i])),decision_flip=bool((fp[0]>.5)!=(prob[i]>.5)),behavior_gate=False))
        write(out/'NUMERICAL_AUDIT.json',dict(independent_checkpoint_logit_error=err,cross_path=fresh_rows,fit_uses_cached_only=True,bank_accesses=0))
        passed=passes(ref_metrics)
        finish(out,'DEVELOPMENT_14B_TEACHER_FIDELITY_SUPPORTED' if passed else 'TEACHER_FIDELITY_NOT_ESTABLISHED',training_executed=True,teacher_agreement=ref_metrics['agreement'],reference_total=8,teacher_rejects_preserved=ref_metrics['rejects_reproduced'],teacher_enables_preserved=ref_metrics['enables_reproduced'],V_eligible=passed)
    except BaseException as ex:
        write(out/'FAILURE.json',dict(status='STOP_NUMERICAL' if 'STOP_NUMERICAL' in str(ex) else 'STOP_ENGINEERING',error=repr(ex),traceback=traceback.format_exc(),training_executed=training_executed));raise
    finally:
        changed=[p for p,h in read(ROOT/'data/source_sha.json').items() if sha(p)!=h]
        write(out/'PROTECTION_AUDIT.json',dict(unchanged=not changed,changed=changed));verify_package()
        write(out/'COST.json',dict(local_seconds=time.time()-started,training_executed=training_executed,main_model_forward=0,bank_accesses=0));seal_output(out)
        assert not changed,'protected sources changed'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);main(p.parse_args())
