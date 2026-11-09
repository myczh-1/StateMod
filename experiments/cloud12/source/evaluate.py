"""Local fixed-readout analysis AFTER verified retrieval and shutdown; never fits."""
import argparse
import numpy as np
from common import *
from readout import chain,factorial

def main(args):
    verify_package();run=Path(args.run);out=Path(args.output);out.mkdir(exist_ok=False)
    for n,h in read(run/'manifest.json')['files'].items():assert sha(run/n)==h,n
    for n,h in read(ROOT/'data/source_sha.json').items():assert sha(n)==h,n
    if (run/'FAILURE.json').exists() or not (run/'DECISION.json').exists() or (run/'INTEGRITY_FAILURE.json').exists():
        write(out/'DECISION.json',dict(status='NO_MECHANISM_CLAIM',failure=read(run/'FAILURE.json') if (run/'FAILURE.json').exists() else 'INCOMPLETE_OR_INTEGRITY_FAILURE'))
        seal_output(out);return
    assert read(run/'DECISION.json')['status']=='DIAGNOSTIC_CAPTURE_COMPLETE'
    assert read(run/'INTEGRITY.json')['passed']
    with np.load(ROOT/'data/frozen_readout.npz',allow_pickle=False) as a:w=dict(a)
    records=read(ROOT/'data/inputs.json');raw=[];fresh=[]
    for r in records:
        with np.load(run/r['id']/'decision.npz',allow_pickle=False) as a:raw.append(a['hidden_cached'])
        f=run/r['id']/'fresh.npz'
        if f.exists():
            with np.load(f,allow_pickle=False) as a:fresh.append(a['hidden_fresh'])
        else:fresh.append(np.full_like(raw[-1],np.nan))
    values=chain(np.stack(raw),w);save(out/'LAYER_VALUES.npz',ids=np.array([r['id'] for r in records]),**values)
    rows=[]
    for i,r in enumerate(records):
        q={k:r[k] for k in ['id','family','wording','format','absolute_position','prompt_tokens','prefix_tokens']}
        q.update(logit=float(values['logits'][i]),probability=float(values['probability'][i]),enable=bool(values['probability'][i]>.5),
            contributions=values['contribution'][i].tolist(),preactivation=values['preactivation'][i].tolist(),threshold=.5,forced_prefix=True)
        if r['historical_reference']:
            err=abs(q['probability']-r['historical_reference']['probability']);q['old_score_error']=err
            assert err<=1e-12,'REFERENCE_READOUT_NOT_REPRODUCED'
        if np.isfinite(fresh[i]).all():
            fv=chain(np.array(fresh[i])[None],w)
            q['fresh']=dict(logit=float(fv['logits'][0]),probability=float(fv['probability'][0]),decision_flip=bool((fv['probability'][0]>.5)!=q['enable']))
        else:q['fresh']=dict(status='DIAGNOSTIC_MISSING_OR_NONFINITE')
        rows.append(q)
    effects=factorial(rows);write(out/'SCORES.json',rows);write(out/'FACTORIAL_EFFECTS.json',effects)
    delta=[]
    for family in ['boolean_complement','lower']:
        indexes={(r['wording'],r['format']):i for i,r in enumerate(records) if r['family']==family}
        for name,a,b in [('wording_bare',('train','bare'),('heldout','bare')),('wording_fenced',('train','fenced'),('heldout','fenced')),('format_train',('train','bare'),('train','fenced')),('format_heldout',('heldout','bare'),('heldout','fenced'))]:
            i,j=indexes[a],indexes[b];stages={}
            for stage in ['raw','rms','preprocessed','preactivation','gelu']:
                x,y=values[stage][i],values[stage][j]
                stages[stage]=dict(cos=float(x@y/max(float(np.linalg.norm(x)*np.linalg.norm(y)),1e-30)),**metrics(x,y))
            delta.append(dict(family=family,effect=name,stages=stages,output_contribution_change=(values['contribution'][j]-values['contribution'][i]).tolist()))
    write(out/'STATE_CHANGES.json',delta)
    write(out/'DECISION.json',dict(status='CONTROLLED_DIAGNOSTIC_COMPLETE',not_independent_confirmation=True,not_natural_generation=True,
         no_threshold_selection=True,training_steps=0,bank_calls=0,old_cloud11_status='TEACHER_FIDELITY_NOT_ESTABLISHED',
         interpretation='Report wording/format effects and interaction separately; absolute-position effects remain coupled.'))
    lines=['# Cloud12 措辞与代码格式交叉诊断','', '固定前缀诊断；不是自然生成或独立确认。旧模块与 0.5 阈值不变。','',
       '|题目|措辞|格式|模块分数|是否启用|','|---|---|---|---:|---|']
    for q in rows:lines.append(f"|{q['family']}|{q['wording']}|{q['format']}|{q['probability']:.9f}|{q['enable']}|")
    lines+=['','## 因素效应（模块输出 logit）','']
    for e in effects:lines.append(str(e))
    lines+=['','围栏同时改变格式、Token 数和位置；措辞也改变内容与长度。本设计不声称分离纯长度或纯位置作用。',
        '8维分量贡献仅为固定读出算术分解，不是模型语义或神经元因果消融。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for n,h in read(ROOT/'data/source_sha.json').items():assert sha(n)==h,n
    write(out/'AUDIT.json',dict(source_unchanged=True,module_sha=sha(ROOT/'data/frozen_readout.npz'),training_steps=0,threshold=.5,bank_loaded=False))
    seal_output(out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);main(p.parse_args())
