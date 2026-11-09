"""Unchanged execution scorer; separate descriptive A/B/C results, no parameter selection."""
import argparse, ast
from pathlib import Path
import numpy as np
from common import *
from scoring import score

def mutation_calls(s):
    try:
        tree=ast.parse(s['code'])
        return [n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('add','remove')]
    except (SyntaxError,TypeError,KeyError):return None

def main(args):
    verify_package();run=Path(args.run);out=Path(args.output);out.mkdir(exist_ok=False)
    for n,h in read(run/'manifest.json')['files'].items():assert sha(run/n)==h,n
    tasks={t['id']:t for t in read(ROOT/'TASKS.json')};m=read(ROOT/'MATRIX.json');ct=m['c']['task'];rows=[]
    for p in sorted(run.glob('*/*/generation.json')):
        g=read(p);t=ct if p.parent.parent.name=='C' else tasks[g['id']]
        s=score(g['text'],t,g['stop']=='EOS')
        rows.append(dict(id=g['id'],condition=g['condition'],source=str(p),score=s,generation=g,mutation_calls=mutation_calls(s)))
    write(out/'SCORED_RESULTS.json',rows);idx={(r['id'],r['condition']):r for r in rows}
    families=list(dict.fromkeys(t['family'] for t in tasks.values()))
    a=[]
    for f in families:
        x=idx.get((f+'_N_W_INTERFACE','BASE'));y=idx.get((f+'_N_CLEAR_CALLER','BASE'))
        a.append(dict(family=f,old=None if not x else dict(functional=x['score']['functional'],mutation_calls=x['mutation_calls'],code=x['score'].get('code')),clear=None if not y else dict(functional=y['score']['functional'],mutation_calls=y['mutation_calls'],code=y['score'].get('code'))))
    recovered=sum(bool(r['old'] and r['clear'] and r['old']['mutation_calls'] and r['clear']['mutation_calls']==[] and not r['old']['functional'] and r['clear']['functional']) for r in a)
    regress=sum(bool(r['old'] and r['clear'] and r['old']['functional'] and not r['clear']['functional']) for r in a)
    astatus='WORDING_DISAMBIGUATION_EFFECT_OBSERVED' if recovered>=2 and regress==0 else 'WORDING_PATTERN_CHANGED_FUNCTION_NOT_RECOVERED' if any(r['old'] and r['clear'] and r['old']['mutation_calls'] and r['clear']['mutation_calls']==[] for r in a) else 'WORDING_EFFECT_NOT_ESTABLISHED'
    write(out/'A_WORDING.json',dict(status=astatus,rows=a,recovered=recovered,regressions=regress,references=read(run/'A_REFERENCE.json') if (run/'A_REFERENCE.json').exists() else None,note='AST counts are descriptive; inspect code for legitimate computation vs misplaced acceptance actions'))
    b=[]
    for item in m['controlled_readouts']:
        ident=item['task_id'];base=idx.get((ident,'BASE'));text=idx.get((ident,'TEXT_CODE'));p=run/'controlled_readouts'/ident/'READOUT.json'
        b.append(dict(id=ident,category=tasks[ident]['category'],natural_judgment=base['generation'].get('judgment') if base else None,controlled_judgment=read(p)['judgment'] if p.exists() else None,base_functional=base['score']['functional'] if base else None,text_functional=text['score']['functional'] if text else None,text_joint=text['score']['joint'] if text else None))
    write(out/'B_MATCHER.json',dict(rows=b,threshold=.5,trained=False))
    c=[dict(condition=r['condition'],functional=r['score']['functional'],joint=r['score']['joint'],exact_frozenset=r['score']['exact_frozenset'],code=r['score'].get('code'),failures=r['score']['failure_reasons'],forced=r['generation'].get('forced_constructor',False)) for r in rows if r['id']==ct['id']]
    compare=[]
    for typ in ['SET','FRO']:
        pa,pb=[run/'C'/(mode+'_'+typ) for mode in ['OFF','ON']]
        if not (pa/'constructor_boundary.json').exists() or not (pb/'constructor_boundary.json').exists():continue
        x,y=read(pa/'constructor_boundary.json'),read(pb/'constructor_boundary.json');assert x['history_ids']==y['history_ids']
        with np.load(pa/'constructor_boundary_logits.npz') as f,np.load(pb/'constructor_boundary_logits.npz') as g:
            az,bz=f['logits'],g['logits'];diff=metrics(az,bz)
        layers={}
        with np.load(pa/'constructor_boundary_cache.npz') as f,np.load(pb/'constructor_boundary_cache.npz') as g:
            for key in f.files:layers[key]=metrics(f[key],g[key])
        compare.append(dict(constructor=typ,same_history=True,logits=diff,off_top1=int(az.argmax()),on_top1=int(bz.argmax()),cache_layers=layers))
    write(out/'C_V_COMPUTATION.json',dict(rows=c,same_history_comparisons=compare,forced_branches_not_natural_success=True))
    nums=[s for r in rows for s in r['generation']['steps'] if 'numerical' in s]
    audit=dict(completed=len(rows),model_job=read(run/'DECISION.json') if (run/'DECISION.json').exists() else None,failure=read(run/'FAILURE.json') if (run/'FAILURE.json').exists() else None,max_logits_rms=max([s['numerical']['rms'] for s in nums] or [0]),max_logp_error=max([s['selected_logp_error'] for s in nums] or [0]),historical_sha_changed=[p for p,h in read(ROOT/'data/source_sha.json').items() if sha(p)!=h],bank_sha=sha(ROOT/'data/bank.npz'),judgment_sha=sha(ROOT/'data/judgment.npz'))
    write(out/'audit.json',audit)
    (out/'REPORT.md').write_text('# Cloud07 初步分离结果\n\n'+f"轨迹 {len(rows)}/28。A：{astatus}。\n\n"+'B、C逐题结果见对应JSON；强制构造器分支不计自然生成成绩。未运行不计零成功。\n')
    seal_output(out);print('Offline scoring complete',len(rows),astatus)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);main(p.parse_args())
