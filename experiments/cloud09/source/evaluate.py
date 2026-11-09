"""Separate selection and complete-code outcomes; no fitting or repair."""
import argparse,math
from common import *
from scoring import score

def choice_winner(candidates,key):
    a,b=candidates['frozenset'][key],candidates['set'][key]
    return 'frozenset' if a>b else 'set' if b>a else 'TIE'

def summarize(rows,tasks,criteria,engineering=True,failure=None):
    idx={(r['task_id'],r['condition']):r for r in rows}
    assert len(idx)==len(rows)
    total=all((t['id'],c) in idx for t in tasks for c in ['BASE','TEXT_CODE','TEXT_SELECT'])
    groups={}
    for c in ['BASE','TEXT_CODE']:
        groups[c]={}
        for cat in ['E','N']:
            rr=[idx[t['id'],c] for t in tasks if t['category']==cat and (t['id'],c) in idx]
            groups[c][cat]=dict(planned=4,completed=len(rr),functional=sum(bool(r['score']['functional']) for r in rr),joint=sum(bool(r['score']['joint']) for r in rr),format_ok=sum(bool(r['score']['format_ok']) for r in rr),covered=sum(bool(r['generation'].get('coverage')) for r in rr))
    ss=[r for r in rows if r['condition']=='TEXT_SELECT']
    select=dict(completed=len(ss),valid=sum(r['generation']['valid'] for r in ss),correct=sum(r['correct'] for r in ss),N_correct=sum(r['correct'] and r['category']=='N' for r in ss),E_correct=sum(r['correct'] and r['category']=='E' for r in ss),natural_candidate_disagreement=[r['task_id'] for r in ss if r['generation']['decision']!=r['candidate_type_eos_choice']],invalid=[r['task_id'] for r in ss if not r['generation']['valid']])
    pairs=[]
    for t in tasks:
        b=idx.get((t['id'],'BASE'));tx=idx.get((t['id'],'TEXT_CODE'))
        if not b or not tx:continue
        pairs.append(dict(id=t['id'],category=t['category'],base_functional=b['score']['functional'],text_functional=tx['score']['functional'],base_joint=b['score']['joint'],text_joint=tx['score']['joint'],regression=bool(b['score']['functional'] and not tx['score']['functional']),new_E_joint=bool(t['category']=='E' and not b['score']['joint'] and tx['score']['joint'])))
    regress=[p['id'] for p in pairs if p['category']=='N' and p['regression']]
    base_ok=total and groups['BASE']['E']['functional']>=criteria['base_E_functional_min'] and groups['BASE']['N']['functional']>=criteria['base_N_functional_min']
    select_ok=len(ss)==8 and select['valid']==criteria['select_valid'] and select['correct']>=criteria['select_correct_min'] and select['N_correct']==criteria['select_N_correct']
    code_ok=total and groups['TEXT_CODE']['E']['joint']>=criteria['text_E_joint_min'] and len(regress)<=criteria['text_N_regression_max']
    status=failure['status'] if failure else 'INCOMPLETE' if not total else 'STOP_ENGINEERING' if not engineering else 'TEXT_REFERENCE_READY_IN_DEVELOPMENT' if base_ok and select_ok and code_ok else 'TEXT_REFERENCE_NOT_READY'
    decision=dict(status=status,complete=total,baseline_coverage_ready=bool(base_ok),selection_reference='SELECT_REFERENCE_READY' if select_ok and engineering else 'SELECT_REFERENCE_NOT_READY',code_reference='CODE_REFERENCE_READY' if code_ok and base_ok and engineering else 'CODE_REFERENCE_NOT_READY',bank_tested=False,module_tested=False,independent_confirmation=False,automatic_next_stage=False)
    return dict(code=groups,select=select,pairs=pairs,N_new_regressions=regress,new_E_joint=sum(p['new_E_joint'] for p in pairs)),decision

def main(args):
    verify_package();run=Path(args.run);out=Path(args.output);out.mkdir(exist_ok=False)
    for n,h in read(run/'manifest.json')['files'].items():assert sha(run/n)==h,n
    inputs=read(ROOT/'data/inputs.json');tasks=read(ROOT/'data/evaluation.json');td={t['id']:t for t in tasks};rows=[];missing=[]
    for rec in inputs:
        p=run/rec['id']/'result.json'
        if not p.exists():missing.append(rec['id']);continue
        g=read(p);t=td[rec['task_id']]
        r=dict(task_id=t['id'],family=t['family'],category=t['category'],condition=rec['condition'],source=str(p),generation=g)
        if rec['condition']=='TEXT_SELECT':
            r.update(expected=t['expected_choice'],correct=g['decision']==t['expected_choice'],candidate_type_choice=choice_winner(g['candidates'],'type_logp'),candidate_type_eos_choice=choice_winner(g['candidates'],'type_eos_logp'),fresh_type_eos_choice=choice_winner(g['candidates'],'fresh_type_eos_logp'))
        else:r['score']=score(g['text'],t,g['stop']=='EOS')
        rows.append(r)
    write(out/'SCORED_RESULTS.json',rows)
    changed=[p for p,h in read(ROOT/'data/source_sha.json').items() if sha(p)!=h]
    eng=read(run/'DECISION.json') if (run/'DECISION.json').exists() else {}
    failure=read(run/'FAILURE.json') if (run/'FAILURE.json').exists() else None
    if changed:failure=dict(status='STOP_ENGINEERING',reason='historical source SHA changed',files=changed)
    summary,decision=summarize(rows,tasks,read(ROOT/'data/config.json')['criteria'],engineering=eng.get('status')=='TEXT_PRECHECK_COMPLETE_PENDING_SCORING' and not changed,failure=failure)
    write(out/'SUMMARY.json',summary);write(out/'DECISION.json',decision);write(out/'MISSING.json',missing)
    nums=[];candidate_totals=[]
    for r in rows:
        g=r['generation'];nums.extend(g['steps'])
        if r['condition']=='TEXT_SELECT':
            for name,c in g['candidates'].items():
                nums.extend(c['steps']);candidate_totals.append(dict(id=r['task_id'],candidate=name,type_error=abs(c['type_logp']-c['fresh_type_logp']),type_eos_error=abs(c['type_eos_logp']-c['fresh_type_eos_logp'])))
    path_flips=[dict(task=r['task_id'],condition=r['condition'],step=j,cached=s['cached_top1'],fresh=s['fresh_top1']) for r in rows for j,s in enumerate(r['generation']['steps']) if s['cached_top1']!=s['fresh_top1']]
    candidate_flips=[r['task_id'] for r in rows if r['condition']=='TEXT_SELECT' and r['candidate_type_eos_choice']!=r['fresh_type_eos_choice']]
    write(out/'PATH_SENSITIVITY.json',dict(natural_top1_flips=path_flips,candidate_rank_flips=candidate_flips,path_robust=len(path_flips)==0 and not candidate_flips,behavioral_decision=decision,interpretation='A behavioral pass with any flip is not a path-robust pass; do not fit new tolerances.'))
    write(out/'NUMERICAL_AUDIT.json',dict(steps=len(nums),max_rms=max([r['metrics']['rms'] for r in nums] or [0]),max_token_logp_error=max([r['logp_error'] for r in nums] or [0]),candidate_totals=candidate_totals))
    write(out/'CLOUD08_COMPARISON.json',dict(cloud08=read(ROOT/'data/cloud08_reference.json'),cloud09=dict(model='Qwen3-14B',dtype='bfloat16',summary=summary,decision=decision),size_only_causal_claim=False))
    write(out/'audit.json',dict(historical_sha_changed=changed,remote_engineering=eng,bank_loaded=False,module_loaded=False,scorer_sha=sha(ROOT/'src/scoring.py')))
    lines=['# Cloud09：14B BF16文字预检结果','',f"状态：`{decision['status']}`。",'', '直接选类型和完整代码分别评分；候选排序不得补救自然答案失败。',f'BF16自然步Cached/Fresh第一名差异：{len(path_flips)}处。主路径为Cached；存在差异时不宣称数值路径鲁棒通过。','14B BF16与旧8B FP32不是只改变参数规模的对照。','', '|任务|条件|直接答案/代码功能|E-JOINT或N合同|','|---|---|---|---|']
    for r in rows:
        if r['condition']=='TEXT_SELECT':v=r['generation']['decision'];j=str(r['correct'])
        else:v=str(r['score']['functional']);j=str(r['score']['joint'])
        lines.append(f"|{r['task_id']}|{r['condition']}|{v}|{j}|")
    lines+=['','未运行不记为零成功。已见4族开发预检，不代表新任务独立确认；本轮结束，不自动编译Bank或开展下一阶段。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');seal_output(out);print(decision['status'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);main(p.parse_args())
