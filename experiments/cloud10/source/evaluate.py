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

def diagnostic_summary(run):
    cfg=read(ROOT/'data/config.json');root=run/'diagnostics';steps=[];engineering=[]
    for p in sorted(root.rglob('*.json')) if root.exists() else []:
        x=read(p)
        if x.get('status')=='RECORDED' and 'metrics' in x:
            row=dict(path=str(p.relative_to(root)),**x)
            (engineering if p.parent.name=='engineering' else steps).append(row)
    complete=(root/'DECISION.json').exists() and read(root/'DECISION.json')['status']=='DIAGNOSTICS_COMPLETE'
    candidates=read(root/'SUMMARY.json')['candidates'] if (root/'SUMMARY.json').exists() else []
    flips=[x['path'] for x in steps if x['top1_changed'] and '/natural_steps/' in x['path']]
    candidate_flips=[]
    for rid in sorted({c['id'] for c in candidates}):
        cc={x['candidate']:x for x in candidates if x['id']==rid}
        if len(cc)==2:
            old=choice_winner(cc,'cached_type_eos_logp');fresh=choice_winner(cc,'fresh_type_eos_logp')
            if old!=fresh:candidate_flips.append(dict(id=rid,cached=old,fresh=fresh))
    all_steps=engineering+steps
    exceeded=[x['path'] for x in all_steps if x['reference_exceeded']]
    total_exceeded=[dict(id=x['id'],candidate=x['candidate']) for x in candidates if x['reference_exceeded']]
    numeric=dict(status='CROSS_PATH_DIAGNOSTIC_INCOMPLETE' if not complete else 'CROSS_PATH_DIFFERENCE_OBSERVED' if exceeded or total_exceeded or flips or candidate_flips else 'WITHIN_REGISTERED_REFERENCE_BOUNDS',
        complete=complete,behavior_gate=False,engineering_steps=len(engineering),other_steps=len(steps),
        max_rms=max([x['metrics']['rms'] for x in all_steps],default=None),
        max_token_logp_error=max([x['logp_error'] for x in all_steps],default=None),
        reference_exceeded_steps=exceeded,candidate_totals=candidates,candidate_total_exceeded=total_exceeded,
        reference_limits=dict(rms=cfg['cached_fresh_logits_rms_max'],token_logp=cfg['selected_logp_error_max'],candidate_logp=cfg['candidate_logp_error_max']),
        diagnostic_failure=read(run/'DIAGNOSTIC_FAILURE.json') if (run/'DIAGNOSTIC_FAILURE.json').exists() else None)
    sensitivity=dict(natural_top1_flips=flips,candidate_rank_flips=candidate_flips,
        no_observed_ranking_flips=complete and not flips and not candidate_flips,
        path_robust=complete and not flips and not candidate_flips and not exceeded and not total_exceeded,
        interpretation='Only same-history comparisons; no independently generated Fresh trajectory. Never replace Cached behavior or revise Cloud09.')
    return numeric,sensitivity


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
            cp=run/rec['id']/'candidates.json';c=read(cp) if cp.exists() else None
            r.update(expected=t['expected_choice'],correct=g['decision']==t['expected_choice'],
                candidate_type_choice=choice_winner(c,'type_logp') if c else None,
                candidate_type_eos_choice=choice_winner(c,'type_eos_logp') if c else None,candidate_scoring_status='COMPLETE' if c else 'NOT_RUN_OR_INCOMPLETE')
        else:r['score']=score(g['text'],t,g['stop']=='EOS')
        rows.append(r)
    write(out/'SCORED_RESULTS.json',rows)
    changed=[p for p,h in read(ROOT/'data/source_sha.json').items() if sha(p)!=h]
    eng=read(run/'engineering/DECISION.json') if (run/'engineering/DECISION.json').exists() else {}
    integrity=read(run/'INTEGRITY.json') if (run/'INTEGRITY.json').exists() else {}
    failure=read(run/'FAILURE.json') if (run/'FAILURE.json').exists() else None
    if changed or (run/'INTEGRITY_FAILURE.json').exists():failure=dict(status='STOP_ENGINEERING',reason='SHA/parameter integrity failure',files=changed)
    summary,decision=summarize(rows,tasks,read(ROOT/'data/config.json')['criteria'],
        engineering=eng.get('status')=='CACHED_ENGINEERING_PASSED' and integrity.get('passed',False) and not changed,failure=failure)
    decision.update(primary_path='CACHED',fresh_is_behavior_gate=False,old_cloud09_status='STOP_NUMERICAL')
    # Missing auxiliary probability paths must not count as a natural-answer disagreement.
    summary['select']['natural_candidate_disagreement']=[r['task_id'] for r in rows if r['condition']=='TEXT_SELECT' and r['candidate_type_eos_choice'] is not None and r['generation']['decision']!=r['candidate_type_eos_choice']]
    numeric,sensitivity=diagnostic_summary(run)
    write(out/'SUMMARY.json',summary);write(out/'DECISION.json',decision);write(out/'MISSING.json',dict(ids=missing,status='NOT_RUN_OR_INCOMPLETE' if missing else 'COMPLETE'))
    write(out/'NUMERICAL_AUDIT.json',numeric);write(out/'PATH_SENSITIVITY.json',sensitivity)
    write(out/'COMPARISON.json',dict(cloud08=read(ROOT/'data/cloud08_reference.json'),cloud09=read(ROOT/'data/cloud09_reference.json'),cloud10=dict(summary=summary,decision=decision),size_only_causal_claim=False))
    write(out/'audit.json',dict(historical_sha_changed=changed,cached_engineering=eng,integrity=integrity,bank_loaded=False,module_loaded=False,scorer_sha=sha(ROOT/'src/scoring.py')))
    lines=['# Cloud10：Cached文字能力与独立数值诊断','',f"文字能力：`{decision['status']}`。",f"数值诊断：`{numeric['status']}`。",'',
        '行为成绩只来自Cached自然输出；Fresh重放不得改变答案或补救格式失败。旧Cloud09保持STOP_NUMERICAL。',
        '8B FP32与14B BF16同时改变模型规模和精度，不能作纯规模因果归因。','',
        '|任务|条件|直接答案或功能正确|联合/合同正确|','|---|---|---|---|']
    for r in rows:
        if r['condition']=='TEXT_SELECT':value=r['generation']['decision'];ok=r['correct']
        else:value=r['score']['functional'];ok=r['score']['joint']
        lines.append(f"|{r['task_id']}|{r['condition']}|{value}|{ok}|")
    lines+=['',f"缺失输出：{len(missing)}，不计为零成功。",f"Fresh自然步第一名差异：{len(sensitivity['natural_top1_flips'])}处。跨路径稳健性：{sensitivity['path_robust']}。",'本轮是同一8题上的修订开发预检，不是独立确认；不自动训练模块或编译Bank。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');seal_output(out);print(decision['status'],numeric['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);main(p.parse_args())
