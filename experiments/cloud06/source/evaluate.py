"""Local-only scoring and anchor replay audit after verified retrieval/shutdown."""
import argparse,ast
import numpy as np
from pathlib import Path
from common import read,write,sha,verify_package,seal_output,ROOT,metrics,logp
from scoring import score
CONDITIONS=['BASE','TEXT_CODE','MANUAL_V','MODULE_V']
WORDINGS=['W_USAGE','W_INTERFACE']

def valid_decision_prefix(g,task):
    gate=g.get('gate')
    if not gate:return False
    text=gate['generated_prefix'].lstrip()
    if text.startswith('```'):text=text.split('\n',1)[1]
    try:
        t=ast.parse(text)
        return (len(t.body)==1 and isinstance(t.body[0],ast.FunctionDef) and t.body[0].name==task['function_name'] and
                len(t.body[0].body)==1 and isinstance(t.body[0].body[0],ast.Return) and t.body[0].body[0].value is None)
    except (SyntaxError,ValueError,IndexError):return False

def covered(r):return bool(r['generation']['coverage'] and r.get('valid_decision_prefix',True))

def summarize(tasks,rows,cfg,engineering,failure=None):
    idx={(r['id'],r['condition']):r for r in rows}
    assert len(idx)==len(rows),'duplicate condition'
    summary={}
    for c in CONDITIONS:
        summary[c]={}
        for w in ['ALL']+WORDINGS:
            summary[c][w]={}
            for cat in ['E','N']:
                planned=[t for t in tasks if t['category']==cat and (w=='ALL' or t['wording']==w)]
                rr=[idx[t['id'],c] for t in planned if (t['id'],c) in idx]
                reg=[];gains=[];losses=[]
                for r in rr:
                    b=idx.get((r['id'],'BASE'))
                    if not b:continue
                    if b['score']['functional'] and not r['score']['functional']:reg.append(r['id'])
                    if not b['score']['joint'] and r['score']['joint']:gains.append(r['id'])
                    if b['score']['joint'] and not r['score']['joint']:losses.append(r['id'])
                summary[c][w][cat]=dict(planned=len(planned),completed=len(rr),not_run=len(planned)-len(rr),
                    functional=sum(bool(r['score']['functional']) for r in rr),joint=sum(bool(r['score']['joint']) for r in rr),
                    exact_frozenset=sum(bool(r['score'].get('exact_frozenset')) for r in rr),regression=reg,new_joint=gains,lost_joint=losses,
                    net_joint=len(gains)-len(losses),covered=sum(covered(r) for r in rr),
                    actual_bank_tasks=sum(r['generation']['bank_calls']>0 for r in rr))
    matching={}
    for w in WORDINGS:
        counts=dict(E_accept=0,N_reject=0,denominator_each=4,uncovered_or_missing=[],rows=[])
        for t in [t for t in tasks if t['wording']==w]:
            r=idx.get((t['id'],'MODULE_V'));g=r['generation'] if r else {}
            if not r or not covered(r) or not g.get('judgment'):
                counts['uncovered_or_missing'].append(t['id']);continue
            j=g['judgment'];counts['rows'].append(dict(id=t['id'],category=t['category'],**j))
            counts['E_accept']+=int(t['category']=='E' and j['enabled'])
            counts['N_reject']+=int(t['category']=='N' and not j['enabled'])
        matching[w]=counts
    full=len(rows)==len(tasks)*4
    cr=cfg['criteria']
    capable=full and all(summary['BASE'][w][cat]['functional']>=cr['per_wording_BASE_functional_min'] for w in WORDINGS for cat in ['E','N'])
    match=full and all(matching[w]['E_accept']>=3 and matching[w]['N_reject']==4 and not matching[w]['uncovered_or_missing'] for w in WORDINGS)
    def behavior(c):
        return (full and capable and engineering and
            all(summary[c][w]['E']['joint']>=3 and summary[c][w]['E']['net_joint']>=2 for w in WORDINGS) and
            all(not summary[c]['ALL'][cat]['regression'] for cat in ['E','N']) and
            summary[c]['ALL']['E']['actual_bank_tasks']==8 and summary[c]['ALL']['E']['covered']==8)
    manual=behavior('MANUAL_V');auto=behavior('MODULE_V') and manual and match and summary['MODULE_V']['ALL']['N']['actual_bank_tasks']==0
    status=(failure['status'] if failure else 'INCOMPLETE' if not full else 'STOP_ENGINEERING' if not engineering else
        'CAPABILITY_LIMITED' if not capable else
        'INDEPENDENT_SIMPLE_WORDING_LOOP_SUPPORTED' if auto else 'INDEPENDENT_SIMPLE_WORDING_LOOP_NOT_CONFIRMED')
    decision=dict(status=status,engineering_complete=bool(engineering),complete_matrix=full,
        simple_baseline_stable=bool(capable),matcher_stable=bool(match and engineering),manual_V_stable=bool(manual),
        module_V_stable=bool(auto),statistical_units=4,independent_confirmation=True,threshold_changed=False,
        bank_compiled=False,module_trained=False,all_historical_statuses_preserved=True)
    return summary,matching,decision

def main(args):
    verify_package();run=Path(args.run);dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    for n,h in read(run/'manifest.json')['files'].items():assert sha(run/n)==h,n
    cfg=read(ROOT/'data/config.json');tasks=read(ROOT/'data/evaluation.json');td={t['id']:t for t in tasks};rows=[]
    for p in sorted(run.glob('*/*/generation.json')):
        g=read(p);assert g['id'] in td and g['condition'] in CONDITIONS;t=td[g['id']]
        s=score(g['text'],t,g['stop']=='EOS')
        rows.append(dict(id=t['id'],family=t['family'],wording=t['wording'],category=t['category'],condition=g['condition'],score=s,generation=g,
          valid_decision_prefix=valid_decision_prefix(g,t),source=str(p)))
    write(dest/'SCORED_RESULTS.json',rows)
    eng=read(run/'DECISION.json') if (run/'DECISION.json').exists() else {}
    failure=read(run/'FAILURE.json') if (run/'FAILURE.json').exists() else None
    changed=[p for p,h in read(ROOT/'data/source_sha.json').items() if sha(p)!=h]
    if changed:failure=dict(status='STOP_ENGINEERING',reason='Historical SHA changed',files=changed)
    summary,matching,decision=summarize(tasks,rows,cfg,eng.get('status')=='ENGINEERING_COMPLETE_PENDING_EXECUTION_SCORING' and not changed,failure)
    for n,v in [('SUMMARY.json',summary),('MATCHING.json',matching),('DECISION.json',decision)]:write(dest/n,v)
    idx={(r['id'],r['condition']):r for r in rows};pairs=[]
    for family in sorted({t['family'] for t in tasks}):
        for cat in ['E','N']:
            a,b=[next(t for t in tasks if t['family']==family and t['category']==cat and t['wording']==w) for w in WORDINGS]
            for c in CONDITIONS:
                x,y=idx.get((a['id'],c)),idx.get((b['id'],c))
                pairs.append(dict(family=family,category=cat,condition=c,usage_id=a['id'],interface_id=b['id'],complete=bool(x and y),
                    usage=None if not x else dict(functional=x['score']['functional'],joint=x['score']['joint'],judgment=x['generation'].get('judgment')),
                    interface=None if not y else dict(functional=y['score']['functional'],joint=y['score']['joint'],judgment=y['generation'].get('judgment'))))
    write(dest/'WORDING_PAIRS.json',pairs)
    effects=[];numerics=[]
    for t in tasks:
        base=idx.get((t['id'],'BASE'));manual=idx.get((t['id'],'MANUAL_V'))
        if base and manual and base['generation'].get('gate'):
            bg,mg=base['generation'],manual['generation'];j=bg['gate']['prediction_step']
            same=bg['input_ids']==mg['input_ids'] and bg['output_ids'][:j]==mg['output_ids'][:j]
            item=dict(id=t['id'],same_pre_gate_history=same,valid_decision_prefix=base['valid_decision_prefix'],step=j,
                manual_bank_calls=mg['bank_calls'],first_tokens=dict(BASE=bg['output_ids'][j],MANUAL_V=mg['output_ids'][j]))
            if same:
                measures={}
                for c in ['BASE','MANUAL_V']:
                    with np.load(run/t['id']/c/'logits.npz') as z:x=z['logits'][j].astype('float64')
                    measures[c]=dict(fro_rank=int(np.sum(x>x[12799])+1),top1_id=int(x.argmax()),
                        fro_set=float(x[12799]-x[738]),fro_list=float(x[12799]-x[1140]))
                item['first_token_logits']=measures
                item['delta_fro_set']=measures['MANUAL_V']['fro_set']-measures['BASE']['fro_set']
                item['delta_fro_list']=measures['MANUAL_V']['fro_list']-measures['BASE']['fro_list']
            att=run/t['id']/'MANUAL_V/steps'/f'{j:03d}_attention.json'
            if att.exists():item['all_layer_head_attention']=read(att)['records']
            effects.append(item)
    for r in rows:
        steps=r['generation'].get('steps',[])
        numerics.append(dict(id=r['id'],condition=r['condition'],active_predictions=r['generation'].get('active_prediction_steps',0),
            max_logits_rms=max([s['numerical']['rms'] for s in steps if 'numerical' in s] or [0]),
            max_selected_logp_error=max([s['selected_logp_error'] for s in steps if 'selected_logp_error' in s] or [0])))
    write(dest/'DECISION_EFFECTS.json',dict(scope='FIRST_TOKEN_ONLY_NOT_FULL_EXPRESSION_PROBABILITY',rows=effects))
    write(dest/'NUMERICAL_SUMMARY.json',numerics)
    write(dest/'COST.json',read(run/'COST.json') if (run/'COST.json').exists() else dict(status='NOT_AVAILABLE_INCOMPLETE'))
    lines=['# Cloud06 新简单计算族与未见措辞独立确认','',f"状态：`{decision['status']}`。",'',
       '|措辞|条件|E JOINT|E功能|N合同|E/N新增退化|未运行|','|---|---|---:|---:|---:|---:|---:|']
    for w in WORDINGS:
        for c in CONDITIONS:
            e,n=summary[c][w]['E'],summary[c][w]['N']
            lines.append(f"|{w}|{c}|{e['joint']}/4|{e['functional']}/4|{n['functional']}/4|{len(e['regression'])}/{len(n['regression'])}|{e['not_run']+n['not_run']}|")
    lines+=['','4个新计算族，两版未见措辞为重复测量，不计作8或16个独立族。只支持当前小样本确认范围。未运行不是零成功。',
       'BASE能力、判断拒绝、人工V及AUTO分别判定；不得把自动组失败全部归因于判断，文字代码参照不等于选择题老师。',
       '实际长度在TOKEN_AUDIT中冻结；文字与长度/位置未独立分离。合法set子类采用Cloud05-Rescore01修正评分，历史状态不变。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(dest/'FAILURES.json',[dict(id=r['id'],condition=r['condition'],reasons=r['score']['failure_reasons'],coverage=r['generation']['coverage']) for r in rows if not r['score']['functional'] or not r['generation']['coverage']])
    write(dest/'audit.json',dict(changed=changed,remote_engineering=eng,bank_sha=sha(ROOT/'data/bank.npz'),judgment_sha=sha(ROOT/'data/judgment.npz')))
    seal_output(dest);print(decision['status'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);main(p.parse_args())
