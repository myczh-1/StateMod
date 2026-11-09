"""Offline scoring after retrieval/shutdown; no model, parameter fitting or task selection."""
import argparse
import numpy as np
from pathlib import Path
from common import *
from scoring import score
CONDITIONS=['BASE','TEXT_CODE','MANUAL_V','MODULE_V']

def summarize(tasks,rows,cfg,engineering,failure=None):
    indexed={(r['id'],r['condition']):r for r in rows}
    assert len(indexed)==len(rows),'duplicate result'
    summary={}
    for c in CONDITIONS:
        summary[c]={}
        for w in ['ALL','W3','W4']:
            summary[c][w]={}
            for category in ['E','N']:
                planned=[t for t in tasks if t['category']==category and (w=='ALL' or t['wording']==w)]
                cc=[indexed[t['id'],c] for t in planned if (t['id'],c) in indexed]
                reg=[];gain=[];loss=[]
                for r in cc:
                    b=indexed.get((r['id'],'BASE'))
                    if b:
                        if b['score']['functional'] and not r['score']['functional']:reg.append(r['id'])
                        if not b['score']['joint'] and r['score']['joint']:gain.append(r['id'])
                        if b['score']['joint'] and not r['score']['joint']:loss.append(r['id'])
                summary[c][w][category]=dict(planned=len(planned),completed=len(cc),not_run=len(planned)-len(cc),
                    functional=sum(bool(r['score']['functional']) for r in cc),joint=sum(bool(r['score']['joint']) for r in cc),
                    exact_frozenset=sum(bool(r['score']['exact_frozenset']) for r in cc),regression=reg,new_joint=gain,
                    lost_joint=loss,net_joint=len(gain)-len(loss),covered=sum(bool(r['generation']['coverage']) for r in cc),
                    actual_bank_tasks=sum(r['generation']['bank_calls']>0 for r in cc))
    matching={}
    for w in ['W3','W4']:
        ww=[t for t in tasks if t['wording']==w];e=[];n=[];missing=[];margins=[]
        for t in ww:
            r=indexed.get((t['id'],'MODULE_V'))
            if not r or not r['generation']['coverage'] or r['generation'].get('judgment') is None:
                missing.append(t['id']);continue
            j=r['generation']['judgment'];margins.append(dict(id=t['id'],p=j['p'],threshold_margin=j['p']-.5))
            if t['category']=='E' and j['enabled']:e.append(t['id'])
            if t['category']=='N' and not j['enabled']:n.append(t['id'])
        matching[w]=dict(E_accept=len(e),N_reject=len(n),denominator_each=8,uncovered_or_missing=missing,margins=margins)
    full=len(indexed)==len(tasks)*len(CONDITIONS)
    cr=cfg['criteria'];families=sorted({t['family'] for t in tasks})
    capable=all(summary['BASE'][w][cat]['functional']>=cr['per_wording_BASE_functional_min'] for w in ['W3','W4'] for cat in ['E','N'])
    match=full and all(matching[w]['E_accept']>=cr['per_wording_E_accept_min'] and matching[w]['N_reject']>=cr['per_wording_N_reject_min'] for w in matching)
    both={}
    for c in ['MANUAL_V','MODULE_V']:
        both[c]=[f for f in families if all((t['id'],c) in indexed and indexed[t['id'],c]['score']['joint'] for t in tasks if t['family']==f and t['category']=='E')]
    def behavior(c):
        return (full and capable and engineering and len(both[c])>=cr['both_wordings_joint_family_min'] and
          all(summary[c][w]['E']['joint']>=cr['per_wording_E_joint_min'] for w in ['W3','W4']) and
          summary[c]['ALL']['E']['net_joint']>=cr['E_net_joint_gain_min'] and
          all(len(summary[c]['ALL'][cat]['regression'])<=cr['max_function_regression'] for cat in ['E','N']))
    manual=behavior('MANUAL_V')
    auto=behavior('MODULE_V') and manual and match and summary['MODULE_V']['ALL']['N']['actual_bank_tasks']==0
    status=(failure['status'] if failure else 'INCOMPLETE' if not full else 'STOP_ENGINEERING' if not engineering else
        'CAPABILITY_OR_COVERAGE_LIMITED' if not capable else 'INDEPENDENT_8B_MODULE_V_LOOP_SUPPORTED' if auto else
        'INDEPENDENT_8B_MODULE_V_LOOP_NOT_CONFIRMED')
    decision=dict(status=status,engineering_complete=engineering,complete_matrix=full,capability_coverage=capable,
      matcher_supported=bool(match and engineering),manual_V_supported=bool(manual),module_V_loop_supported=bool(auto),
      both_wordings_E_joint_families=both,statistical_units=8,independent_confirmation=True,
      threshold_changed=False,bank_compiled=False,judge_trained=False,old_results_unchanged=True)
    # Cluster resampling: both wordings stay within the family. No task/token pseudoreplication.
    bootstrap={}
    if full:
        rng=np.random.default_rng(cfg['bootstrap_seed']);ix=rng.integers(0,len(families),(cfg['bootstrap_repetitions'],len(families)))
        for c in ['TEXT_CODE','MANUAL_V','MODULE_V']:
            for category,metric in [('E','joint'),('N','functional')]:
                vector=[]
                for f in families:
                    tt=[t for t in tasks if t['family']==f and t['category']==category]
                    vector.append(float(np.mean([int(indexed[t['id'],c]['score'][metric])-int(indexed[t['id'],'BASE']['score'][metric]) for t in tt])))
                samples=np.asarray(vector)[ix].mean(axis=1)
                bootstrap[c+'/'+category+'/'+metric]=dict(family_values=vector,mean=float(np.mean(vector)),
                    interval95=np.quantile(samples,[.025,.975]).tolist(),unit='family with both wordings',descriptive_only=True)
    return summary,matching,decision,bootstrap

def main(args):
    verify_package();run=Path(args.run);dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    for n,h in read(run/'manifest.json')['files'].items():assert sha(run/n)==h,n
    cfg=read(ROOT/'data/config.json');tasks=read(ROOT/'data/evaluation.json');td={t['id']:t for t in tasks};rows=[]
    for p in sorted(run.glob('*/*/generation.json')):
        r=read(p);assert r['id'] in td and r['condition'] in CONDITIONS
        t=td[r['id']];s=score(r['text'],t,r['stop']=='EOS')
        rows.append(dict(id=t['id'],family=t['family'],wording=t['wording'],category=t['category'],
          condition=r['condition'],score=s,generation=r,source=str(p)))
    write(dest/'SCORED_RESULTS.json',rows)
    eng=read(run/'DECISION.json') if (run/'DECISION.json').exists() else {}
    fail=read(run/'FAILURE.json') if (run/'FAILURE.json').exists() else None
    changed=[p for p,h in read(ROOT/'data/source_sha.json').items() if sha(p)!=h]
    if changed:fail=dict(status='STOP_ENGINEERING',reason='Protected historical files changed',files=changed)
    summary,matching,decision,bootstrap=summarize(tasks,rows,cfg,eng.get('status')=='ENGINEERING_COMPLETE_PENDING_EXECUTION_SCORING' and not changed,fail)
    for name,obj in [('SUMMARY.json',summary),('MATCHING.json',matching),('DECISION.json',decision),('FAMILY_BOOTSTRAP.json',bootstrap)]:write(dest/name,obj)
    idx={(r['id'],r['condition']):r for r in rows};pairs=[]
    for t in tasks:
        a=idx.get((t['id'],'MANUAL_V'));b=idx.get((t['id'],'MODULE_V'));base=idx.get((t['id'],'BASE'))
        if not (a and b and base):continue
        def divergence(r):
            x,y=r['generation']['output_ids'],base['generation']['output_ids']
            return next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),min(len(x),len(y)) if len(x)!=len(y) else None)
        pairs.append(dict(id=t['id'],family=t['family'],wording=t['wording'],
          routes_equal=a['generation']['enabled']==b['generation']['enabled'],
          tokens_equal=a['generation']['output_ids']==b['generation']['output_ids'],
          manual_minus_auto_joint=int(a['score']['joint'])-int(b['score']['joint']),
          manual_first_divergence=divergence(a),auto_first_divergence=divergence(b)))
    write(dest/'PAIRED.json',pairs)
    lines=['# Cloud04 独立确认','',f"状态：`{decision['status']}`。",'',
        '|措辞|条件|E JOINT/计划|E功能/计划|N功能/计划|E/N新增功能退化|未运行|','|---|---|---:|---:|---:|---:|---:|']
    for w in ['W3','W4']:
        for c in CONDITIONS:
            e,n=summary[c][w]['E'],summary[c][w]['N']
            lines.append(f"|{w}|{c}|{e['joint']}/8|{e['functional']}/8|{n['functional']}/8|{len(e['regression'])}/{len(n['regression'])}|{e['not_run']+n['not_run']}|")
    lines+=['','固定分母用于门槛；NOT_RUN单列，缺失并非观测到的失败或零成功。全部逐例代码、AST、合同测试与失败原因见SCORED_RESULTS。',
        '',f"匹配通过：{decision['matcher_supported']}；MANUAL通过：{decision['manual_V_supported']}；AUTO闭环通过：{decision['module_V_loop_supported']}。",'',
        '8个计算族为统计单位；W3/W4是重复测量。TEXT_CODE为本轮新生成，不复用旧教师成绩。未针对确认数据选择阈值、Bank、窗口、措辞或任务。',
        '人工调度只作V上界；AUTO只读取无规则自然生成的Block13隐藏态。匹配失败不作人工纠正。未到Gate不算正确拒绝。',
        '固定输出格式沿用旧模板；全新的是计算合同及两套适用性表达。历史近邻和共享原语见SEMANTIC_REVIEW。',
        '本轮无随机/错误Bank对照，不单独证明载荷语义特异性；结论仅限当前8B、此规则与新任务，不外推任意模板或总体误触发率。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
    failures=[dict(id=r['id'],condition=r['condition'],reasons=r['score']['failure_reasons'],coverage=r['generation']['coverage']) for r in rows if not r['score']['functional'] or not r['generation']['coverage']]
    write(dest/'FAILURES.json',failures)
    write(dest/'audit.json',dict(changed=changed,freeze_verified=True,remote_engineering=eng or fail,
       judgment_sha=sha(ROOT/'data/judgment.npz'),bank_sha=sha(ROOT/'data/bank.npz')))
    seal_output(dest);print(decision['status'])
    assert not changed

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);main(p.parse_args())
