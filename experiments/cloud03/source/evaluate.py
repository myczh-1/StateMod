"""Local post-shutdown scoring; function tests never enter module or payload compilation."""
import argparse,time
from pathlib import Path
from common import *
from scoring import score

def main(args):
    verify_package();run=Path(args.run);dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    for n,h in read(run/'manifest.json')['files'].items():assert sha(run/n)==h,n
    cfg=read(ROOT/'data/config.json');tasks={t['id']:t for t in read(ROOT/'data/evaluation.json')}
    results=[]
    # Retain partial generations if execution stopped; never count missing as zero successes.
    for p in sorted(run.glob('*/[BM]*/generation.json')):
        r=read(p);t=tasks[r['id']]
        s=score(r['text'],t,r['stop']=='EOS')
        results.append(dict(id=t['id'],category=t['category'],condition=r['condition'],source=str(p),
                            score=s,generation=r,archived=False))
    for r in read(ROOT/'data/text_reference.json'):
        t=tasks[r['source_id']];complete=r.get('stop')=='EOS'
        # Archive uses stop=EOS; reject unknown record schema instead of guessing.
        assert r['stop'] in ['EOS','TRUNCATED']
        results.append(dict(id=t['id'],category=t['category'],condition='TEXT_CODE',archived=True,
                            source='Cloud01 frozen TEXT_CODE',score=score(r['text'],t,complete),generation=r))
    write(dest/'SCORED_RESULTS.json',results)
    indexed={(r['id'],r['condition']):r for r in results};summary={}
    for condition in ['BASE','TEXT_CODE','MANUAL_V','MODULE_V']:
        rr=[r for r in results if r['condition']==condition];data={}
        for category in ['E','N']:
            cc=[r for r in rr if r['category']==category];reg=[];gain=[]
            for r in cc:
                baseline=indexed.get((r['id'],'BASE'))
                if baseline:
                    if baseline['score']['functional'] and not r['score']['functional']:reg.append(r['id'])
                    if not baseline['score']['joint'] and r['score']['joint']:gain.append(r['id'])
            data[category]=dict(planned=8,completed=len(cc),not_run=8-len(cc),functional=sum(r['score']['functional'] for r in cc),
                exact_frozenset=sum(bool(r['score']['exact_frozenset']) for r in cc),joint=sum(r['score']['joint'] for r in cc),
                regression=reg,new_joint=gain,actual_bank_tasks=sum(r['generation'].get('bank_calls',0)>0 for r in cc))
        summary[condition]=data
    write(dest/'SUMMARY.json',summary)
    pairs=[]
    for tid in tasks:
        a=indexed.get((tid,'MANUAL_V'));b=indexed.get((tid,'MODULE_V'));base=indexed.get((tid,'BASE'))
        if a and b and base:
            x=a['generation'];y=b['generation'];tokens0=base['generation']['output_ids']
            def divergence(seq):return next((i for i,(a,b) in enumerate(zip(seq,tokens0)) if a!=b),min(len(seq),len(tokens0)) if len(seq)!=len(tokens0) else None)
            pairs.append(dict(id=tid,route_agree=x['enabled']==y['enabled'],tokens_agree=x['output_ids']==y['output_ids'],
                manual_first_divergence=divergence(x['output_ids']),module_first_divergence=divergence(y['output_ids']),
                coverage=y['coverage'],module_bank_calls=y['bank_calls'],module_enabled=y['enabled']))
    write(dest/'PAIRED.json',pairs)
    full=len(pairs)==16 and all(summary[k][c]['completed']==8 for k in summary for c in ['E','N'])
    engineering=(run/'DECISION.json').exists() and read(run/'DECISION.json')['status']=='ENGINEERING_COMPLETE_PENDING_EXECUTION_SCORING'
    manual=engineering and full and summary['MANUAL_V']['E']['joint']>=cfg['manual_min_E_joint'] and summary['MANUAL_V']['E']['joint']>summary['BASE']['E']['joint'] and not summary['MANUAL_V']['E']['regression'] and not summary['MANUAL_V']['N']['regression']
    auto=manual and all(p['route_agree'] and p['tokens_agree'] and p['coverage'] for p in pairs) and summary['MODULE_V']['E']['joint']>=cfg['module_min_E_joint'] and not summary['MODULE_V']['E']['regression'] and not summary['MODULE_V']['N']['regression'] and summary['MODULE_V']['N']['actual_bank_tasks']==0
    failure=read(run/'FAILURE.json') if (run/'FAILURE.json').exists() else None
    status=failure['status'] if failure else 'INCOMPLETE' if not full else 'DEVELOPMENT_8B_MODULE_V_LOOP_SUPPORTED' if auto else 'DEVELOPMENT_8B_MODULE_V_LOOP_NOT_ESTABLISHED'
    decision=dict(status=status,manual_V_effect_supported=bool(manual),module_V_loop_supported=bool(auto),
        independent_confirmation=False,old_results_unchanged=True,engineering_complete=engineering,
        no_parameter_search=True,judge_sha=sha(ROOT/'data/judgment.npz'))
    write(dest/'DECISION.json',decision)
    lines=['# Cloud03 开发闭环结果','',f"状态：`{status}`。",'',
      '|条件|E联合正确|E功能正确|N合同正确|E新增功能退化|N新增退化|', '|---|---:|---:|---:|---:|---:|']
    for k,v in summary.items():
        e,n=v['E'],v['N'];lines.append(f"|{k}|{e['joint']}/{e['completed']}|{e['functional']}/{e['completed']}|{n['functional']}/{n['completed']}|{len(e['regression'])}|{len(n['regression'])}|")
    lines+=['', '分母为实际执行数；缺失条件见SUMMARY，不计零成功。TEXT_CODE为同任务冻结的Cloud01参照，不重跑。', '',
      '判断、V执行、联合收益分别评价。MANUAL使用已知适用标签作为调度上界，MODULE仅看冻结隐藏态读出。原1.7B Bank未载入8B；本轮为8B新编译同规则载荷，不能继承旧Bank成功。', '',
      '全部为已有开发任务、仅从当前判断模块拟合中留出，不构成独立确认。无随机/错误载荷正式对照，因此不单独证明载荷语义特异性。', '',
      '当前规则、层位14–27、槽位36、权重1、单次窗口固定；失败不调整。完整原始代码、AST、执行错误与缺失见SCORED_RESULTS。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
    changed=[p for p,h in read(ROOT/'data/source_sha.json').items() if sha(p)!=h]
    write(dest/'audit.json',dict(changed=changed,freeze_verified=True,model_audit=read(run/'DECISION.json') if (run/'DECISION.json').exists() else failure))
    assert not changed
    seal_output(dest);print(status,summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);main(p.parse_args())
