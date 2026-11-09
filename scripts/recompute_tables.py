"""Count archived scores only. Never executes generated code or changes scoring."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def main():
 output={}
 for name,where in [('03','evaluation'),('04','evaluation'),('05','evaluation'),('05-rescore01','results'),('06','evaluation')]:
  base=f'experiments/cloud{name}/{where}/';rows=read(base+'SCORED_RESULTS.json');summary=read(base+'SUMMARY.json')
  if isinstance(rows,dict):raise AssertionError('Unexpected archived record shape')
  counted={}
  for condition in summary:
   group=summary[condition];group=group.get('ALL',group)
   counted[condition]={}
   for category in ['E','N']:
    selected=[r for r in rows if r['condition']==condition and r['category']==category]
    for metric in ['functional','joint','exact_frozenset']:
     if metric not in group[category]:continue
     got=sum(bool(r.get('score',{}).get(metric,False)) for r in selected)
     assert got==group[category][metric],(name,condition,category,metric,got,group[category][metric])
    counted[condition][category]={k:group[category][k] for k in ['planned','completed','functional','joint']}
  output['cloud'+name]=counted
 # Freeze headline teacher/fidelity quantities against the archived primary summaries.
 for name,n in [('08',6),('10',8)]:
  s=read(f'experiments/cloud{name}/evaluation/SUMMARY.json');assert s['select']['correct']==n
  output['cloud'+name]=s
 p=ROOT/'reproduction-output';p.mkdir(exist_ok=True);(p/'tables.json').write_text(json.dumps(output,indent=2)+'\n')
 print('Archived score counts agree for Cloud03/04/05/05-rescore01/06; teacher headlines agree for Cloud08/10. No rescoring.')
if __name__=='__main__':main()
