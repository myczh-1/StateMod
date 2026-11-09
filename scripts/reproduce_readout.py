"""Recompute frozen outputs from bundled states. No training or model inference."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from statemod.readout import chain,factorial

def read(p):return json.loads((ROOT/p).read_text())
def main():
 with np.load(ROOT/'artifacts/judgment_14b.npz',allow_pickle=False) as f:w={k:f[k] for k in f.files}
 with np.load(ROOT/'artifacts/cloud11_states.npz',allow_pickle=False) as f:
  ids=f['task_ids'].tolist();raw=f['hidden'];assert f['valid'].all()
 a=chain(raw,w);pred=read('experiments/cloud11-attempt02/evaluation/PREDICTIONS.json')
 by={x['task_id']:x for x in pred};errors=[]
 for i,key in enumerate(ids):
  q=by[key];errors.append(abs(float(a['probability'][i])-q['prob']))
  assert np.isclose(a['logits'][i],q['logit'],atol=1e-10,rtol=0)
  assert (a['probability'][i]>.5)==(q['action']=='frozenset')
 counts={s:sum(x['teacher_agreement'] for x in pred if x['split']==s) for s in ['train','reference']}
 assert counts=={'train':8,'reference':7},counts
 with np.load(ROOT/'artifacts/cloud12_layer_values.npz',allow_pickle=False) as f:
  ids=f['ids'].tolist();b=chain(f['raw'],w)
  for key,arr in b.items():assert np.allclose(arr,f[key],atol=1e-10,rtol=0),key
 rows=read('experiments/cloud12/evaluation/SCORES.json');by={x['id']:x for x in rows}
 rebuilt=[]
 for i,key in enumerate(ids):
  r=by[key].copy();assert abs(float(b['probability'][i])-r['probability'])<1e-10
  assert (b['probability'][i]>.5)==r['enable']
  r['logit']=float(b['logits'][i]);rebuilt.append(r)
 effects=factorial(rebuilt);old=read('experiments/cloud12/evaluation/FACTORIAL_EFFECTS.json')
 assert len(old)==len(effects)
 for a,c in zip(effects,old):
  for k in ['wording_bare','wording_fenced','format_train','format_heldout','interaction']:
   assert abs(a[k]-c[k])<1e-10,(k,a[k],c[k])
 out=dict(status='ARCHIVED_READOUT_REPRODUCED',new_training=False,new_model_inference=False,cloud11_agreement=counts,
          cloud11_max_probability_error=max(errors),cloud12_scores=[{k:r[k] for k in ['id','logit','probability','enable']} for r in rebuilt],factorial_effects=effects)
 p=ROOT/'reproduction-output';p.mkdir(exist_ok=True);(p/'readout.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
