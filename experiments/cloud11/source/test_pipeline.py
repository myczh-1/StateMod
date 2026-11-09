"""Synthetic collected-evidence integration; never runs the real model or SSH."""
import argparse,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import evaluate as ev
from common import ROOT,read,write,save,sha,logp,seal_output

class PipelineTests(unittest.TestCase):
    def fixture(self,root,failed=False):
        (root/'data').mkdir();run=root/'runs';run.mkdir()
        (root/'freeze.json').write_text('{"files":{}}')
        write(root/'data/source_sha.json',{})
        for name in ['config.json','TRAIN_TASKS.json','REFERENCE_TASKS.json','inputs.json','reference.json']:
            write(root/'data'/name,read(ROOT/'data'/name))
        write(run/'START.json',dict(freeze_sha=sha(root/'freeze.json')))
        if failed:
            write(run/'FAILURE.json',dict(status='STOP_ENGINEERING',reason='synthetic capture error'));seal_output(run);return run
        inputs=read(root/'data/inputs.json');tasks=read(root/'data/TRAIN_TASKS.json')+read(root/'data/REFERENCE_TASKS.json')
        ref={r['task_id']:r for r in read(root/'data/reference.json')};results=[];states=np.random.default_rng(19).normal(size=(16,5120)).astype('float32')
        k=0
        for rec in inputs:
            p=run/rec['id'];p.mkdir();task=next(t for t in tasks if t['id']==rec['task_id'])
            if rec['condition']=='BASE':
                reference_id=task['source_id'] if rec['split']=='train' else task['id']
                old=copy.deepcopy(ref[reference_id]['base']);old.update(id=rec['id'],task_id=task['id'],coverage=True,input_ids=rec['input_ids'])
                results.append(old);save(p/'decision.npz',hidden_cached=states[k]);k+=1
            else:
                natural=dict(id=rec['id'],task_id=task['id'],condition='TEXT_SELECT',stop='EOS',text=task['expected_choice'],input_ids=rec['input_ids'])
                results.append(natural);cs={}
                for name in ['frozenset','set']:
                    cp=p/name;cp.mkdir();ids=rec['candidates'][name]['ids'];total=0.
                    for j,token in enumerate(ids):
                        z=np.zeros(151936,dtype='float32');z[token]=15 if name==task['expected_choice'] else -15
                        total+=logp(z,token);save(cp/f'{j:03}.npz',cached=z)
                    cs[name]=dict(token_ids=ids,type_eos_logp=total)
                write(p/'candidates.json',cs)
        save(run/'states.npz',hidden=states,valid=np.ones(16,dtype=bool),task_ids=np.array([t['id'] for t in tasks]))
        write(run/'RESULTS.json',results);write(run/'INTEGRITY.json',dict(passed=True));write(run/'DECISION.json',dict(status='CAPTURE_COMPLETE'));seal_output(run)
        return run

    def test_capture_failure_never_trains(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=self.fixture(root,True);out=root/'evaluation'
            with patch.object(ev,'ROOT',root),patch.object(ev,'verify_package'),patch.object(ev,'fit') as fit:
                ev.main(argparse.Namespace(run=str(run),output=str(out)));fit.assert_not_called()
            self.assertFalse((out/'judgment_14b.npz').exists());self.assertEqual(read(out/'DECISION.json')['status'],'CAPTURE_FAILED')

    def test_baseline_errors_do_not_gate_fit_and_reference_cannot_change_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=self.fixture(root);out=root/'first'
            with patch.object(ev,'ROOT',root),patch.object(ev,'verify_package'):
                ev.main(argparse.Namespace(run=str(run),output=str(out)))
            self.assertTrue(read(out/'COST.json')['training_executed'])
            self.assertTrue(any(not s['result']['functional'] for s in read(out/'CODE_SCORES.json')))
            # Change only held-out hidden states. Fitted arrays must be identical.
            with np.load(run/'states.npz') as a:arrays={k:a[k].copy() for k in a.files}
            arrays['hidden'][8:]*=-3
            (run/'states.npz').unlink();save(run/'states.npz',**arrays)
            for i,t in enumerate(read(root/'data/REFERENCE_TASKS.json'),8):
                p=run/(t['id']+'__BASE')/'decision.npz';p.unlink();save(p,hidden_cached=arrays['hidden'][i])
            (run/'manifest.json').unlink();seal_output(run)
            with patch.object(ev,'ROOT',root),patch.object(ev,'verify_package'):
                ev.main(argparse.Namespace(run=str(run),output=str(root/'second')))
            with np.load(out/'judgment_14b.npz') as a,np.load(root/'second/judgment_14b.npz') as b:
                for k in a.files:self.assertTrue(np.array_equal(a[k],b[k]),k)

if __name__=='__main__':unittest.main()
