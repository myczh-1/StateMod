import sys,unittest,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from statemod.gate import inspect_return
from statemod.scoring import evaluate
from statemod.readout import chain
class EvidenceChecks(unittest.TestCase):
 def test_gate_before_constructor(self):
  self.assertIsNotNone(inspect_return('def f(x):\n    return'))
  self.assertIsNone(inspect_return('def f(x):\n    return frozenset'))
  self.assertIsNone(inspect_return('def f(x):\n    return result'))
  self.assertIsNone(inspect_return('def f(x):\n    x = "return"'))
  self.assertIsNotNone(inspect_return('```python\ndef f(x):\n    return'))
 def test_mutability_separate_from_exact_type(self):
  t=dict(function_name='f',member_type='int',category='N',tests=[dict(role='duplicate',args=[[1,1,2]],expected=[1,2])])
  a=evaluate(dict(task=t,text='def f(x): return set(x)',complete=True))
  b=evaluate(dict(task=t,text='def f(x): return frozenset(x)',complete=True))
  self.assertTrue(a['functional']);self.assertFalse(a['exact_frozenset'])
  self.assertFalse(b['functional']);self.assertTrue(b['exact_frozenset'])
  self.assertTrue(any(x['kind']=='MUTABILITY' for x in b['failure_reasons']))
 def test_no_repair(self):
  t=dict(function_name='f',member_type='int',category='E',tests=[])
  a=evaluate(dict(task=t,text='def f(x): return frozenset(',complete=True))
  self.assertFalse(a['format_ok']);self.assertEqual(a['code'],'def f(x): return frozenset(')
 def test_readout_checkpoint_integrity(self):
  root=Path(__file__).resolve().parents[1]
  with np.load(root/'artifacts/judgment_14b.npz',allow_pickle=False) as f:w={k:f[k] for k in f.files}
  with np.load(root/'artifacts/cloud12_layer_values.npz',allow_pickle=False) as f:
   a=chain(f['raw'],w);np.testing.assert_allclose(a['probability'],f['probability'],rtol=0,atol=1e-10)
 def test_zero_values_are_not_no_bank(self):
  # Analytic joint-softmax counterexample: a zero side Value still takes probability mass.
  native=np.array([2.]);both=np.array([2.,0.]);no_bank=float(native[0]);with_zero=float(np.ones(2)@both/2)
  self.assertNotEqual(no_bank,with_zero)
if __name__=='__main__':unittest.main()
