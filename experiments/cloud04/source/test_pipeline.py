"""Synthetic CPU checks only. Fixtures are not confirmation outcomes."""
import tempfile,unittest,sys,tarfile,io,copy,ast
from pathlib import Path
import numpy as np,torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from common import read
from run import task,text_task
from evaluate import summarize
from scoring import score
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from launch_collect_shutdown import safe_extract

class PipelineTests(unittest.TestCase):
 def test_same_checkpoint_routes_including_false_accept_and_missing_gate(self):
  torch.set_num_threads(1);torch.manual_seed(15)
  config=Qwen3Config(vocab_size=67,hidden_size=32,intermediate_size=48,num_hidden_layers=28,num_attention_heads=4,num_key_value_heads=1,head_dim=8)
  config._attn_implementation='eager';m=Qwen3ForCausalLM(config).eval();m.requires_grad_(False)
  class Tok:
   def apply_chat_template(self,*a,**kw):return [2,3,4]
   def decode(self,ids,skip_special_tokens=True):
    if not skip_special_tokens:return ','.join(map(str,ids))
    return 'def f(x): return' if len(ids)==1 else 'def f(x): return 0'
   def encode(self,text,**kw):return [int(x) for x in text.split(' ')[0].split(',')]+[6]
  class Judge:
   def __init__(self,en):self.en=en
   def predict(self,h):return dict(p=.9 if self.en else .1,enabled=self.en,threshold=.5)
  rng=np.random.default_rng(4);payload={k:rng.normal(size=(28,1,36,8)).astype('float32')*.1 for k in ['K','V']}
  cfg=dict(max_new_tokens=4,layers=list(range(14,28)))
  t=dict(id='synthetic',messages=[],input_ids=[2,3,4],text_messages=[],text_input_ids=[2,3,4])
  for manual,auto in [(True,True),(False,False),(False,True),(True,False)]:
   with self.subTest(manual=manual,auto=auto),tempfile.TemporaryDirectory() as d:
    rows=task(m,Tok(),t,manual,Judge(auto),payload,cfg,Path(d),[])
    self.assertEqual([r['bank_calls'] for r in rows],[0,14*manual,14*auto])
    reference=1 if manual==auto else 0
    if not auto or manual==auto:self.assertEqual(rows[2]['output_ids'],rows[reference]['output_ids'])
    text=text_task(m,Tok(),t,cfg,Path(d),[])
    self.assertEqual(text['bank_calls'],0)
  class NoGate(Tok):
   def decode(self,ids,skip_special_tokens=True):return 'def f(x): pass' if skip_special_tokens else super().decode(ids,False)
  with tempfile.TemporaryDirectory() as d:
   rows=task(m,NoGate(),t,True,Judge(True),payload,cfg,Path(d),[])
   self.assertTrue(all(not r['coverage'] and r['bank_calls']==0 for r in rows))

 def test_fixed_criteria_do_not_pool_wordings_or_drop_failures(self):
  tasks=read(R/'TASKS.json');cfg=read(R/'data/config.json');rows=[]
  for t in tasks:
   for c in cfg['conditions']:
    enabled=t['category']=='E' and c in ['MANUAL_V','MODULE_V']
    rows.append(dict(id=t['id'],condition=c,score=dict(functional=True,joint=t['category']=='N' or c!='BASE',exact_frozenset=t['category']=='E' and c!='BASE'),
     generation=dict(coverage=True,enabled=enabled,bank_calls=14 if enabled else 0,judgment=dict(p=.9 if enabled else .1,enabled=enabled))))
  def status(rs,eng=True,fail=None):return summarize(tasks,rs,cfg,eng,fail)[2]
  self.assertEqual(status(rows)['status'],'INDEPENDENT_8B_MODULE_V_LOOP_SUPPORTED')
  self.assertEqual(status(rows[:-1])['status'],'INCOMPLETE')
  bad=copy.deepcopy(rows)
  for t in [t for t in tasks if t['category']=='E' and t['wording']=='W4'][:2]:
   r=next(r for r in bad if r['id']==t['id'] and r['condition']=='MODULE_V');r['generation']['judgment'].update(enabled=False,p=.1)
  self.assertFalse(status(bad)['matcher_supported'])
  self.assertEqual(status(bad)['status'],'INDEPENDENT_8B_MODULE_V_LOOP_NOT_CONFIRMED')
  bad=copy.deepcopy(rows);n=next(r for r in bad if '_N_W3' in r['id'] and r['condition']=='MODULE_V');n['generation']['coverage']=False
  self.assertFalse(status(bad)['matcher_supported'])
  bad=copy.deepcopy(rows);n=next(r for r in bad if '_N_W4' in r['id'] and r['condition']=='MODULE_V');n['score']['functional']=False
  self.assertFalse(status(bad)['module_V_loop_supported'])
  bad=copy.deepcopy(rows)
  for r in bad:
   if r['condition']=='BASE' and '_N_W4' in r['id']:r['score']['functional']=False
  self.assertEqual(status(bad)['status'],'CAPABILITY_OR_COVERAGE_LIMITED')
  self.assertEqual(status(rows,False)['status'],'STOP_ENGINEERING')
  self.assertEqual(status(rows,False,dict(status='STOP_NUMERICAL'))['status'],'STOP_NUMERICAL')

 def test_gold_and_mutability_contracts(self):
  for t in read(R/'TASKS.json'):
   with self.subTest(id=t['id']):
    s=score(t['gold'],t);self.assertTrue(s['functional']);self.assertTrue(s['joint'])
    if t['category']=='E':
     alternate=score(t['gold'].replace('frozenset(', 'set('),t)
     self.assertTrue(alternate['functional']);self.assertFalse(alternate['joint'])
    else:
     alternate=score(t['gold'].replace('return set(', 'return frozenset('),t)
     self.assertFalse(alternate['functional']);self.assertTrue(alternate['values_correct'])
     self.assertFalse(alternate['mutability_correct'])

 def test_archive_and_source_boundaries(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'bad.tar'
   with tarfile.open(p,'w') as tar:
    m=tarfile.TarInfo('../escape');m.size=1;tar.addfile(m,io.BytesIO(b'x'))
   with self.assertRaises(ValueError):safe_extract(p,Path(d)/'out')
  # Test the exact detached remote launcher locally with a harmless stub, never SSH.
  import subprocess,time
  tree=ast.parse((R/'tools/launch_collect_shutdown.py').read_text())
  launch=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='launch' for t in n.targets))
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'tools').mkdir()
   (root/'tools/remote_job.py').write_text('from pathlib import Path; import time; time.sleep(.3); Path("stub_complete").write_text("ok")')
   reply=subprocess.run([sys.executable,'-B','-c',launch,d],capture_output=True,text=True,timeout=3)
   self.assertEqual(reply.returncode,0,reply.stderr);self.assertTrue(reply.stdout.strip().isdigit())
   deadline=time.monotonic()+3
   while not (root/'stub_complete').exists() and time.monotonic()<deadline:time.sleep(.05)
   self.assertTrue((root/'stub_complete').exists())
   second=subprocess.run([sys.executable,'-B','-c',launch,d],capture_output=True,text=True,timeout=3)
   self.assertNotEqual(second.returncode,0)  # launch claim prevents replay after unknown SSH result
  for name in ['common.py','backend.py','gate.py','readout.py','scoring.py']:
   self.assertEqual((R/'src'/name).read_bytes(),(R.parent/'v-conditional-cloud03/src'/name).read_bytes())
  self.assertFalse((R/'src/compile_bank.py').exists())
  for folder in ['src','tools']:
   for p in (R/folder).glob('*.py'):ast.parse(p.read_text())

if __name__=='__main__':unittest.main()
