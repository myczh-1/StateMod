"""Synthetic CPU and contract tests; never use a server or main model."""
import tempfile,unittest,sys,copy,ast,tarfile,io,json,hashlib
from pathlib import Path
import numpy as np,torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from common import read
from run import task,text_task
from evaluate import summarize,audit_anchors,valid_decision_prefix
from scoring import score
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from launch_collect_shutdown import safe_extract,verify_collection,LAUNCH_CODE

class PipelineTests(unittest.TestCase):
 def test_setup_shutdown_requires_idle_gpu(self):
  import launch_collect_shutdown as controller
  from unittest.mock import patch
  from types import SimpleNamespace
  for code in [0,1]:
   with tempfile.TemporaryDirectory() as d,patch.object(controller,'ssh',return_value=SimpleNamespace(returncode=code,stdout='',stderr='')),patch.object(controller,'shutdown') as close:
    controller.shutdown_setup_failure_if_idle(Path(d))
    self.assertEqual(close.call_count,int(code==0))
 def test_semantic_window_audit_does_not_move_gate(self):
  t={'function_name':'collect'}
  def valid(s):return valid_decision_prefix({'gate':{'generated_prefix':s}},t)
  self.assertTrue(valid('def collect(x): return'))
  self.assertTrue(valid('```python\ndef collect(x):\n    return'))
  self.assertFalse(valid('def helper(x): return'))
  self.assertFalse(valid('def collect(x):\n    result=set(x)\n    return'))

 def test_preflight_failure_archived_without_model_work(self):
  import remote_job
  from unittest.mock import patch,Mock
  for safe in [True,False]:
   with tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'data').mkdir();(p/'data/config.json').write_text(json.dumps(dict(emergency_shutdown_seconds=5400)))
    pre=dict(passed=False,safe_to_shutdown=safe,checks={'synthetic_failure':False})
    with patch.object(remote_job,'R',p),patch.object(remote_job,'verify_package'),patch.object(remote_job,'inspect',return_value=pre),patch.object(remote_job.subprocess,'Popen',return_value=Mock(pid=123)) as pop,patch.object(sys,'argv',['remote_job','--execute-after-user-start']):
     remote_job.main()
    self.assertEqual(pop.call_count,int(safe)) # watchdog only; no main model
    ready=json.loads((p/'READY_TO_COLLECT.json').read_text());self.assertEqual(ready['safe_to_shutdown'],safe)
    self.assertTrue((p/'results.tar').exists());self.assertFalse((p/'runs').exists())
    self.assertEqual(read(p/'JOB.json')['status'],'STOP_ENGINEERING')

 def test_collection_verification_precedes_shutdown_and_scoring(self):
  import launch_collect_shutdown as controller
  from unittest.mock import patch
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'data').mkdir();(p/'artifacts').mkdir();(p/'execution').mkdir();(p/'execution/LAUNCH_ATTEMPT.json').write_text('{}')
   (p/'artifacts/runtime.tar.gz').write_bytes(b'package')
   (p/'artifacts/package.json').write_text(json.dumps(dict(sha256=hashlib.sha256(b'package').hexdigest())))
   (p/'data/source_sha.json').write_text('{}');(p/'data/config.json').write_text(json.dumps(dict(minimum_local_free_gib=1,remote_output='/unused',run_timeout_seconds=1)))
   events=[]
   def download(*args):
    events.append('verified');(p/'execution/runs').mkdir();(p/'execution/runs/manifest.json').write_text('{}')
   def shutdown(*args):events.append('shutdown')
   def score_call(*args,**kw):events.append('local_score');return SimpleNamespace(returncode=0)
   with (patch.object(controller,'R',p),patch.object(controller,'verify_package'),patch.object(controller,'ssh',return_value=SimpleNamespace(returncode=0,stdout=json.dumps({'safe_to_shutdown':True}),stderr='')),
        patch.object(controller,'download',side_effect=download),patch.object(controller,'shutdown',side_effect=shutdown),patch.object(controller.subprocess,'run',side_effect=score_call),patch.object(sys,'argv',['controller','--execute-after-user-start','--collect-only'])):
    controller.main()
   self.assertEqual(events,['verified','shutdown','local_score'])
 def test_criteria_keep_wordings_and_missing_data_separate(self):
  tasks=read(R/'TASKS.json');cfg=read(R/'data/config.json');rows=[]
  for t in tasks:
   for c in cfg['conditions']:
    enabled=t['category']=='E' and c in ['MANUAL_V','MODULE_V']
    rows.append(dict(id=t['id'],condition=c,score=dict(functional=True,joint=t['category']=='N' or c!='BASE',exact_frozenset=t['category']=='E' and c!='BASE'),
     generation=dict(coverage=True,enabled=enabled,bank_calls=14 if enabled else 0,judgment=dict(p=.9 if enabled else .1,enabled=enabled))))
  anchor=dict(complete=True,passed=True)
  def decide(rs,eng=True,a=anchor,fail=None):return summarize(tasks,rs,cfg,eng,a,fail)[2]
  self.assertEqual(decide(rows)['status'],'DEVELOPMENT_SIMPLE_WORDING_LOOP_STABLE')
  self.assertEqual(decide(rows[:-1])['status'],'INCOMPLETE')
  self.assertEqual(decide(rows,a=dict(complete=True,passed=False))['status'],'ANCHOR_NOT_REPRODUCED')
  bad=copy.deepcopy(rows)
  r=next(r for r in bad if '_N_W_LIFECYCLE' in r['id'] and r['condition']=='MODULE_V');r['generation']['judgment'].update(enabled=True,p=.9);r['generation']['bank_calls']=14
  self.assertFalse(decide(bad)['matcher_stable']);self.assertFalse(decide(bad)['module_V_stable'])
  bad=copy.deepcopy(rows)
  for r in bad:
   if '_E_W_LIFECYCLE' in r['id'] and r['condition']=='MANUAL_V':r['score']['joint']=False
  self.assertFalse(decide(bad)['manual_V_stable'])
  bad=copy.deepcopy(rows)
  r=next(r for r in bad if r['condition']=='MANUAL_V');r['score']['functional']=False
  self.assertFalse(decide(bad)['manual_V_stable'])
  bad=copy.deepcopy(rows)
  for r in bad:
   if r['condition']=='BASE' and '_N_W_LIFECYCLE' in r['id']:r['score']['functional']=False
  self.assertEqual(decide(bad)['status'],'SIMPLE_BASELINE_NOT_STABLE')
  self.assertEqual(decide(rows,False)['status'],'STOP_ENGINEERING')
  self.assertEqual(decide(rows,False,fail=dict(status='STOP_NUMERICAL'))['status'],'STOP_NUMERICAL')

 def test_anchor_logit_audit_and_different_history(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'t/BASE').mkdir(parents=True);z=np.array([[1.,2.,3.],[3.,2.,1.]],dtype='float32')
   np.savez(p/'old.npz',logits=z);np.savez(p/'t/BASE/logits.npz',logits=z)
   ref=dict(id='t',condition='BASE',source_logits=str(p/'old.npz'),source_logits_sha=hashlib.sha256((p/'old.npz').read_bytes()).hexdigest(),input_ids=[4],output_ids=[2,0],functional=True,joint=False)
   row=dict(id='t',condition='BASE',generation=dict(input_ids=[4],output_ids=[2,0]),score=dict(functional=True,joint=False))
   self.assertTrue(audit_anchors(p,[row],[ref])['passed'])
   row['generation']['output_ids']=[1,0];a=audit_anchors(p,[row],[ref]);self.assertFalse(a['passed']);self.assertNotIn('max_step_logits_rms',a['rows'][0])
   self.assertFalse(audit_anchors(p,[],[ref])['complete'])

 def test_archive_guard_and_every_file_hash(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);bad=p/'bad.tar'
   with tarfile.open(bad,'w') as t:
    m=tarfile.TarInfo('../escape');m.size=1;t.addfile(m,io.BytesIO(b'x'))
   with self.assertRaises(ValueError):safe_extract(bad,p/'out')
   out=p/'ok';out.mkdir();(out/'JOB.json').write_text('{}')
   h=hashlib.sha256(b'{}').hexdigest();(out/'COLLECTION_MANIFEST.json').write_text(json.dumps({'files':{'JOB.json':h}}))
   (out/'results.tar').write_bytes(b'dummy');meta=dict(bytes=5,sha256=hashlib.sha256(b'dummy').hexdigest())
   self.assertEqual(verify_collection(out,meta)['files_verified'],1)
   (out/'JOB.json').write_text('corrupt')
   with self.assertRaises(AssertionError):verify_collection(out,meta)

 def test_launch_claim_no_duplicate_job(self):
  import subprocess,time
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'tools').mkdir();(p/'tools/remote_job.py').write_text('from pathlib import Path; Path("stub_done").write_text("ok")')
   a=subprocess.run([sys.executable,'-B','-c',LAUNCH_CODE,d],capture_output=True,text=True,timeout=3)
   self.assertEqual(a.returncode,0,a.stderr)
   deadline=time.monotonic()+3
   while not (p/'stub_done').exists() and time.monotonic()<deadline:time.sleep(.05)
   self.assertTrue((p/'stub_done').exists())
   b=subprocess.run([sys.executable,'-B','-c',LAUNCH_CODE,d],capture_output=True,text=True,timeout=3);self.assertNotEqual(b.returncode,0)

 def test_readonly_plan_and_core_identity(self):
  self.assertEqual((R/'TASKS.json').read_bytes(),(R.parent/'v-conditional-cloud05-plan/TASKS.json').read_bytes())
  self.assertEqual((R/'PROTOCOL.md').read_bytes(),(R.parent/'v-conditional-cloud05-plan/PROTOCOL.md').read_bytes())
  for n in ['common.py','backend.py','gate.py','readout.py','scoring.py']:
   self.assertEqual((R/'src'/n).read_bytes(),(R.parent/'v-conditional-cloud04/src'/n).read_bytes())
  self.assertFalse((R/'src/compile_bank.py').exists())
  for folder in ['src','tools']:
   for p in (R/folder).glob('*.py'):ast.parse(p.read_text())
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
