"""Synthetic runner, scoring and transport tests; no SSH/main-model effects."""
import argparse,sys,tempfile,unittest,json,tarfile,io
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from common import forward,read,write,seal_output
from backend import capture
from run import task
import evaluate
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from launch_collect_shutdown import safe_extract

class PipelineTests(unittest.TestCase):
    def test_three_branch_runner_same_checkpoint(self):
        torch.set_num_threads(1);torch.manual_seed(15)
        cfg=Qwen3Config(vocab_size=67,hidden_size=32,intermediate_size=48,num_hidden_layers=28,
                       num_attention_heads=4,num_key_value_heads=1,head_dim=8)
        cfg._attn_implementation='eager';m=Qwen3ForCausalLM(cfg).eval();m.requires_grad_(False)
        class Tok:
            def apply_chat_template(self,*a,**k):return [2,3,4]
            def decode(self,ids,skip_special_tokens=True):
                if not skip_special_tokens:return ','.join(map(str,ids))
                return 'def f(x): return' if len(ids)==1 else 'def f(x): return 0'
            def encode(self,text,**k):return [int(x) for x in text.split(' ')[0].split(',')]+[6]
        class Judge:
            def __init__(self,enabled): self.enabled=enabled
            def predict(self,h):return dict(p=.9 if self.enabled else .1,enabled=self.enabled,logit=2.197 if self.enabled else -2.197,threshold=.5)
        toks=[];zs=[];cache=None;h=None
        for j in range(4):
            if j==1:z,cache,h=capture(m,[toks[-1]],cache)
            else:z,cache=forward(m,[2,3,4] if j==0 else [toks[-1]],cache)
            toks.append(int(z.argmax()));zs.append(z)
        rng=np.random.default_rng(2);payload={k:rng.normal(size=(28,1,36,8)).astype('float32')*.1 for k in ['K','V']}
        ref=dict(hidden=h,p=.9,enabled=True,tokens=toks,logits=np.stack(zs))
        for enabled in [True,False]:
            with self.subTest(enabled=enabled),tempfile.TemporaryDirectory() as d:
                ref.update(p=.9 if enabled else .1,enabled=enabled)
                rows=task(m,Tok(),dict(id='synthetic',messages=[],input_ids=[2,3,4]),enabled,Judge(enabled),payload,
                          dict(max_new_tokens=4,layers=list(range(14,28))),ref,Path(d),[])
                self.assertEqual([r['bank_calls'] for r in rows],[0,14,14] if enabled else [0,0,0])
                self.assertEqual(rows[1]['output_ids'],rows[2]['output_ids'])
                self.assertEqual(rows[1]['final_cache_sha'],rows[2]['final_cache_sha'])
                if not enabled:self.assertEqual(rows[0]['final_cache_sha'],rows[2]['final_cache_sha'])
        class NoGateTok(Tok):
            def decode(self,ids,skip_special_tokens=True):
                return 'def f(x): pass' if skip_special_tokens else super().decode(ids,False)
        with tempfile.TemporaryDirectory() as d:
            rows=task(m,NoGateTok(),dict(id='synthetic',messages=[],input_ids=[2,3,4]),True,Judge(True),payload,
                      dict(max_new_tokens=4,layers=list(range(14,28))),ref,Path(d),[])
            self.assertTrue(all(not r['coverage'] and r['bank_calls']==0 for r in rows))

    def test_scoring_separates_manual_auto_and_missing(self):
        # These are explicit scorer fixtures, not generated-model results or experiment evidence.
        tasks=read(R/'data/evaluation.json');cfg=read(R/'data/config.json')
        for remove_module in [False,True]:
            with self.subTest(remove_module=remove_module),tempfile.TemporaryDirectory() as d:
                root=Path(d);(root/'data').mkdir();run=root/'runs';run.mkdir()
                for name,obj in [('config.json',cfg),('evaluation.json',tasks),('source_sha.json',{}),('text_reference.json',read(R/'data/text_reference.json'))]:write(root/'data'/name,obj)
                (root/'data/judgment.npz').write_bytes((R/'data/judgment.npz').read_bytes())
                for t in tasks:
                    for name in ['BASE','MANUAL_V','MODULE_V']:
                        if remove_module and name=='MODULE_V':continue
                        code=t['gold'] if name!='BASE' else t['gold'].replace('frozenset(','set(')
                        enabled=t['category']=='E' and name!='BASE'
                        write(run/t['id']/name/'generation.json',dict(id=t['id'],condition=name,text=code,stop='EOS',
                            enabled=enabled,bank_calls=14 if enabled else 0,output_ids=[2 if enabled else 1],coverage=True))
                write(run/'DECISION.json',dict(status='ENGINEERING_COMPLETE_PENDING_EXECUTION_SCORING'));seal_output(run)
                with patch.object(evaluate,'ROOT',root),patch.object(evaluate,'verify_package',lambda:None):
                    evaluate.main(argparse.Namespace(run=str(run),output=str(root/'evaluation')))
                result=read(root/'evaluation/DECISION.json');summary=read(root/'evaluation/SUMMARY.json')
                self.assertEqual(result['status'],'INCOMPLETE' if remove_module else 'DEVELOPMENT_8B_MODULE_V_LOOP_SUPPORTED')
                if remove_module:self.assertEqual(summary['MODULE_V']['E']['not_run'],8)
                else:self.assertEqual(summary['MODULE_V']['E']['joint'],8)

    def test_transfer_archive_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'bad.tar'
            with tarfile.open(p,'w') as t:
                m=tarfile.TarInfo('../outside');m.size=1;t.addfile(m,io.BytesIO(b'x'))
            with self.assertRaises(ValueError):safe_extract(p,root/'out')
            self.assertFalse((root.parent/'outside').exists())

if __name__=='__main__':unittest.main()
