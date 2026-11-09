"""Optional CPU synthetic architecture tests; never loads a research model checkpoint."""
import tempfile, unittest, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from statemod.runtime_helpers import forward,cache_hash,metrics
    from statemod.backend import Bank,call,clone,disjoint,clean,replay,capture,checked_step
    AVAILABLE=True
except ImportError:
    AVAILABLE=False

@unittest.skipUnless(AVAILABLE,'optional torch/Transformers runtime not installed')
class BackendChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(9)
        c=Qwen3Config(vocab_size=67,hidden_size=32,intermediate_size=48,num_hidden_layers=28,
                     num_attention_heads=4,num_key_value_heads=1,head_dim=8)
        c._attn_implementation='eager';cls.model=Qwen3ForCausalLM(c).eval();cls.model.requires_grad_(False)
        r=np.random.default_rng(9);cls.payload={k:r.normal(size=(28,1,36,8)).astype('float32')*.1 for k in ['K','V']}
        cls.layers=list(range(14,28))

    def test_zero_is_nohook_and_independent(self):
        m=self.model;a,c=forward(m,[2,3,4]);b,cc=call(m,[2,3,4],None,Bank(m,self.payload,self.layers,0))
        self.assertTrue(np.array_equal(a,b));self.assertEqual(cache_hash(c),cache_hash(cc));self.assertTrue(disjoint(c,cc));self.assertTrue(clean(m))

    def test_actual_single_step_and_history_replay(self):
        m=self.model;_,c=forward(m,[2,3,4]);before=cache_hash(c);bank=Bank(m,self.payload,self.layers)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            a,cc,row=checked_step(m,[5],clone(m,c),[2,3,4,5],bank,3,p/'first',True)
            self.assertEqual(row['bank_calls'],14)
            z,zc=forward(m,[5],clone(m,c));self.assertGreater(float(abs(a-z).max()),0)
            b,bc,row2=checked_step(m,[6],cc,[2,3,4,5,6],bank,3,p/'second',False)
            self.assertEqual(row2['bank_calls'],0);self.assertEqual(bc.get_seq_length(),5)
            repeat,rc=replay(m,[2,3,4,5,6],bank,3)
            self.assertLess(metrics(b,repeat)['rms'],1e-6)
        self.assertEqual(cache_hash(c),before);self.assertTrue(clean(m))

    def test_hook_cleanup_on_exception(self):
        bank=Bank(self.model,self.payload,self.layers)
        with self.assertRaises(RuntimeError):
            with bank.context():raise RuntimeError('synthetic fault')
        self.assertTrue(clean(self.model))

    def test_probe_is_readonly(self):
        m=self.model;_,c=forward(m,[3,4,5]);before=cache_hash(c)
        z,cc,h=capture(m,[6],clone(m,c));plain,pc=forward(m,[6],clone(m,c))
        self.assertTrue(np.array_equal(z,plain));self.assertEqual(cache_hash(cc),cache_hash(pc));self.assertEqual(cache_hash(c),before)
        self.assertEqual(h.shape,(32,));self.assertTrue(clean(m))
