"""Optional real tiny-T5 tests; no pretrained download."""
import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cuphead.world_model import build_frozen_t5_sequence_model

@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "optional ML dependencies absent")
class SequenceTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import T5Config, T5Model
        self.t = torch
        torch.manual_seed(7)
        torch.set_num_threads(1)
        backbone = T5Model(T5Config(vocab_size=16, d_model=16, d_kv=4, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0.5))
        self.m = build_frozen_t5_sequence_model(3, 5, backbone=backbone)
        self.c, self.a, self.y = torch.randn(2,3,3), torch.randn(2,4,5), torch.randn(2,4,3)

    def test_frozen_weights_and_adapter_gradients(self):
        self.m.train()
        self.assertFalse(self.m.backbone.training)
        before = {k:v.clone() for k,v in self.m.backbone.state_dict().items()}
        opt = self.t.optim.Adam(p for p in self.m.parameters() if p.requires_grad)
        self.m.loss(self.c, self.a, self.y).backward()
        for adapter in self.m.adapters:
            self.assertGreater(adapter.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.m.backbone.parameters()))
        opt.step()
        for k,v in self.m.backbone.state_dict().items():
            self.assertTrue(self.t.equal(before[k], v))

    def test_causal_alignment(self):
        baseline = self.m(self.c, self.a, self.y)
        targets, actions = self.y.clone(), self.a.clone()
        targets[:,1:] += 100
        actions[:,2:] -= 100
        self.t.testing.assert_close(baseline[:,:2], self.m(self.c, actions, targets)[:,:2])

    def test_padding_and_gradients(self):
        cm = self.t.tensor([[True,True,False]]*2)
        am = self.t.tensor([[True,True,False,False]]*2)
        expected = self.m.loss(self.c[:,:2], self.a[:,:2], self.y[:,:2])
        self.c[:,2:] = float("nan")
        self.a[:,2:] = float("nan")
        self.y[:,2:] = float("nan")
        actual = self.m.loss(self.c,self.a,self.y,context_mask=cm,action_mask=am)
        self.t.testing.assert_close(expected, actual)
        actual.backward()
        self.assertTrue(self.t.isfinite(self.m.context_adapter.weight.grad).all())

    def test_autoregressive_rollout(self):
        predicted = self.m.rollout(self.c,self.a)
        self.t.testing.assert_close(predicted,self.m(self.c,self.a,predicted))
        changed = self.a.clone()
        changed[:,2:] += 100
        self.t.testing.assert_close(predicted[:,:2],self.m.rollout(self.c,changed)[:,:2])
        self.assertFalse(predicted.requires_grad)

    def test_invalid_inputs(self):
        for mask in (self.t.zeros(2,3,dtype=self.t.bool), self.t.tensor([[True,False,True]]*2)):
            with self.assertRaises(ValueError):
                self.m.rollout(self.c,self.a,context_mask=mask)
        with self.assertRaises(ValueError):
            self.m.rollout(self.c,self.a[:1])
        self.a[0,0,0] = float("inf")
        with self.assertRaises(ValueError):
            self.m.rollout(self.c,self.a)
