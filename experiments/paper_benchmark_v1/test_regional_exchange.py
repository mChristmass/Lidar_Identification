"""Small CPU tensor tests; no dataset, checkpoint or training job required."""
import copy
import unittest
import numpy as np
import torch
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from experiments.cmx_initial_transfer.official.net_utils import FeatureRectifyModule
from .regional_exchange import RegionalFRM, evidence


class ExchangeTests(unittest.TestCase):
    def test_baseline_and_active_regional_initialization(self):
        torch.manual_seed(4)
        original=FeatureRectifyModule(8)
        x,y=torch.randn(2,8,12,16),torch.randn(2,8,12,16)
        expected=original(x,y)
        for arm in ('B','P'):
            module=RegionalFRM(copy.deepcopy(original),3,arm)
            actual=module(x,y)
            if arm=='B':
                for a,b in zip(actual,expected): torch.testing.assert_close(a,b)
            else:
                self.assertGreater(float((actual[0]-expected[0]).detach().abs().mean()),1e-5)

    def test_probe_loss_does_not_train_encoder(self):
        module=RegionalFRM(FeatureRectifyModule(8),3,'G')
        x=torch.randn(1,8,8,8,requires_grad=True)
        y=torch.randn_like(x,requires_grad=True)
        module(x,y)
        sum(z.square().mean() for z in module.logits).backward()
        self.assertIsNone(x.grad); self.assertIsNone(y.grad)
        self.assertIsNotNone(module.probes[0].weight.grad)

    def test_hard_gate_and_evidence(self):
        module=RegionalFRM(FeatureRectifyModule(8),2,'H')
        module.threshold.fill_(1.1); module.capture=True
        x,y=torch.randn(1,8,8,8),torch.randn(1,8,8,8)
        a,b=module(x,y)
        torch.testing.assert_close(a,x); torch.testing.assert_close(b,y)
        self.assertEqual(float(module.maps['gate_d_to_i'].sum()),0.)
        p=torch.tensor([[[[20.]], [[-20.]]]])
        c,ri,rd=evidence(p,p.flip(1))
        self.assertLess(float(c),1e-5); self.assertGreater(float(ri),.99)

    def test_real_backbone_snapshot(self):
        from .model import CMXPaper
        from .regional_exchange import attach
        from .regional_exchange_diagnostics import snapshot
        model=attach(CMXPaper('CMX_B0',40,128,32),40,'PG').eval()
        data=[dict(inputs=torch.rand(3,128,128),labels=torch.randint(0,40,(128,128)),sample_id='synthetic')]
        tmp=Path('data/analysis/regional_exchange_smoke')/uuid.uuid4().hex
        with patch('experiments.paper_benchmark_v1.regional_exchange_diagnostics.datasets',return_value={'clean_medium':data}):
            result=snapshot(SimpleNamespace(device='cpu',visual_samples=1),model,'nyuv2',tmp,'final',full=True)
        self.assertIn('miou',result['clean_medium'])
        with np.load(tmp/'diagnostics/final/clean_medium/frame_0000.npz') as archive:
            self.assertEqual(archive['s0_ca_local_grid'].shape[-2:],(4,4))
        self.assertTrue((tmp/'diagnostics/final/clean_medium/frame_0000.png').is_file())

    def test_training_and_completed_resume(self):
        from .model import CMXPaper
        from .regional_exchange import attach
        from .run_regional_exchange import train
        class Toy(torch.utils.data.Dataset):
            def __len__(self): return 2
            def __getitem__(self,index):
                gen=torch.Generator().manual_seed(index)
                return torch.rand(3,128,128,generator=gen),torch.randint(0,40,(128,128),generator=gen)
        args=SimpleNamespace(output_dir=Path('data/analysis/regional_exchange_smoke')/uuid.uuid4().hex,
            deadline=float('inf'),device='cpu',probe_epochs=1,epochs=2,lr=1e-5)
        def factory(*unused):
            return attach(CMXPaper('CMX_B0',40,128,32),40,'PG')
        with patch('experiments.paper_benchmark_v1.run_regional_exchange.build',side_effect=factory), \
             patch('experiments.paper_benchmark_v1.run_regional_exchange.dataset',return_value=Toy()), \
             patch('experiments.paper_benchmark_v1.run_regional_exchange.snapshot',return_value={'clean':{'miou':.1}}):
            with patch('experiments.paper_benchmark_v1.run_regional_exchange.expired',
                       side_effect=lambda a:(a.output_dir/'nyuv2/PG_seed42/last.pth').exists()):
                paused=train(args,'nyuv2',42,'PG',{'test':True})
            self.assertIsNone(paused)
            result=train(args,'nyuv2',42,'PG',{'test':True})
            resumed=train(args,'nyuv2',42,'PG',{'test':True})
            self.assertEqual(result,resumed)
            self.assertTrue((args.output_dir/'nyuv2/PG_seed42/midpoint.pth').is_file())


if __name__=='__main__': unittest.main()
