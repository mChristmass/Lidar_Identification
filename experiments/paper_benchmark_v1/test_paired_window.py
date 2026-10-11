"""Synthetic CPU tests only; scratch retained."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from .model import CMXPaper
from .task_control_model import TaskControlCMX
from .paired_model import (PairedCMX, METHODS, perturb_view, safe_ce, consistency_loss,
                           completion_loss, backward_objective, CONS_WEIGHT, MASK_WEIGHT)
from . import run_paired_window as runner
from .test_context_window import scratch, ToyTrain


class ToyModel(nn.Module):
    def __init__(self, method, *args):
        super().__init__()
        self.backbone = nn.Conv2d(3, 8, 1)
        self.decode_head = nn.Conv2d(8, 2, 1)
        self.adapters = nn.ModuleDict({'completion': nn.ModuleList([nn.Conv2d(8,8,1)])})
        self.capture_features = False
        self.adapters_enabled = True
        self.audit_enabled = False
        self.audit_rows = []
    def load_baseline(self, state):
        self.load_state_dict(state)
    def forward(self, x):
        f = self.backbone(x)
        return dict(logits=self.decode_head(f), features=(f,))


class PairedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_views_deterministic_preserve_inputs_and_holes(self):
        x=torch.rand(4,3,32,32);x[:,2]=(x[:,2]>.3).float();x[:,1]*=x[:,2]
        before=x.clone()
        for domain in ('private','nyuv2'):
            a,hidden=perturb_view(x,domain,55)
            torch.rand(100)
            b,mask=perturb_view(x,domain,55)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertTrue(torch.equal(mask,hidden))
            self.assertTrue(torch.equal(x,before))
            self.assertFalse((hidden & (x[:,2:3]==0)).any())
            self.assertTrue((a[:,1:2][hidden]==0).all())
            self.assertTrue((a[:,2:3][hidden]==0).all())
            self.assertTrue(torch.equal(a[:,1:2][~hidden],x[:,1:2][~hidden]))
            self.assertGreater(hidden.sum(),0)
        _,hidden=perturb_view(torch.zeros_like(x),'private',55)
        self.assertEqual(hidden.sum(),0)

    def test_consistency_detached_correctness_and_empty(self):
        labels=torch.tensor([[[0,1],[255,1]]])
        reference=torch.tensor([[[[5.,5.],[5.,0.]],[[0.,0.],[0.,5.]]]],requires_grad=True)
        student=torch.zeros_like(reference,requires_grad=True)
        loss,coverage=consistency_loss(student,reference,labels)
        self.assertEqual(float(coverage),.5)
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertEqual(student.grad[...,0,1].abs().sum(),0)
        self.assertEqual(student.grad[...,1,0].abs().sum(),0)
        self.assertGreater(student.grad[...,1,1].abs().sum(),0)
        empty,_=consistency_loss(student,reference,torch.full_like(labels,255))
        self.assertEqual(float(empty.detach()),0)
        self.assertTrue(torch.isfinite(safe_ce(student,torch.full_like(labels,255))))

    def test_completion_masks_targets_and_gradients(self):
        model=ToyModel('MASK')
        feature=torch.rand(2,8,4,4,requires_grad=True)
        target=torch.rand(2,8,4,4,requires_grad=True)
        hidden=torch.zeros(2,1,8,8,dtype=torch.bool)
        labels=torch.zeros(2,8,8,dtype=torch.long)
        self.assertEqual(float(completion_loss(model,(feature,),(target,),hidden,labels).detach()),0)
        hidden[:,:,2:6,2:6]=True
        self.assertEqual(float(completion_loss(model,(feature,),(target,),hidden,torch.full_like(labels,255)).detach()),0)
        completion_loss(model,(feature,),(target,),hidden,labels).backward()
        self.assertIsNone(target.grad)
        self.assertGreater(feature.grad.abs().sum(),0)
        self.assertGreater(model.adapters['completion'][0].weight.grad.abs().sum(),0)

    def test_class_balance(self):
        labels=torch.zeros(1,1,10,dtype=torch.long);labels[...,9]=1
        reference=torch.full((1,2,1,10),-5.)
        reference[:,0,:,:9]=5.;reference[:,1,:,9]=5.
        student=torch.randn_like(reference,requires_grad=True)
        loss,_=consistency_loss(student,reference,labels)
        a,_=consistency_loss(student[:,:,:,:9],reference[:,:,:,:9],labels[:,:,:9])
        b,_=consistency_loss(student[:,:,:,9:],reference[:,:,:,9:],labels[:,:,9:])
        torch.testing.assert_close(loss,(a+b)/2)

    def test_all_initial_inference_matches_baseline(self):
        base=CMXPaper('CMX_B0',3,64,32).eval();x=torch.rand(2,3,64,64)
        with torch.no_grad():
            expected=base(x)['logits']
            for method in METHODS:
                model=PairedCMX(method,'CMX_B0',3,64,32).eval()
                model.load_baseline(base.state_dict())
                torch.testing.assert_close(model(x)['logits'],expected,rtol=0,atol=0)

    def test_completion_heads_inference_independent_and_reload(self):
        model=PairedCMX('BOTH','CMX_B0',3,64,32).eval();x=torch.rand(2,3,64,64)
        with torch.no_grad():
            before=model(x)['logits']
            for p in model.adapters['completion'].parameters():
                p.add_(10)
            torch.testing.assert_close(before,model(x)['logits'],rtol=0,atol=0)
            restored=PairedCMX('BOTH','CMX_B0',3,64,32).eval()
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(before,restored(x)['logits'],rtol=0,atol=0)

    def test_initialization_shared_raw_control_unchanged(self):
        torch.manual_seed(42)
        common=TaskControlCMX('S_RAW','CMX_B0',3,64,32).state_dict();heads=None
        for method in METHODS[1:]:
            torch.manual_seed(42);model=PairedCMX(method,'CMX_B0',3,64,32)
            for k,v in common.items():
                torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
            if method in ('MASK','BOTH'):
                current=model.adapters['completion'].state_dict()
                if heads is not None:
                    for k,v in heads.items():
                        torch.testing.assert_close(v,current[k],rtol=0,atol=0)
                heads=current

    def test_real_backwards_all_methods(self):
        x=torch.rand(2,3,64,64);x[:,2]=1
        labels=torch.randint(3,(2,64,64))
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        for method in METHODS:
            model=PairedCMX(method,'CMX_B0',3,64,32).train()
            stats=backward_objective(model,x,labels,'private',method,5,42,scaler,1.)
            self.assertTrue(np.isfinite(list(stats.values())).all())
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertFalse(model.capture_features);self.assertIsNone(model._captured)
            if method in ('MASK','BOTH'):
                self.assertGreater(sum(p.grad.abs().sum() for p in model.adapters['completion'].parameters()),0)
            if method not in ('B0','S_RAW'):
                self.assertGreater(stats['ce_secondary'],0)

    def test_sequential_gradient_matches_joint_objective(self):
        torch.manual_seed(5);model=ToyModel('BOTH').eval()
        ref=ToyModel('BOTH').eval();ref.load_state_dict(model.state_dict())
        x=torch.rand(2,3,32,32);x[:,2]=1;labels=torch.randint(2,(2,32,32))
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        backward_objective(model,x,labels,'nyuv2','BOTH',5,13,scaler,.3)
        observed=ref(x);secondary,hidden=perturb_view(x,'nyuv2',13);changed=ref(secondary)
        ce=.5*(safe_ce(observed['logits'],labels)+safe_ce(changed['logits'],labels))
        cons,_=consistency_loss(changed['logits'],observed['logits'],labels)
        recover=completion_loss(ref,changed['features'],observed['features'],hidden,labels)
        (.3*(ce+CONS_WEIGHT*cons+MASK_WEIGHT*recover)).backward()
        for p,q in zip(model.parameters(),ref.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)

    def test_runtime_schedule_and_resume(self):
        rows=[dict(dataset=d,step_seconds=.2,eval_seconds=.03) for m in METHODS for d in ('nyuv2','private')]
        for n in (20,16,12):
            self.assertEqual(runner.choose_epochs(rows,(runner.estimate_minutes(rows,n)+.01)/60),n)
        with self.assertRaisesRegex(RuntimeError,'No training started'):
            runner.choose_epochs(rows,.01)
        root=scratch();runner.save(root/'schedule.json',dict(epochs=16,budget_hours=8))
        args=SimpleNamespace(output_dir=root,budget_hours=8)
        self.assertEqual(runner.prepare_schedule(args),16)
        args.budget_hours=10
        with self.assertRaises(ValueError):
            runner.prepare_schedule(args)

    def test_execute_all_twelve_groups(self):
        args=SimpleNamespace(output_dir=scratch(),budget_hours=10.,device='cpu');calls=[]
        def train(a,d,m,s,p):
            calls.append((d,m,s));return dict(dataset=d,method=m,seed=s)
        with contextlib.ExitStack() as stack:
            for key,value in [('EPOCHS',20),('prepare_schedule',lambda a:16),('preflight',lambda a:{}),
                              ('audit_inputs',lambda a:None),('train_group',train),('summarize',lambda *a:None)]:
                stack.enter_context(patch.object(runner,key,value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()));runner.execute(args)
        self.assertEqual(calls,[(d,m,42) for m in METHODS for d in ('nyuv2','private')])
        result=runner.read(args.output_dir/'complete.json')
        self.assertEqual(result['training_groups'],12);self.assertEqual(result['public_condition_evaluations'],132)
        self.assertEqual(result['epochs_per_group'],16);self.assertFalse(result['final_test_access'])

    def test_summary_pair_control_and_combination(self):
        args=SimpleNamespace(output_dir=scratch())
        rows=[dict(dataset='nyuv2',method=m,seed=42,clean_miou=v,corruption_mean_miou=v,
                   photon_proxy_heavy_miou=v,joint_heavy_miou=v)
              for m,v in zip(METHODS,[.4,.41,.42,.44,.45,.48])]
        with contextlib.redirect_stdout(io.StringIO()):
            runner.summarize(args,rows)
        result=runner.read(args.output_dir/'comparison.json')
        pairs={r['comparison']:r['delta_pp']['clean_miou'] for r in result['paired_comparisons']}
        self.assertAlmostEqual(pairs['CONS_minus_PAIR'],2.);self.assertAlmostEqual(pairs['MASK_minus_PAIR'],3.)
        self.assertAlmostEqual(pairs['factorial_interaction'],1.)
        self.assertTrue(all(r['clean_miou']['std'] is None for r in result['summaries']))

    def test_training_resume_and_completed_reuse(self):
        root=scratch();args=SimpleNamespace(output_dir=root,private_dir=root,nyuv2_dir=root,device='cpu')
        source=dict(model=ToyModel('B0').state_dict(),args={'model':'CMX_B2','decoder_dim':32})
        metric=dict(regions={'all':{'miou':.4}},degradation_audit=[])
        with contextlib.ExitStack() as stack:
            for key,value in [('load_source',lambda *a:source),('PairedCMX',ToyModel),('TrainingData',ToyTrain),
                ('PrivateTrain',ToyTrain),('PrivatePaperDataset',lambda *a:[]),
                ('evaluate',lambda *a:{'foreground_iou':.4}),('private_details',lambda *a:{'rows':[]}),
                ('DiagnosticData',lambda *a:None),('regional',lambda *a:metric),('EPOCHS',2)]:
                stack.enter_context(patch.object(runner,key,value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            original=runner.atomic_checkpoint
            def interrupt(path,state):
                original(path,state);raise RuntimeError('synthetic stop')
            with patch.object(runner,'atomic_checkpoint',interrupt):
                with self.assertRaisesRegex(RuntimeError,'synthetic stop'):
                    runner.train_group(args,'nyuv2','B0',42,{'test':1})
            with self.assertRaisesRegex(ValueError,'protocol/group'):
                runner.train_group(args,'nyuv2','B0',42,{'test':2})
            for dataset in ('nyuv2','private'):
                for method in METHODS:
                    result=runner.train_group(args,dataset,method,42,{'test':1})
                    self.assertEqual(result,runner.train_group(args,dataset,method,42,{'test':1}))

    def test_preflight_freezes_inputs_code_and_schedule(self):
        root=scratch()
        args=SimpleNamespace(output_dir=root/'out',private_dir=root/'private',nyuv2_dir=root/'nyu',
                             repair_dir=root/'repair',private_source=root/'ps',public_source=root/'ns',budget_hours=10)
        args.output_dir.mkdir();split=args.private_dir/'paper_split_v1';split.mkdir(parents=True)
        np.save(split/'train_indices.npy',np.arange(1,971));np.save(split/'dev_indices.npy',np.arange(971,1108))
        runner.save(args.repair_dir/'protocol.json',dict(private_sha256={'paper_split_v1/dev_indices.npy':'same'},
                    manifests={'train':'same','dev':'same'}))
        with contextlib.ExitStack() as stack:
            for key,value in [('load_source',lambda *a:{}),('read_manifest',lambda *a:[]),
                              ('TrainingData',ToyTrain),('digest',lambda p:'same')]:
                stack.enter_context(patch.object(runner,key,value))
            original=runner.preflight(args);self.assertEqual(original,runner.preflight(args))
            with patch.object(runner,'digest',lambda p:'changed' if p.name=='schedule.json' else 'same'):
                with self.assertRaisesRegex(ValueError,'Protocol/input/code changed'):
                    runner.preflight(args)
            with patch.object(runner,'digest',lambda p:'changed' if p.name=='train.jsonl' else 'same'):
                with self.assertRaisesRegex(ValueError,'manifest changed'):
                    runner.preflight(args)


if __name__ == '__main__':
    unittest.main()
