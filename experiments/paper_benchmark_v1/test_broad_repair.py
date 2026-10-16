"""Synthetic CPU checks; no research dataset or training run is launched."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from .broad_window_model import BroadCMX, AdaptiveFilter, METHODS, backward_objective
from .paired_model import PairedCMX, backward_objective as pair_backward
from .test_paired_window import ToyModel, ToyTrain, scratch
from . import run_broad_repair as runner


class BroadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_no_mamba_and_expected_methods(self):
        self.assertEqual(METHODS,('R_RES','R_FILTER','F_DETAIL'))
        self.assertEqual(runner.METHODS,METHODS)
        with self.assertRaises(ValueError):
            BroadCMX('M_SCAN')

    def test_common_initialization_rng_identity_and_toggle(self):
        x=torch.rand(2,3,64,64); x[:,2]=1
        torch.manual_seed(42); base=PairedCMX('PAIR','CMX_B0',3,64,32).eval()
        rng=torch.get_rng_state()
        with torch.no_grad():
            expected=base(x)['logits']
        for method in METHODS:
            torch.manual_seed(42); model=BroadCMX(method,'CMX_B0',3,64,32).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for key,value in base.state_dict().items():
                torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
            before=x.clone()
            with torch.no_grad():
                if method!='R_FILTER':
                    torch.testing.assert_close(model(x)['logits'],expected,rtol=0,atol=0)
                model.candidate_enabled=False
                torch.testing.assert_close(model(x)['logits'],expected,rtol=0,atol=0)
            torch.testing.assert_close(x,before,rtol=0,atol=0)

    def test_filter_convex_bounded_constant_and_gradient(self):
        model=AdaptiveFilter()
        x=torch.rand(2,3,17,23)
        out,weights=model(x)
        self.assertEqual(out.shape,(2,1,17,23))
        torch.testing.assert_close(weights.sum(1),torch.ones_like(weights[:,0]))
        self.assertTrue((out>=0).all() and (out<=1).all())
        out.square().mean().backward()
        self.assertGreater(float(model.predict[-1].weight.grad.abs().sum()),0)
        x[:,:1]=.37
        torch.testing.assert_close(model(x)[0],x[:,:1])

    def test_backward_updates_candidate_and_encoder(self):
        x=torch.rand(2,3,64,64); x[:,2]=1
        y=torch.randint(3,(2,64,64)); y[:,0]=255
        for method in METHODS:
            torch.manual_seed(42)
            model=BroadCMX(method,'CMX_B0',3,64,32).train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            initial={k:v.detach().clone() for k,v in model.named_parameters() if k.startswith('adapters.candidate.')}
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                stats=backward_objective(model,x,y,'private',method,1,17,
                                        torch.amp.GradScaler('cuda',enabled=False),1.)
                self.assertEqual(stats['consistency'],0)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.backbone.parameters() if p.grad is not None),0)
                optimizer.step()
            self.assertTrue(any(not torch.equal(initial[k],v) for k,v in model.named_parameters() if k in initial))

    def test_disabled_candidate_exact_pair_objective(self):
        # This also checks same views, sample weighting and sequential backwards.
        x=torch.rand(2,3,64,64); x[:,2]=1; y=torch.randint(3,(2,64,64))
        torch.manual_seed(9); base=PairedCMX('PAIR','CMX_B0',3,64,32).eval()
        torch.manual_seed(9); model=BroadCMX('R_RES','CMX_B0',3,64,32).eval()
        model.candidate_enabled=False
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        a=pair_backward(base,x,y,'private','PAIR',1,11,scaler,.5)
        b=backward_objective(model,x,y,'private','R_RES',1,11,scaler,.5)
        self.assertAlmostEqual(a['loss'],b['loss'],places=6)
        for key,p in base.named_parameters():
            q=dict(model.named_parameters())[key]
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_ignored_labels_finite(self):
        model=ToyModel('PAIR')
        stats=backward_objective(model,torch.rand(2,3,8,8),torch.full((2,8,8),255),
            'private','R_RES',1,3,torch.amp.GradScaler('cuda',enabled=False),1.)
        self.assertEqual(stats['loss'],0)

    def test_complete_matrix(self):
        args=SimpleNamespace(output_dir=scratch(),budget_hours=10,reference_results=[
            dict(method='PAIR',seed=42,elapsed_minutes=90)])
        calls=[]
        def train(a,d,m,s,p):
            calls.append((d,m,s)); return dict(dataset=d,method=m,seed=s,group=f'{d}/{m}_seed{s}')
        with contextlib.ExitStack() as stack:
            for key,value in [('preflight',lambda a:{}),('train_group',train),('summarize',lambda *a:None)]:
                stack.enter_context(patch.object(runner,key,value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            runner.execute(args)
        self.assertEqual(calls,[(d,m,s) for s in (42,777) for m in METHODS for d in ('nyuv2','private')])
        complete=runner.read(args.output_dir/'complete.json')
        self.assertEqual(complete['training_groups'],12)
        self.assertEqual(complete['public_condition_evaluations'],132)
        self.assertEqual(complete['private_evaluations'],6)
        self.assertFalse(complete['final_test_access'])
        self.assertEqual(runner.read(args.output_dir/'status.json')['groups_completed'],12)

    def test_summary_matched_seeds_means(self):
        def row(m,s,v):
            return dict(dataset='nyuv2',method=m,seed=s,clean_miou=v,corruption_mean_miou=v,
                        photon_proxy_heavy_miou=v,joint_heavy_miou=v)
        args=SimpleNamespace(output_dir=scratch(),reference_results=[row('PAIR',42,.5),row('PAIR',777,.6)])
        with contextlib.redirect_stdout(io.StringIO()):
            runner.summarize(args,[row('R_RES',42,.52),row('R_RES',777,.63),row('F_DETAIL',42,.51)])
        rows=runner.read(args.output_dir/'comparison.json')['summaries']
        a=next(r for r in rows if r['method']=='R_RES')
        self.assertEqual(a['seeds'],[42,777])
        self.assertAlmostEqual(a['clean_miou']['mean_delta_pp'],2.5)
        self.assertAlmostEqual(a['clean_miou']['mean'],.575)
        a=next(r for r in rows if r['method']=='F_DETAIL')
        self.assertEqual(a['seeds'],[42]); self.assertIsNone(a['clean_miou']['std'])

    def test_resume_and_completed_identity_and_seed_curves(self):
        root=scratch(); reference=scratch(); replication=scratch()
        args=SimpleNamespace(output_dir=root,reference_dir=reference,replication_dir=replication,
                             private_dir=root,nyuv2_dir=root,device='cpu')
        source=dict(model=ToyModel('PAIR').state_dict(),args={'model':'CMX_B2','decoder_dim':32})
        metric=dict(regions={'all':{'miou':.4}},degradation_audit=[])
        for seed in (42,777):
            parent=reference if seed==42 else replication
            for f,v in runner.CONDITIONS:
                runner.save(parent/f'nyuv2/PAIR_seed{seed}/curves/{f}_{v}.json',metric)
        with contextlib.ExitStack() as stack:
            for key,value in [('load_source',lambda *a:source),('PairedCMX',ToyModel),
                ('TrainingData',ToyTrain),('PrivateTrain',ToyTrain),('PrivatePaperDataset',lambda *a:[]),
                ('evaluate',lambda *a:{'foreground_iou':.4}),('private_details',lambda *a:{'rows':[]}),
                ('DiagnosticData',lambda *a:None),('regional',lambda *a:metric),('EPOCHS',2)]:
                stack.enter_context(patch.object(runner,key,value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            checkpoint=runner.atomic_checkpoint
            def stop(path,state):
                checkpoint(path,state); raise RuntimeError('synthetic interruption')
            with patch.object(runner,'atomic_checkpoint',stop):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    runner.train_group(args,'nyuv2','R_RES',42,{'test':1})
            with self.assertRaisesRegex(ValueError,'protocol/group'):
                runner.train_group(args,'nyuv2','R_RES',42,{'test':2})
            for seed,method in ((42,'R_RES'),(777,'R_FILTER'),(777,'F_DETAIL')):
                for dataset in ('nyuv2','private'):
                    result=runner.train_group(args,dataset,method,seed,{'test':1})
                    self.assertEqual(result,runner.train_group(args,dataset,method,seed,{'test':1}))
            path=root/'private/F_DETAIL_seed777/result.json'
            wrong=runner.read(path); wrong['seed']=42; runner.save(path,wrong)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                runner.train_group(args,'private','F_DETAIL',777,{'test':1})


if __name__=='__main__':
    unittest.main()
