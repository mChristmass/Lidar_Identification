"""Synthetic tests for scalar matching without replacing class-balanced gradients."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from .paired_model import consistency_loss
from .consistency_refine_model import RefinedCMX, backward_objective as pixel_backward
from .consistency_scale_model import ScaleCMX, matched_consistency, backward_objective
from .test_paired_window import ToyModel, ToyTrain, scratch
from . import run_consistency_scale as runner


class ScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        labels=torch.tensor([[[0,0,0,1]]])
        reference=torch.tensor([[[[5.,5.,5.,-5.]], [[-5.,-5.,-5.,5.]]]],requires_grad=True)
        student=torch.tensor([[[[0.,.1,.2,.3]], [[.4,.3,.2,.1]]]],requires_grad=True)
        return student,reference,labels

    def test_scalar_matches_pixel_gradient_stays_balanced(self):
        student,reference,labels=self.fixture()
        scaled,balanced,pixel,ratio,_=matched_consistency(student,reference,labels)
        torch.testing.assert_close(scaled,pixel)
        self.assertFalse(ratio.requires_grad)
        grad=torch.autograd.grad(scaled,student,retain_graph=True)[0]
        bg=torch.autograd.grad(balanced,student,retain_graph=True)[0]
        pg=torch.autograd.grad(pixel,student,retain_graph=True)[0]
        torch.testing.assert_close(grad,bg*ratio)
        self.assertFalse(torch.allclose(grad,pg))
        scaled.backward()
        self.assertIsNone(reference.grad)

    def test_balanced_term_identical_to_cons(self):
        student,reference,labels=self.fixture()
        _,balanced,_,_,_=matched_consistency(student,reference,labels)
        old,_=consistency_loss(student,reference,labels)
        torch.testing.assert_close(old,balanced)

    def test_no_support_and_near_zero_loss_finite(self):
        student,reference,labels=self.fixture()
        output=matched_consistency(student,reference,torch.full_like(labels,255))
        self.assertEqual(float(output[0].detach()),0)
        self.assertEqual(float(output[3]),0)
        output[0].backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        same=reference.detach().clone().requires_grad_()
        output=matched_consistency(same,reference,labels)
        output[0].backward()
        self.assertTrue(torch.isfinite(same.grad).all())
        self.assertTrue(all(torch.isfinite(x).all() for x in output[:4]))

    def test_exact_pixel_reuse_and_same_model_initialization(self):
        torch.manual_seed(42);old=RefinedCMX('PIXEL','CMX_B0',3,64,32).eval()
        torch.manual_seed(42);new=ScaleCMX('PIXEL','CMX_B0',3,64,32).eval()
        for k,v in old.state_dict().items():
            torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)
        x=torch.rand(2,3,64,64);x[:,2]=1;y=torch.randint(3,(2,64,64))
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        a=pixel_backward(old,x,y,'private','PIXEL',5,77,scaler,1.)
        b=backward_objective(new,x,y,'private','PIXEL',5,77,scaler,1.)
        self.assertEqual(a,b)
        for p,q in zip(old.parameters(),new.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_scale_real_model_backward(self):
        model=ScaleCMX('SCALE','CMX_B0',3,64,32).train()
        x=torch.rand(2,3,64,64);x[:,2]=1;y=torch.randint(3,(2,64,64))
        stats=backward_objective(model,x,y,'private','SCALE',5,42,
                                 torch.amp.GradScaler('cuda',enabled=False),1.)
        self.assertLess(stats['scale_match_abs_error'],1e-6)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_fixed_matrix_ten_new_groups(self):
        args=SimpleNamespace(output_dir=scratch(),budget_hours=10,reference_results=[
            dict(method='PIXEL',seed=42,elapsed_minutes=90)])
        calls=[]
        def train(a,d,m,s,p):
            calls.append((d,m,s));return dict(dataset=d,method=m,seed=s)
        with contextlib.ExitStack() as stack:
            for key,value in [('preflight',lambda a:{}),('train_group',train),('summarize',lambda *a:None)]:
                stack.enter_context(patch.object(runner,key,value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()));runner.execute(args)
        expected=[(d,m,s) for s in (42,777,2025) for m in ('PIXEL','SCALE')
                  if not(s==42 and m=='PIXEL') for d in ('nyuv2','private')]
        self.assertEqual(calls,expected)
        complete=runner.read(args.output_dir/'complete.json')
        self.assertEqual(complete['training_groups'],10)
        self.assertEqual(complete['public_condition_evaluations'],110)
        self.assertEqual(complete['epochs_per_group'],20)
        self.assertFalse(complete['final_test_access'])

    def test_summary_only_uses_matching_seeds(self):
        def row(m,s,v):
            return dict(dataset='nyuv2',method=m,seed=s,clean_miou=v,corruption_mean_miou=v,
                        photon_proxy_heavy_miou=v,joint_heavy_miou=v)
        args=SimpleNamespace(output_dir=scratch(),reference_results=[
            row('PAIR',42,.5),row('CONS',42,.49),row('PIXEL',42,.51),
            row('PAIR',777,.6),row('CONS',777,.59)])
        with contextlib.redirect_stdout(io.StringIO()):
            runner.summarize(args,[row('SCALE',42,.52),row('SCALE',777,.63)])
        result=runner.read(args.output_dir/'comparison.json')
        s=next(r for r in result['summaries'] if r['dataset']=='nyuv2' and r['method']=='SCALE' and r['reference']=='PIXEL')
        self.assertEqual(s['seeds'],[42]);self.assertAlmostEqual(s['clean_miou']['mean_delta_pp'],1)
        self.assertIsNone(s['clean_miou']['std_delta_pp'])
        s=next(r for r in result['summaries'] if r['dataset']=='nyuv2' and r['method']=='SCALE' and r['reference']=='PAIR')
        self.assertEqual(s['seeds'],[42,777]);self.assertAlmostEqual(s['clean_miou']['mean_delta_pp'],2.5)

    def test_resume_uses_correct_historical_seed_curves(self):
        root=scratch();reference=scratch();replication=scratch()
        args=SimpleNamespace(output_dir=root,reference_dir=reference,replication_dir=replication,
                             private_dir=root,nyuv2_dir=root,device='cpu')
        source=dict(model=ToyModel('CONS').state_dict(),args={'model':'CMX_B2','decoder_dim':32})
        metric=dict(regions={'all':{'miou':.4}},degradation_audit=[])
        for seed in (42,777,2025):
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
                checkpoint(path,state);raise RuntimeError('synthetic interruption')
            with patch.object(runner,'atomic_checkpoint',stop):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    runner.train_group(args,'nyuv2','SCALE',42,{'test':1})
            with self.assertRaisesRegex(ValueError,'protocol/group'):
                runner.train_group(args,'nyuv2','SCALE',42,{'test':2})
            for seed,method in ((42,'SCALE'),(777,'PIXEL'),(2025,'SCALE')):
                for dataset in ('nyuv2','private'):
                    result=runner.train_group(args,dataset,method,seed,{'test':1})
                    self.assertEqual(result,runner.train_group(args,dataset,method,seed,{'test':1}))


if __name__=='__main__':
    unittest.main()
