"""Synthetic-only tests for isolated spatial-support and reduction changes."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from .paired_model import PairedCMX, consistency_loss, perturb_view
from .consistency_refine_model import RefinedCMX, loss_and_support, affected_region, backward_objective
from .test_paired_window import ToyModel, ToyTrain, scratch
from . import run_consistency_refine as runner


class RefineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_local_full_region_equals_original_consistency(self):
        reference=torch.tensor([[[[5.,-5.]], [[-5.,5.]]]],requires_grad=True)
        labels=torch.tensor([[[0,1]]]);student=torch.randn_like(reference,requires_grad=True)
        a,_=consistency_loss(student,reference,labels)
        b,selected,_=loss_and_support(student,reference,labels,torch.ones(1,1,1,2,dtype=torch.bool),'LOCAL')
        torch.testing.assert_close(a,b)
        b.backward()
        self.assertIsNone(reference.grad)
        self.assertTrue(selected.all())

    def test_local_excludes_unaffected_and_ignored_pixels(self):
        ref=torch.tensor([[[[5.,5.,5.]], [[-5.,-5.,-5.]]]])
        labels=torch.tensor([[[0,0,255]]]);student=torch.randn_like(ref,requires_grad=True)
        region=torch.tensor([[[[True,False,True]]]])
        loss,selected,_=loss_and_support(student,ref,labels,region,'LOCAL')
        loss.backward()
        self.assertEqual(int(selected.sum()),1)
        self.assertEqual(student.grad[:,:,:,1:].abs().sum(),0)
        zero,_,_=loss_and_support(student,ref,labels,torch.zeros_like(region),'LOCAL')
        self.assertEqual(float(zero.detach()),0)

    def test_pixel_reduction_only_changes_weighting(self):
        ref=torch.tensor([[[[5.,5.,-5.]], [[-5.,-5.,5.]]]])
        labels=torch.tensor([[[0,0,1]]]);student=torch.randn_like(ref,requires_grad=True)
        region=torch.zeros(1,1,1,3,dtype=torch.bool)
        loss,selected,eligible=loss_and_support(student,ref,labels,region,'PIXEL')
        teacher=ref.softmax(1)
        kl=(teacher*(teacher.log()-student.log_softmax(1))).sum(1)
        torch.testing.assert_close(loss,kl.mean())
        self.assertTrue(torch.equal(selected,eligible))
        self.assertTrue(selected.all())

    def test_region_uses_actual_change_and_hidden_mask(self):
        x=torch.zeros(1,3,2,2);y=x.clone()
        y[0,0,0,0]=.03
        hidden=torch.zeros(1,1,2,2,dtype=torch.bool);hidden[0,0,1,1]=True
        region=affected_region(x,y,hidden,'private')
        self.assertTrue(region[0,0,0,0]);self.assertTrue(region[0,0,1,1])
        self.assertEqual(int(region.sum()),2)
        self.assertFalse(affected_region(x,x,torch.zeros_like(hidden),'private').any())

    def test_model_and_backward(self):
        torch.manual_seed(42);original=PairedCMX('CONS','CMX_B0',3,64,32).eval()
        x=torch.rand(2,3,64,64);x[:,2]=1;labels=torch.randint(3,(2,64,64))
        for method in runner.METHODS:
            model=RefinedCMX(method,'CMX_B0',3,64,32).eval()
            model.load_state_dict(original.state_dict())
            with torch.no_grad():
                torch.testing.assert_close(model(x)['logits'],original(x)['logits'],rtol=0,atol=0)
            stats=backward_objective(model,x,labels,'private',method,5,42,
                                     torch.amp.GradScaler('cuda',enabled=False),1.)
            self.assertGreater(stats['ce_secondary'],0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertLessEqual(stats['confidence_coverage'],stats['eligible_coverage'])

    def test_schedule_four_groups_no_old_control_training(self):
        args=SimpleNamespace(output_dir=scratch(),budget_hours=3,reference_results=[
            dict(method='CONS',elapsed_minutes=90)])
        calls=[]
        def train(a,d,m,s,p):
            calls.append((d,m,s));return dict(dataset=d,method=m,seed=s)
        with contextlib.ExitStack() as stack:
            for key,value in [('preflight',lambda a:{}),('train_group',train),('summarize',lambda *a:None)]:
                stack.enter_context(patch.object(runner,key,value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()));runner.execute(args)
        self.assertEqual(calls,[(d,m,42) for m in ('LOCAL','PIXEL') for d in ('nyuv2','private')])
        complete=runner.read(args.output_dir/'complete.json')
        self.assertEqual(complete['training_groups'],4)
        self.assertEqual(complete['epochs_per_group'],20)
        self.assertEqual(complete['public_condition_evaluations'],44)
        self.assertFalse(complete['final_test_access'])

    def test_comparison_uses_seed42_not_three_seed_average(self):
        def row(m,v):
            return dict(dataset='nyuv2',method=m,seed=42,clean_miou=v,corruption_mean_miou=v,
                        photon_proxy_heavy_miou=v,joint_heavy_miou=v)
        args=SimpleNamespace(output_dir=scratch(),reference_results=[row('PAIR',.5),row('CONS',.49)])
        with contextlib.redirect_stdout(io.StringIO()):
            runner.summarize(args,[row('LOCAL',.51)])
        comparisons=runner.read(args.output_dir/'comparison.json')['paired_comparisons']
        self.assertEqual(len(comparisons),2)
        self.assertAlmostEqual(comparisons[0]['delta_pp']['clean_miou'],1.)
        self.assertAlmostEqual(comparisons[1]['delta_pp']['clean_miou'],2.)

    def test_resume_with_historical_pair_curves(self):
        root=scratch();reference=scratch()
        args=SimpleNamespace(output_dir=root,reference_dir=reference,private_dir=root,nyuv2_dir=root,device='cpu')
        source=dict(model=ToyModel('CONS').state_dict(),args={'model':'CMX_B2','decoder_dim':32})
        metric=dict(regions={'all':{'miou':.4}},degradation_audit=[])
        for f,v in runner.CONDITIONS:
            runner.save(reference/f'nyuv2/PAIR_seed42/curves/{f}_{v}.json',metric)
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
                    runner.train_group(args,'nyuv2','LOCAL',42,{'test':1})
            with self.assertRaisesRegex(ValueError,'protocol/group'):
                runner.train_group(args,'nyuv2','LOCAL',42,{'test':2})
            for method in runner.METHODS:
                for dataset in ('nyuv2','private'):
                    result=runner.train_group(args,dataset,method,42,{'test':1})
                    self.assertEqual(result,runner.train_group(args,dataset,method,42,{'test':1}))


if __name__=='__main__':
    unittest.main()
