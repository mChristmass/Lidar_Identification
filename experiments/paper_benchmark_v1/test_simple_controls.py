"""Synthetic CPU tests only. Scratch files retained; no research training."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from .simple_control_model import SimpleControlCMX,METHODS,backward_objective,count_parameters,plain_count
from .broad_window_model import BroadCMX
from .paired_model import PairedCMX
from .test_paired_window import ToyModel,ToyTrain,scratch
from . import run_simple_controls as runner


class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_initialization_source_and_parameter_control(self):
        for classes in (2,40):
            torch.manual_seed(42);base=PairedCMX('PAIR','CMX_B0',classes,64,32)
            rng=torch.get_rng_state()
            for method in METHODS:
                torch.manual_seed(42);model=SimpleControlCMX(method,'CMX_B0',classes,64,32)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                for k,v in base.state_dict().items():
                    torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
                metadata=model.control_metadata()
                if method=='FIX_SMOOTH':self.assertEqual(metadata['candidate_parameters'],0)
                elif method=='D_CONV':
                    self.assertEqual(count_parameters(model.adapters['candidate']),plain_count(model.plain_width,classes))
                    self.assertLess(abs(metadata['candidate_to_detail_parameter_ratio']-1),.1)
                else:self.assertEqual(metadata['candidate_to_detail_parameter_ratio'],1)
                # Baseline loader accepts only backbone/decode keys; new buffer is not a source requirement.
                model.load_baseline({k:v for k,v in base.state_dict().items() if k.startswith(('backbone.','decode_head.'))})

    def test_fixed_filter_constant_bounds_and_depth_validity(self):
        model=SimpleControlCMX('FIX_SMOOTH','CMX_B0',2,64,32)
        x=torch.rand(2,3,17,23);before=x.clone();out=model.smooth_inputs(x)
        torch.testing.assert_close(x,before,rtol=0,atol=0)
        torch.testing.assert_close(out[:,1:],x[:,1:],rtol=0,atol=0)
        self.assertTrue((out[:,:1]>=0).all() and (out[:,:1]<=1).all())
        x[:,:1]=.37;torch.testing.assert_close(model.smooth_inputs(x),x)
        self.assertFalse(model.smooth_kernel.requires_grad)

    def test_exact_smooth_detail_composition_and_plain_identity(self):
        x=torch.rand(2,3,64,64);x[:,2]=1
        torch.manual_seed(42);old=BroadCMX('F_DETAIL','CMX_B0',3,64,32).eval()
        torch.manual_seed(42);model=SimpleControlCMX('SMOOTH_DETAIL','CMX_B0',3,64,32).eval()
        for k,v in old.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        with torch.no_grad():torch.testing.assert_close(old(model.smooth_inputs(x))['logits'],model(x)['logits'],rtol=0,atol=0)
        torch.manual_seed(42);plain=SimpleControlCMX('D_CONV','CMX_B0',3,64,32).eval()
        torch.manual_seed(42);base=PairedCMX('PAIR','CMX_B0',3,64,32).eval()
        with torch.no_grad():torch.testing.assert_close(plain(x)['logits'],base(x)['logits'],rtol=0,atol=0)

    def test_backward_and_no_repair_loss(self):
        x=torch.rand(2,3,64,64);x[:,2]=1;y=torch.randint(3,(2,64,64))
        for method in METHODS:
            model=SimpleControlCMX(method,'CMX_B0',3,64,32).train()
            stats=backward_objective(model,x,y,'private',method,1,7,torch.amp.GradScaler('cuda',enabled=False),1.)
            self.assertNotIn('correction_penalty',stats)
            self.assertAlmostEqual(stats['loss'],.5*(stats['ce_reference']+stats['ce_secondary']),places=6)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if method!='FIX_SMOOTH':
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.adapters['candidate'].parameters() if p.grad is not None),0)

    def test_twelve_group_matrix(self):
        args=SimpleNamespace(output_dir=scratch(),budget_hours=12,reference_results=[dict(method='PAIR',seed=42,elapsed_minutes=90)])
        calls=[]
        def train(a,d,m,s,p):
            calls.append((d,m,s));return dict(dataset=d,method=m,seed=s,group=f'{d}/{m}_seed{s}')
        with contextlib.ExitStack() as stack:
            for k,v in [('preflight',lambda a:{}),('train_group',train),('summarize',lambda *a:None)]:
                stack.enter_context(patch.object(runner,k,v))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()));runner.execute(args)
        self.assertEqual(calls,[(d,m,s) for s in (42,777) for m in METHODS for d in ('nyuv2','private')])
        complete=runner.read(args.output_dir/'complete.json')
        self.assertEqual(complete['training_groups'],12);self.assertEqual(complete['public_condition_evaluations'],132)
        self.assertFalse(complete['final_test_access'])

    def test_summary_seed_matching(self):
        def row(m,s,v):return dict(dataset='nyuv2',method=m,seed=s,clean_miou=v,corruption_mean_miou=v,photon_proxy_heavy_miou=v,joint_heavy_miou=v)
        args=SimpleNamespace(output_dir=scratch(),reference_results=[row('PAIR',42,.5),row('PAIR',777,.6),row('R_RES',42,.51)])
        with contextlib.redirect_stdout(io.StringIO()):runner.summarize(args,[row('FIX_SMOOTH',42,.52),row('FIX_SMOOTH',777,.63)])
        rows=runner.read(args.output_dir/'comparison.json')['summaries']
        a=next(r for r in rows if r['reference']=='R_RES')
        self.assertEqual(a['seeds'],[42]);self.assertIsNone(a['clean_miou']['std_delta_pp'])

    def test_resume_and_seed_specific_curves(self):
        root=scratch();ref=scratch();rep=scratch()
        args=SimpleNamespace(output_dir=root,reference_dir=ref,replication_dir=rep,private_dir=root,nyuv2_dir=root,device='cpu')
        source=dict(model=ToyModel('PAIR').state_dict(),args={'model':'CMX_B2','decoder_dim':32})
        metric=dict(regions={'all':{'miou':.4}},degradation_audit=[])
        for seed in (42,777):
            for f,v in runner.CONDITIONS:runner.save((ref if seed==42 else rep)/f'nyuv2/PAIR_seed{seed}/curves/{f}_{v}.json',metric)
        with contextlib.ExitStack() as stack:
            for k,v in [('load_source',lambda *a:source),('PairedCMX',ToyModel),('TrainingData',ToyTrain),('PrivateTrain',ToyTrain),
                ('PrivatePaperDataset',lambda *a:[]),('evaluate',lambda *a:{'foreground_iou':.4}),('private_details',lambda *a:{'rows':[]}),
                ('DiagnosticData',lambda *a:None),('regional',lambda *a:metric),('EPOCHS',2)]:stack.enter_context(patch.object(runner,k,v))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            checkpoint=runner.atomic_checkpoint
            def stop(path,state):checkpoint(path,state);raise RuntimeError('synthetic interruption')
            with patch.object(runner,'atomic_checkpoint',stop):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):runner.train_group(args,'nyuv2','FIX_SMOOTH',42,{'v':1})
            with self.assertRaisesRegex(ValueError,'protocol/group'):runner.train_group(args,'nyuv2','FIX_SMOOTH',42,{'v':2})
            for seed,method in ((42,'FIX_SMOOTH'),(777,'D_CONV'),(777,'SMOOTH_DETAIL')):
                for dataset in ('nyuv2','private'):
                    result=runner.train_group(args,dataset,method,seed,{'v':1})
                    self.assertEqual(result,runner.train_group(args,dataset,method,seed,{'v':1}))


if __name__=='__main__':unittest.main()
