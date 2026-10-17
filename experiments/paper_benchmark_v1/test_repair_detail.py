"""Synthetic CPU validation only; scratch files retained, no research training."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from .repair_detail_model import RepairDetailCMX, build_model, COMBINATIONS, backward_objective
from .broad_window_model import BroadCMX
from .paired_model import PairedCMX
from .test_paired_window import ToyModel,ToyTrain,scratch
from . import run_repair_detail as runner


class RepairDetailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_common_initialization_exact_components_and_identity(self):
        torch.manual_seed(42); pair=PairedCMX('PAIR','CMX_B0',3,64,32).eval()
        rng=torch.get_rng_state()
        torch.manual_seed(42); repair=BroadCMX('R_RES','CMX_B0',3,64,32).eval()
        torch.manual_seed(42); detail=BroadCMX('F_DETAIL','CMX_B0',3,64,32).eval()
        x=torch.rand(2,3,64,64);x[:,2]=1
        with torch.no_grad(): expected=pair(x)['logits']
        for method in COMBINATIONS:
            torch.manual_seed(42); model=build_model(method,'CMX_B0',3,64,32).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for k,v in pair.state_dict().items():
                torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
            for key,source in (('repair',repair),('detail',detail)):
                for k,v in source.adapters['candidate'].state_dict().items():
                    torch.testing.assert_close(v,model.adapters['candidate'][key].state_dict()[k],rtol=0,atol=0)
            with torch.no_grad():torch.testing.assert_close(expected,model(x)['logits'],rtol=0,atol=0)

    def test_standalone_uses_exact_historical_class(self):
        for method in ('R_RES','F_DETAIL'):
            torch.manual_seed(7);old=BroadCMX(method,'CMX_B0',3,64,32)
            torch.manual_seed(7);new=build_model(method,'CMX_B0',3,64,32)
            self.assertIs(type(new),BroadCMX)
            for k,v in old.state_dict().items():
                torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)

    def test_actual_detail_input_routing_and_depth_unchanged(self):
        x=torch.rand(2,3,64,64);x[:,2]=1;before=x.clone()
        for method in COMBINATIONS:
            model=build_model(method,'CMX_B0',3,64,32).eval()
            with torch.no_grad():model.adapters['candidate']['repair'][-1].bias.fill_(.2)
            seen=[]
            hook=model.adapters['candidate']['detail']['stem'].register_forward_pre_hook(
                lambda module,args: seen.append(args[0].detach().clone()))
            with torch.no_grad():model(x)
            hook.remove()
            expected=x.clone()
            if method=='RD_CORRECTED':expected[:,:1]=(expected[:,:1]+.2).clamp(0,1)
            torch.testing.assert_close(seen[0],expected)
            torch.testing.assert_close(x,before,rtol=0,atol=0)

    def test_gradients_both_branches_and_toggle_recovery(self):
        x=torch.rand(2,3,64,64);x[:,2]=1;y=torch.randint(3,(2,64,64))
        for method in COMBINATIONS:
            model=build_model(method,'CMX_B0',3,64,32).train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                stats=backward_objective(model,x,y,'private',method,1,42,
                    torch.amp.GradScaler('cuda',enabled=False),1.)
                self.assertEqual(stats['consistency'],0)
                for branch in ('repair','detail'):
                    params=list(model.adapters['candidate'][branch].parameters())
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in params if p.grad is not None),0)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()
            model.eval();model.candidate_enabled=False
            with torch.no_grad():
                expected=model(x)['logits']
                model.candidate_enabled=True
                model.repair_enabled=model.detail_enabled=False
                torch.testing.assert_close(expected,model(x)['logits'],rtol=0,atol=0)

    def test_fixed_sixteen_group_schedule_and_no_test_access(self):
        args=SimpleNamespace(output_dir=scratch(),budget_hours=16,reference_results=[
            dict(method='PAIR',seed=42,elapsed_minutes=90)])
        calls=[]
        def train(a,d,m,s,p):
            calls.append((d,m,s));return dict(dataset=d,method=m,seed=s,group=f'{d}/{m}_seed{s}')
        with contextlib.ExitStack() as stack:
            for key,value in [('preflight',lambda a:{}),('train_group',train),('summarize',lambda *a:None)]:
                stack.enter_context(patch.object(runner,key,value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()));runner.execute(args)
        expected=[(d,m,s) for s in (42,777,2025) for m in
                  (('R_RES','F_DETAIL','RD_CORRECTED','RD_RAW') if s==2025 else COMBINATIONS)
                  for d in ('nyuv2','private')]
        self.assertEqual(calls,expected)
        complete=runner.read(args.output_dir/'complete.json')
        self.assertEqual(complete['training_groups'],16)
        self.assertEqual(complete['public_condition_evaluations'],176)
        self.assertEqual(complete['private_evaluations'],8)
        self.assertFalse(complete['final_test_access'])

    def test_summary_matches_only_available_control_seeds(self):
        def row(m,s,v):
            return dict(dataset='nyuv2',method=m,seed=s,clean_miou=v,corruption_mean_miou=v,
                photon_proxy_heavy_miou=v,joint_heavy_miou=v)
        args=SimpleNamespace(output_dir=scratch(),reference_results=[
            row('PAIR',42,.5),row('PAIR',2025,.6),row('R_RES',42,.51)])
        with contextlib.redirect_stdout(io.StringIO()):
            runner.summarize(args,[row('RD_RAW',42,.52),row('RD_RAW',2025,.63)])
        sums=runner.read(args.output_dir/'comparison.json')['summaries']
        a=next(r for r in sums if r['method']=='RD_RAW' and r['reference']=='R_RES')
        self.assertEqual(a['seeds'],[42]);self.assertIsNone(a['clean_miou']['std_delta_pp'])
        a=next(r for r in sums if r['method']=='RD_RAW' and r['reference']=='PAIR')
        self.assertEqual(a['seeds'],[42,2025]);self.assertAlmostEqual(a['clean_miou']['mean_delta_pp'],2.5)

    def test_preflight_only_does_not_train(self):
        args=SimpleNamespace(output_dir=scratch(),budget_hours=16,preflight_only=True)
        with patch.object(runner,'preflight',return_value={}),patch.object(runner,'train_group') as train:
            with contextlib.redirect_stdout(io.StringIO()):runner.execute(args)
            train.assert_not_called()
        self.assertFalse((args.output_dir/'complete.json').exists())

    def test_resume_correct_reference_curves_all_methods(self):
        root=scratch();reference=scratch();replication=scratch()
        args=SimpleNamespace(output_dir=root,reference_dir=reference,replication_dir=replication,
            private_dir=root,nyuv2_dir=root,device='cpu')
        source=dict(model=ToyModel('PAIR').state_dict(),args={'model':'CMX_B2','decoder_dim':32})
        metric=dict(regions={'all':{'miou':.4}},degradation_audit=[])
        for seed in (42,777,2025):
            parent=reference if seed==42 else replication
            for f,v in runner.CONDITIONS:
                runner.save(parent/f'nyuv2/PAIR_seed{seed}/curves/{f}_{v}.json',metric)
        with contextlib.ExitStack() as stack:
            for key,value in [('load_source',lambda *a:source),('PairedCMX',ToyModel),('TrainingData',ToyTrain),
                ('PrivateTrain',ToyTrain),('PrivatePaperDataset',lambda *a:[]),('evaluate',lambda *a:{'foreground_iou':.4}),
                ('private_details',lambda *a:{'rows':[]}),('DiagnosticData',lambda *a:None),
                ('regional',lambda *a:metric),('EPOCHS',2)]:
                stack.enter_context(patch.object(runner,key,value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            checkpoint=runner.atomic_checkpoint
            def stop(path,state):
                checkpoint(path,state);raise RuntimeError('synthetic interruption')
            with patch.object(runner,'atomic_checkpoint',stop):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    runner.train_group(args,'nyuv2','RD_RAW',42,{'test':1})
            with self.assertRaisesRegex(ValueError,'protocol/group'):
                runner.train_group(args,'nyuv2','RD_RAW',42,{'test':2})
            for seed,method in ((42,'RD_RAW'),(777,'RD_CORRECTED'),(2025,'R_RES'),(2025,'F_DETAIL')):
                for dataset in ('nyuv2','private'):
                    result=runner.train_group(args,dataset,method,seed,{'test':1})
                    self.assertEqual(result,runner.train_group(args,dataset,method,seed,{'test':1}))


if __name__=='__main__':
    unittest.main()
