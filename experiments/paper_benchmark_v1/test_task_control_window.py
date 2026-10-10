"""Synthetic CPU tests. Scratch files retained under ignored tmp; no research training."""
import contextlib
import io
import uuid
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from .model import CMXPaper
from .observation_model import observation_descriptors
from .structure_model import StructureCMX
from .task_control_model import TaskControlCMX, task_loss, CORRECTION_WEIGHT
from . import run_task_control_window as runner


def scratch():
    root=Path(__file__).resolve().parents[2]/'tmp'
    root.mkdir(exist_ok=True)
    path = root/('task_control_test_'+uuid.uuid4().hex)
    path.mkdir()
    return path


class ToyModel(torch.nn.Module):
    def __init__(self, method, *a):
        super().__init__()
        self.backbone=torch.nn.Conv2d(3,2,1)
        self.adapters=torch.nn.Conv2d(3,2,1)
        self.adapters_enabled=True
        self.audit_enabled=False
        self.audit_rows=[]
    def load_baseline(self,state):
        self.load_state_dict(state)
    def forward(self,x):
        return {'logits':self.backbone(x)+(self.adapters(x) if self.adapters_enabled else 0)}


class ToyTrain(torch.utils.data.Dataset):
    def __init__(self,*a):
        self.epoch=0
    def __len__(self):
        return 5
    def __getitem__(self,i):
        return torch.full((3,4,4),.1*i),torch.full((4,4),i%2,dtype=torch.long)


class TaskControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_descriptors_validity_and_constant_fields(self):
        x=torch.ones(2,3,16,16)*.5
        x[:,2]=1
        d=observation_descriptors(x)
        self.assertEqual(d.shape,(2,8,16,16))
        torch.testing.assert_close(d[:,:3],torch.ones_like(d[:,:3]))
        self.assertLess(d[:,3:].abs().max().item(),1e-3)
        x[:,2]=0
        x[:,1]=float('nan')
        d=observation_descriptors(x)
        self.assertTrue(torch.isfinite(d).all())
        self.assertEqual(d[:,:3].abs().sum().item(),0)
        self.assertEqual(d[:,5:].abs().sum().item(),0)

    def test_all_models_start_as_baseline_and_receive_gradients(self):
        base=CMXPaper('CMX_B0',3,64,32).eval()
        x=torch.rand(2,3,64,64)
        x[:,2]=(x[:,2]>.2).float()
        with torch.no_grad():
            target=base(x)['logits']
        for method in runner.METHODS:
            model=TaskControlCMX(method,'CMX_B0',3,64,32).eval()
            model.load_baseline(base.state_dict())
            out=model(x)['logits']
            torch.testing.assert_close(out,target,rtol=0,atol=0)
            if method!='B0':
                out.square().mean().backward()
                self.assertGreater(sum(p.grad.abs().sum().item() for k,p in model.named_parameters() if k.startswith('adapters.') and p.grad is not None),0)

    def test_private_flip_not_model_rng_dependent(self):
        root=scratch()
        (root/'paper_split_v1').mkdir()
        (root/'label').mkdir()
        np.save(root/'intensity.npy',np.arange(64,dtype=np.float32).reshape(1,8,8))
        np.save(root/'depth.npy',np.arange(64,dtype=np.float32).reshape(1,8,8)+1)
        np.save(root/'paper_split_v1/train_indices.npy',np.array([1]))
        Image.fromarray(np.eye(8,dtype=np.uint8)).save(root/'label/1.png')
        data=runner.PrivateTrain(root,42)
        a=data[0]
        torch.rand(100)
        b=data[0]
        torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)

    def test_components_one_to_one(self):
        truth=np.zeros((10,10),bool)
        truth[1:3,1:3]=True;truth[6:8,6:8]=True
        out=runner.component_counts(truth,truth)
        self.assertEqual(out['tiny_detected'],2)
        self.assertEqual(runner.component_counts(np.zeros_like(truth),truth)['detected'],0)

    def test_training_resume_and_protocol_guard(self):
        root=scratch()
        args=SimpleNamespace(output_dir=root,private_dir=root,nyuv2_dir=root,device='cpu')
        source=dict(model=ToyModel('B0').state_dict(),args={'model':'CMX_B2','decoder_dim':32})
        metric=dict(regions={'all':{'miou':.4}},degradation_audit=[])
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner,'load_source',return_value=source))
            stack.enter_context(patch.object(runner,'TaskControlCMX',ToyModel))
            stack.enter_context(patch.object(runner,'TrainingData',ToyTrain))
            stack.enter_context(patch.object(runner,'PrivateTrain',ToyTrain))
            stack.enter_context(patch.object(runner,'PrivatePaperDataset',lambda *a: []))
            stack.enter_context(patch.object(runner,'evaluate',return_value={'foreground_iou':.4}))
            stack.enter_context(patch.object(runner,'private_details',return_value={'rows':[]}))
            stack.enter_context(patch.object(runner,'DiagnosticData',lambda *a: None))
            stack.enter_context(patch.object(runner,'regional',return_value=metric))
            stack.enter_context(patch.object(runner,'EPOCHS',2))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            original=runner.atomic_checkpoint
            def interrupt(path,state):
                original(path,state)
                raise RuntimeError('synthetic stop')
            with patch.object(runner,'atomic_checkpoint',interrupt):
                with self.assertRaisesRegex(RuntimeError,'synthetic stop'):
                    runner.train_group(args,'nyuv2','B0',42,{'test':1})
            with self.assertRaisesRegex(ValueError,'protocol/group'):
                runner.train_group(args,'nyuv2','B0',42,{'test':2})
            for dataset in ('nyuv2','private'):
                for method in runner.METHODS:
                    result=runner.train_group(args,dataset,method,42,{'test':1})
                    self.assertEqual(result,runner.train_group(args,dataset,method,42,{'test':1}))

    def test_dual_domain_schedule_not_score_gated(self):
        for budget,expected in ((8,16),(1,8)):
            args=SimpleNamespace(output_dir=scratch(),budget_hours=budget,device='cpu')
            calls=[]
            elapsed=[0.]
            def train(a,dataset,method,seed,protocol):
                calls.append((dataset,method,seed))
                elapsed[0]+=20.
                return dict(dataset=dataset,method=method,seed=seed,elapsed_minutes=20)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(runner.time,'time',lambda:1000+elapsed[0]*60))
                stack.enter_context(patch.object(runner,'preflight',return_value={}))
                stack.enter_context(patch.object(runner,'audit_inputs',lambda a: None))
                stack.enter_context(patch.object(runner,'train_group',train))
                stack.enter_context(patch.object(runner,'summarize',lambda *a: None))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                runner.execute(args)
            self.assertEqual(len(calls),expected)
            for seed in ({s for _,_,s in calls}):
                self.assertEqual(sum(s==seed and d=='nyuv2' for d,m,s in calls),4)
                self.assertEqual(sum(s==seed and d=='private' for d,m,s in calls),4)

    def test_preflight_rejects_changed_inputs_and_protocol(self):
        root=scratch()
        args=SimpleNamespace(output_dir=root/'out',private_dir=root/'private',nyuv2_dir=root/'nyu',
                             repair_dir=root/'repair',private_source=root/'ps',public_source=root/'ns',budget_hours=8)
        args.output_dir.mkdir()
        split=args.private_dir/'paper_split_v1'
        split.mkdir(parents=True)
        np.save(split/'train_indices.npy',np.arange(1,971))
        np.save(split/'dev_indices.npy',np.arange(971,1108))
        runner.save(args.repair_dir/'protocol.json',dict(private_sha256={'paper_split_v1/dev_indices.npy':'same'},
                    manifests={'train':'same','dev':'same'}))
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner,'load_source',return_value={}))
            stack.enter_context(patch.object(runner,'read_manifest',return_value=[]))
            stack.enter_context(patch.object(runner,'TrainingData',ToyTrain))
            stack.enter_context(patch.object(runner,'digest',return_value='same'))
            original=runner.preflight(args)
            self.assertEqual(original,runner.preflight(args))
            with patch.object(runner,'digest',lambda p: 'changed' if p.name=='train.jsonl' else 'same'):
                with self.assertRaisesRegex(ValueError,'manifest changed'):
                    runner.preflight(args)
            args.budget_hours=7
            with self.assertRaisesRegex(ValueError,'Protocol/input/code changed'):
                runner.preflight(args)

    def test_summary_keeps_domains_and_paired_seeds_separate(self):
        args=SimpleNamespace(output_dir=scratch())
        rows=[]
        for seed in (42,777):
            for method,gain in (('B0',0),('R_TASK',.01)):
                rows.append(dict(dataset='nyuv2',method=method,seed=seed,clean_miou=.5+gain,
                   corruption_mean_miou=.4+gain,photon_proxy_heavy_miou=.3+gain,joint_heavy_miou=.2+gain))
                rows.append(dict(dataset='private',method=method,seed=seed,metrics=dict(
                   foreground_iou=.7+gain,boundary_f1_1px=.8+gain,background_fp_frame_rate=.1-gain,
                   tiny_component_recall_iou50_area80=None)))
        with contextlib.redirect_stdout(io.StringIO()):
            runner.summarize(args,rows)
        summaries=runner.read(args.output_dir/'comparison.json')['summaries']
        row=next(r for r in summaries if r['dataset']=='private' and r['method']=='R_TASK')
        self.assertAlmostEqual(row['foreground_iou']['paired_delta_pp'],1)
        self.assertAlmostEqual(row['background_fp_frame_rate']['paired_delta_pp'],-1)
        self.assertIsNone(row['tiny_component_recall_iou50_area80']['mean'])

    def test_pair_budget_avoids_previous_whole_round_cliff(self):
        rows=[dict(seed=42,method=m,dataset=d,elapsed_minutes=t)
              for m in runner.METHODS for d,t in (('nyuv2',45),('private',8))]
        self.assertTrue(runner.pair_decision(rows,'B0',270.6)['run'])
        self.assertFalse(runner.pair_decision(rows,'S',40)['run'])
        self.assertAlmostEqual(runner.pair_decision(rows,'S',100)['required_minutes'],63.3)

    def test_partial_second_seed_keeps_domain_pairs_and_fixed_order(self):
        args=SimpleNamespace(output_dir=scratch(),budget_hours=5.2,device='cpu')
        calls=[]
        elapsed=[0.]
        def train(a,dataset,method,seed,protocol):
            calls.append((dataset,method,seed))
            elapsed[0]+=20
            return dict(dataset=dataset,method=method,seed=seed,elapsed_minutes=20)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner.time,'time',lambda:1000+60*elapsed[0]))
            stack.enter_context(patch.object(runner,'preflight',return_value={}))
            stack.enter_context(patch.object(runner,'audit_inputs',lambda a: None))
            stack.enter_context(patch.object(runner,'train_group',train))
            stack.enter_context(patch.object(runner,'summarize',lambda *a: None))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            runner.execute(args)
        self.assertEqual(calls[8:],[('nyuv2','B0',777),('private','B0',777),
                                   ('nyuv2','S',777),('private','S',777),
                                   ('nyuv2','S_RAW',777),('private','S_RAW',777)])
        decision=runner.read(args.output_dir/'schedule.json')
        self.assertTrue(decision['S+S_RAW']['run'])
        self.assertFalse(decision['R_TASK']['run'])
        self.assertEqual(runner.read(args.output_dir/'complete.json')['groups_by_seed'],{'42':8,'777':6})

    def test_pair_budget_accounts_for_observed_slowdown(self):
        rows=[dict(seed=42,method=m,dataset=d,elapsed_minutes=20)
              for m in runner.METHODS for d in ('nyuv2','private')]
        rows.append(dict(seed=777,method='B0',dataset='nyuv2',elapsed_minutes=30))
        self.assertAlmostEqual(runner.pair_decision(rows,'R_TASK',100)['estimated_pair_minutes'],60)

    def test_raw_control_parameter_count_and_common_initialization(self):
        torch.manual_seed(42)
        s=TaskControlCMX('S','CMX_B0',3,64,32)
        torch.manual_seed(42)
        raw=TaskControlCMX('S_RAW','CMX_B0',3,64,32)
        count=lambda m:sum(p.numel() for k,p in m.named_parameters() if k.startswith('adapters.'))
        self.assertEqual(count(s)-count(raw),1440)
        for key,value in s.state_dict().items():
            if not key.startswith('adapters.pyramid.0.0.0.'):
                torch.testing.assert_close(value,raw.state_dict()[key],rtol=0,atol=0)
        raw.eval()
        with patch('experiments.paper_benchmark_v1.structure_model.observation_descriptors',
                   side_effect=AssertionError('raw control must not use descriptors')):
            raw(torch.rand(2,3,64,64))

    def test_S_matches_previous_implementation_with_nonzero_adapters(self):
        previous=StructureCMX('S','CMX_B0',3,64,32).eval()
        with torch.no_grad():
            for exchange in previous.adapters['exchange']:
                exchange.out.weight.normal_(std=.01)
        current=TaskControlCMX('S','CMX_B0',3,64,32).eval()
        current.load_state_dict(previous.state_dict())
        x=torch.rand(2,3,64,64)
        with torch.no_grad():
            torch.testing.assert_close(current(x)['logits'],previous(x)['logits'],rtol=0,atol=0)

    def test_internal_branches_learn_after_zero_output_initialization(self):
        x=torch.rand(2,3,64,64)
        x[:,2]=(x[:,2]>.5).float()
        for method in ('S_RAW','S','R_TASK'):
            model=TaskControlCMX(method,'CMX_B0',3,64,32).eval()
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            task_loss(model(x),torch.zeros(2,64,64,dtype=torch.long))[0].backward()
            opt.step();opt.zero_grad()
            task_loss(model(x),torch.zeros(2,64,64,dtype=torch.long))[0].backward()
            key='adapters.0.weight' if method=='R_TASK' else 'adapters.pyramid.0.0.0.weight'
            grad=dict(model.named_parameters())[key].grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(),0)
            model.audit_enabled=True
            with torch.no_grad():
                on=model(x)['logits']
                self.assertTrue(model.audit_rows)
                model.adapters_enabled=False
                off=model(x)['logits']
            self.assertGreater((on-off).abs().sum().item(),0)

    def test_task_loss_needs_only_semantic_labels_and_penalizes_unclipped_delta(self):
        model=TaskControlCMX('R_TASK','CMX_B0',3,64,32).eval()
        with torch.no_grad():
            model.adapters[-1].bias.fill_(2)
        x=torch.full((2,3,64,64),.5)
        seen=[]
        handle=model.backbone.register_forward_pre_hook(lambda m,args:seen.append(args[1].detach().clone()))
        pred=model(x)
        handle.remove()
        torch.testing.assert_close(seen[0],x[:,1:2])
        self.assertAlmostEqual(pred['correction_penalty'].item(),4.)
        total,ce,penalty=task_loss(pred,torch.zeros(2,64,64,dtype=torch.long))
        torch.testing.assert_close(total,ce+CORRECTION_WEIGHT*penalty)
        penalty.backward()
        self.assertGreater(model.adapters[-1].bias.grad.abs().item(),0)
        b=TaskControlCMX('B0','CMX_B0',3,64,32).eval()
        total,ce,penalty=task_loss(b(x),torch.zeros(2,64,64,dtype=torch.long))
        self.assertEqual(penalty.item(),0)
        torch.testing.assert_close(total,ce)

    def test_control_pack_cannot_admit_only_one_structure_method(self):
        rows=[dict(seed=42,method=m,dataset=d,elapsed_minutes=20)
              for m in runner.METHODS for d in ('nyuv2','private')]
        self.assertFalse(runner.pack_decision(rows,('S','S_RAW'),70)['run'])
        self.assertTrue(runner.pack_decision(rows,('S','S_RAW'),100)['run'])
        self.assertAlmostEqual(runner.pack_decision(rows,('S','S_RAW'),100)['required_minutes'],93)

    def test_summary_reports_matched_S_minus_RAW(self):
        args=SimpleNamespace(output_dir=scratch())
        rows=[]
        for seed in (42,777):
            for method,value in [('B0',.5),('S_RAW',.51),('S',.52)]:
                rows.append(dict(dataset='nyuv2',method=method,seed=seed,clean_miou=value,
                    corruption_mean_miou=value,photon_proxy_heavy_miou=value,joint_heavy_miou=value))
        with contextlib.redirect_stdout(io.StringIO()):
            runner.summarize(args,rows)
        row=next(r for r in runner.read(args.output_dir/'comparison.json')['S_minus_S_RAW'] if r['dataset']=='nyuv2')
        self.assertEqual(row['paired_seeds'],[42,777])
        self.assertAlmostEqual(row['mean_delta_pp']['corruption_mean_miou'],1)


if __name__=='__main__':
    unittest.main()
