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
from .context_model import ContextCMX, ContextPyramid, task_loss
from .task_control_model import TaskControlCMX
from .model import BACKBONES
from . import run_context_window as runner


def scratch():
    root=Path(__file__).resolve().parents[2]/'tmp'
    root.mkdir(exist_ok=True)
    path = root/('context_test_'+uuid.uuid4().hex)
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


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)


    def test_all_models_start_as_baseline_and_receive_gradients(self):
        base=CMXPaper('CMX_B0',3,64,32).eval()
        x=torch.rand(2,3,64,64)
        x[:,2]=(x[:,2]>.2).float()
        with torch.no_grad():
            target=base(x)['logits']
        for method in runner.METHODS:
            model=ContextCMX(method,'CMX_B0',3,64,32).eval()
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
            stack.enter_context(patch.object(runner,'ContextCMX',ToyModel))
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
            for method,gain in (('B0',0),('F_CTX',.01)):
                rows.append(dict(dataset='nyuv2',method=method,seed=seed,clean_miou=.5+gain,
                   corruption_mean_miou=.4+gain,photon_proxy_heavy_miou=.3+gain,joint_heavy_miou=.2+gain))
                rows.append(dict(dataset='private',method=method,seed=seed,metrics=dict(
                   foreground_iou=.7+gain,boundary_f1_1px=.8+gain,background_fp_frame_rate=.1-gain,
                   tiny_component_recall_iou50_area80=None)))
        with contextlib.redirect_stdout(io.StringIO()):
            runner.summarize(args,rows)
        summaries=runner.read(args.output_dir/'comparison.json')['summaries']
        row=next(r for r in summaries if r['dataset']=='private' and r['method']=='F_CTX')
        self.assertAlmostEqual(row['foreground_iou']['paired_delta_pp'],1)
        self.assertAlmostEqual(row['background_fp_frame_rate']['paired_delta_pp'],-1)
        self.assertIsNone(row['tiny_component_recall_iou50_area80']['mean'])







    def test_schedule_is_eight_groups_one_seed(self):
        args=SimpleNamespace(output_dir=scratch(),budget_hours=3.,device='cpu')
        calls=[]
        def train(a,dataset,method,seed,protocol):
            calls.append((dataset,method,seed))
            return dict(dataset=dataset,method=method,seed=seed,elapsed_minutes=20)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner,'preflight',return_value={}))
            stack.enter_context(patch.object(runner,'audit_inputs',lambda a: None))
            stack.enter_context(patch.object(runner,'train_group',train))
            stack.enter_context(patch.object(runner,'summarize',lambda *a: None))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            runner.execute(args)
        self.assertEqual(calls,[(d,m,42) for m in runner.METHODS for d in ('nyuv2','private')])
        result=runner.read(args.output_dir/'complete.json')
        self.assertEqual(result['training_groups'],8)
        self.assertEqual(result['epochs_per_group'],10)
        self.assertEqual(result['public_condition_evaluations'],88)
        self.assertFalse(result['final_test_access'])

    def test_raw_control_unchanged_with_learned_weights(self):
        previous=TaskControlCMX('S_RAW','CMX_B0',3,64,32).eval()
        with torch.no_grad():
            for layer in previous.adapters['exchange']:
                layer.out.weight.normal_(std=.01)
        current=ContextCMX('S_RAW','CMX_B0',3,64,32).eval()
        current.load_state_dict(previous.state_dict())
        x=torch.rand(2,3,64,64)
        with torch.no_grad():
            torch.testing.assert_close(previous(x)['logits'],current(x)['logits'],rtol=0,atol=0)

    def test_context_learns_and_switch_is_restorable(self):
        x=torch.rand(2,3,64,64)
        original=x.clone()
        for method in ('F_CTX','SF_CTX'):
            model=ContextCMX(method,'CMX_B0',3,64,32).eval()
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            labels=torch.zeros(2,64,64,dtype=torch.long)
            task_loss(model(x),labels).backward()
            opt.step();opt.zero_grad()
            task_loss(model(x),labels).backward()
            grad=model.adapters['context'].project[-1].weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(),0)
            with torch.no_grad():
                model.audit_enabled=True
                on=model(x)['logits']
                self.assertEqual(sum(str(r['stage']).startswith('context_') for r in model.audit_rows),4)
                model.context_enabled=False
                off=model(x)['logits']
                self.assertGreater((on-off).abs().sum().item(),0)
                model.context_enabled=True
                torch.testing.assert_close(on,model(x)['logits'],rtol=0,atol=0)
                restored=ContextCMX(method,'CMX_B0',3,64,32).eval()
                restored.load_state_dict(model.state_dict())
                torch.testing.assert_close(on,restored(x)['logits'],rtol=0,atol=0)
            torch.testing.assert_close(x,original,rtol=0,atol=0)

    def test_coarse_context_reaches_fine_scale(self):
        layer=ContextPyramid([8,16,24,32])
        features=[torch.rand(2,c,h,h,requires_grad=True) for c,h in [(8,16),(16,8),(24,4),(32,2)]]
        with torch.no_grad():
            layer.out[0].weight.fill_(.01)
        changed,deltas=layer(features)
        self.assertEqual([x.shape for x in changed],[x.shape for x in features])
        deltas[0].sum().backward()
        self.assertGreater(features[-1].grad.abs().sum().item(),0)

    def test_factorial_comparison_uses_current_ten_epoch_controls(self):
        args=SimpleNamespace(output_dir=scratch())
        rows=[dict(dataset='nyuv2',method=m,seed=42,clean_miou=v,
            corruption_mean_miou=v,photon_proxy_heavy_miou=v,joint_heavy_miou=v)
              for m,v in [('B0',.5),('S_RAW',.51),('F_CTX',.52),('SF_CTX',.54)]]
        with contextlib.redirect_stdout(io.StringIO()):
            runner.summarize(args,rows)
        result=runner.read(args.output_dir/'comparison.json')
        pair={r['comparison']:r for r in result['paired_comparisons']}
        self.assertAlmostEqual(pair['F_CTX_minus_B0']['delta_pp']['clean_miou'],2.)
        self.assertAlmostEqual(pair['SF_CTX_minus_S_RAW']['delta_pp']['clean_miou'],3.)
        self.assertAlmostEqual(pair['factorial_interaction']['delta_pp']['clean_miou'],1.)
        self.assertTrue(all(r['clean_miou']['std'] is None for r in result['summaries']))

    def test_context_initialization_shared_across_factorial_candidates(self):
        torch.manual_seed(42)
        single=ContextCMX('F_CTX','CMX_B0',3,64,32)
        torch.manual_seed(42)
        combined=ContextCMX('SF_CTX','CMX_B0',3,64,32)
        for key,value in single.adapters['context'].state_dict().items():
            torch.testing.assert_close(value,combined.adapters['context'].state_dict()[key],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
