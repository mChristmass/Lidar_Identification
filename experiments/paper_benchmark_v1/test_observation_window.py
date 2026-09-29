"""Synthetic CPU tests. Scratch files retained under ignored tmp; no research training."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from .model import CMXPaper
from .observation_model import ObservationCMX, observation_descriptors
from . import run_observation_window as runner


def scratch():
    root=Path(__file__).resolve().parents[2]/'tmp'
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='obs_test_',dir=root))


class ToyModel(torch.nn.Module):
    def __init__(self, method, *a):
        super().__init__()
        self.backbone=torch.nn.Conv2d(3,2,1)
        self.adapters=torch.nn.Conv2d(3,2,1)
    def load_baseline(self,state):
        self.load_state_dict(state)
    def forward(self,x):
        return {'logits':self.backbone(x)+self.adapters(x)}


class ToyTrain(torch.utils.data.Dataset):
    def __init__(self,*a):
        self.epoch=0
    def __len__(self):
        return 5
    def __getitem__(self,i):
        return torch.full((3,4,4),.1*i),torch.full((4,4),i%2,dtype=torch.long)


class ObservationTests(unittest.TestCase):
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
            model=ObservationCMX(method,'CMX_B0',3,64,32).eval()
            model.load_baseline(base.state_dict())
            out=model(x)['logits']
            torch.testing.assert_close(out,target,rtol=0,atol=0)
            if method!='B0':
                out.square().mean().backward()
                self.assertGreater(sum(m[-1].weight.grad.abs().sum().item() for m in model.adapters),0)

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
            stack.enter_context(patch.object(runner,'ObservationCMX',ToyModel))
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
            args=SimpleNamespace(output_dir=scratch(),budget_hours=budget)
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
            for method,gain in (('B0',0),('C',.01)):
                rows.append(dict(dataset='nyuv2',method=method,seed=seed,clean_miou=.5+gain,
                   corruption_mean_miou=.4+gain,photon_proxy_heavy_miou=.3+gain,joint_heavy_miou=.2+gain))
                rows.append(dict(dataset='private',method=method,seed=seed,metrics=dict(
                   foreground_iou=.7+gain,boundary_f1_1px=.8+gain,background_fp_frame_rate=.1-gain,
                   tiny_component_recall_iou50_area80=None)))
        with contextlib.redirect_stdout(io.StringIO()):
            runner.summarize(args,rows)
        summaries=runner.read(args.output_dir/'comparison.json')['summaries']
        row=next(r for r in summaries if r['dataset']=='private' and r['method']=='C')
        self.assertAlmostEqual(row['foreground_iou']['paired_delta_pp'],1)
        self.assertAlmostEqual(row['background_fp_frame_rate']['paired_delta_pp'],-1)
        self.assertIsNone(row['tiny_component_recall_iou50_area80']['mean'])


if __name__=='__main__':
    unittest.main()
