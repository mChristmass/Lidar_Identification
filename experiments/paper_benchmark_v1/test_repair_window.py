"""CPU-only synthetic checks; no research training or final-test access."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import run_repair_window as window
from . import run_noise_window as runner
from .model import CMXPaper
from .noise_model import NoiseCMX
from .test_noise_window import ToyPairedData, ToyNoiseModel


class ToySelective(ToyNoiseModel):
    def forward(self, x):
        out = super().forward(x)
        out['raw_restored_intensity'] = out['restored_intensity']
        out['noise_gate_logits'] = self.fusion_strength.expand(len(x), 1, 1, 1)
        return out


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_gradients_and_frontend_transfer(self):
        base = CMXPaper('CMX_B0', 3, 64, 32).eval()
        x = torch.rand(2, 3, 64, 64)
        for method in ('S1', 'S2'):
            model = NoiseCMX(method, 'CMX_B0', 3, 64, 32).eval()
            model.load_baseline(base.state_dict())
            with torch.no_grad():
                torch.testing.assert_close(model(x)['logits'], base(x)['logits'], rtol=0, atol=0)
            target = x.clone()
            target[0, 0] = (target[0, 0] + .1).clamp(0, 1)
            reconstruction, gate = runner.selective_loss(model(x), target,
                                                         torch.tensor([True, False]), method)
            (5*reconstruction+.1*gate).backward()
            self.assertGreater(model.restorer[-1].weight.grad.abs().sum().item(), 0)
            if method == 'S2':
                self.assertGreater(model.noise_gate[-1].bias.grad.abs().sum().item(), 0)
            private = NoiseCMX(method, 'CMX_B0', 2, 64, 32)
            before = private.decode_head.state_dict()
            window.load_frontend(private, model.state_dict())
            for k, v in before.items():
                torch.testing.assert_close(v, private.decode_head.state_dict()[k], rtol=0, atol=0)
            torch.testing.assert_close(private.restorer[-1].weight, model.restorer[-1].weight)

    def test_inactive_identity_weight_and_gate(self):
        pred = {'restored_intensity': torch.ones(2, 1, 2, 2),
                'raw_restored_intensity': torch.ones(2, 1, 2, 2),
                'noise_gate_logits': torch.zeros(2, 1, 1, 1)}
        loss, _ = runner.selective_loss(pred, torch.zeros(2, 3, 2, 2),
                                       torch.tensor([False, False]), 'S1')
        self.assertEqual(loss.item(), 1.)
        result = dict(method='S1', corruption_mean_gain_pp=.3,
                      scores=[dict(condition=f'{f}_{s}', gain_pp=(0 if f=='clean' else 1))
                              for f, s in runner.CONDITIONS])
        self.assertIsNotNone(window.choose([result]))
        result['scores'][0]['gain_pp'] = -.16
        self.assertIsNone(window.choose([result]))

    def test_training_new_branches_resume_and_stale_checkpoint(self):
        scratch = Path(__file__).resolve().parents[2] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch, prefix='repair_') as folder, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'NoiseCMX', ToySelective))
            stack.enter_context(patch.object(runner, 'PairedIntensityData', ToyPairedData))
            stack.enter_context(patch.object(runner, 'DiagnosticData', lambda *a: None))
            stack.enter_context(patch.object(runner, 'regional', lambda *a: dict(
                degradation_audit=[], regions={'all': {'miou': .4}})))
            stack.enter_context(patch.object(runner, 'EPOCHS', 2))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            args = SimpleNamespace(output_dir=Path(folder), nyuv2_dir=Path(folder), device='cpu')
            source = dict(args={'decoder_dim': 32}, model=ToySelective().state_dict())
            refs = {f'{f}_{s}': dict(degradation_audit=[], regions={'all': {'miou': .4}})
                    for f, s in runner.CONDITIONS}
            for method in ('S1', 'S2'):
                result = runner.train_candidate(args, method, 42, source, {'test': 1}, refs)
                self.assertEqual(len(result['scores']), 22)
                self.assertEqual(result, runner.train_candidate(args, method, 42, source, {'test': 1}, refs))
            path = Path(folder)/'S2_seed42/last.pth'
            state = torch.load(path, weights_only=False)
            state['epoch'] = 1
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                runner.train_candidate(args, 'S2', 42, source, {'test': 1}, refs)

    def test_window_scheduling(self):
        scratch = Path(__file__).resolve().parents[2] / 'tmp'
        scratch.mkdir(exist_ok=True)
        for passes, budget, expected in ((True, 8, 7), (False, 8, 3), (True, 5, 3)):
            with tempfile.TemporaryDirectory(dir=scratch, prefix='repair_schedule_') as folder, contextlib.ExitStack() as stack:
                root = Path(folder)
                args = SimpleNamespace(output_dir=root, reference_dir=root, noise_dir=root,
                                       device='cpu', budget_hours=budget)
                def result(method='R0', seed=42):
                    return dict(method=method, seed=seed, clean_miou=.5,
                                corruption_mean_miou=.5, corruption_mean_gain_pp=.3 if passes else 0,
                                scores=[dict(condition=f'{f}_{s}', miou=.5,
                                             gain_pp=0 if f=='clean' else 1) for f, s in runner.CONDITIONS])
                calls = []
                def train(args, method, seed, *unused):
                    calls.append((method, seed))
                    return result(method, seed)
                stack.enter_context(patch.object(window, 'preflight', lambda a: {}))
                stack.enter_context(patch.object(window, 'checked_reference', lambda p: (result(), {})))
                stack.enter_context(patch.object(noise_module := window.noise, 'load_source', lambda *a: {'args': {'decoder_dim': 32}}))
                stack.enter_context(patch.object(noise_module, 'train_candidate', train))
                stack.enter_context(patch.object(window, 'train_job', train))
                stack.enter_context(patch.object(window, 'private_diagnostic', lambda a: {}))
                stack.enter_context(patch.object(window, 'frontend_audit', lambda *a: {}))
                stack.enter_context(patch.object(window, 'NoiseCMX', lambda *a, **k: ToySelective()))
                stack.enter_context(patch.object(window.torch, 'load', lambda *a, **k: {'model': ToySelective().state_dict()}))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                # Model the mandatory screen consuming two hours.
                window.execute(args, window.time.time()-120*60)
                self.assertEqual(len(calls), expected)
                self.assertEqual(window.read(root/'complete.json')['training_runs'], expected)

    def test_preflight_and_changed_inputs(self):
        scratch = Path(__file__).resolve().parents[2] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch, prefix='repair_preflight_') as folder, contextlib.ExitStack() as stack:
            root = Path(folder)
            args = SimpleNamespace(output_dir=root/'out', reference_dir=root/'ref',
                                   noise_dir=root/'noise', private_dir=root/'private',
                                   private_checkpoint=root/'private.pth', nyuv2_dir=root/'nyu',
                                   phase1_dir=root/'sources', budget_hours=8)
            args.output_dir.mkdir()
            split = args.private_dir/'paper_split_v1'
            split.mkdir(parents=True)
            np.save(split/'train_indices.npy', np.arange(1, 971))
            np.save(split/'dev_indices.npy', np.arange(971, 1108))
            window.save(args.reference_dir/'protocol.json', dict(checkpoint_sha256='same',
                        train_manifest_sha256='same', dev_manifest_sha256='same'))
            window.save(args.noise_dir/'protocol.json', dict(
                manifest_sha256={'train': 'same', 'dev': 'same'},
                checkpoint_sha256={'42': 'same', '777': 'same'}, reference_sha256={'clean_medium': 'same'}))
            stack.enter_context(patch.object(window, 'digest', lambda p: 'same'))
            stack.enter_context(patch.object(window, 'checked_reference', lambda p: ({}, {'clean_medium': {}})))
            first = window.preflight(args)
            self.assertEqual(first, window.preflight(args))
            with patch.object(window, 'digest', lambda p: 'changed' if p.name=='train.jsonl' else 'same'):
                with self.assertRaisesRegex(ValueError, 'manifests changed'):
                    window.preflight(args)
            args.budget_hours = 9
            with self.assertRaisesRegex(ValueError, 'Protocol changed'):
                window.preflight(args)


if __name__ == '__main__':
    unittest.main()
