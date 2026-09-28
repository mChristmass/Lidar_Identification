"""Focused synthetic tests; no real images or training jobs."""
import unittest
import contextlib
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np  # Initialize before torch for the local Windows runtime.
import torch

from .fusion_pilot_model import FusionPilotCMX
from .model import CMXPaper
from .run_fusion_window import select_confirmation
from . import run_fusion_window as runner
from .test_reliability import ToyData, ToyModel


class FusionWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        cls.base = CMXPaper('CMX_B0', 3, 64, 32).eval()
        cls.inputs = torch.rand(2, 3, 64, 64)
        cls.inputs[:, 2] = 1
        cls.inputs[:, 2, 8:40, 8:40] = 0

    def model(self, name):
        model = FusionPilotCMX(name, 'CMX_B0', 3, 64, 32).eval()
        model.load_baseline(self.base.state_dict())
        return model

    def test_exact_initialization_and_state_dict(self):
        with torch.no_grad():
            expected = self.base(self.inputs)['logits']
            for name in ('F1', 'F2'):
                model = self.model(name)
                actual = model(self.inputs)['logits']
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                restored = self.model(name)
                restored.load_state_dict(model.state_dict(), strict=True)
                torch.testing.assert_close(restored(self.inputs)['logits'], actual, rtol=0, atol=0)

    def test_fusion_and_propagation_have_gradients(self):
        for name in ('F1', 'F2'):
            model = self.model(name)
            with torch.no_grad():
                model.cross_gate.fill_(.4)
                if name == 'F2':
                    model.propagation_gate.fill_(.4)
            model(self.inputs)['logits'].square().mean().backward()
            self.assertGreater(model.cross_gate.grad.abs().sum().item(), 0)
            if name == 'F2':
                self.assertGreater(model.propagation_gate.grad.abs().sum().item(), 0)
            with torch.no_grad():
                for validity in (0., 1.):
                    x = self.inputs.clone()
                    x[:, 2] = validity
                    if validity == 0:
                        x[:, 1] = 0
                    self.assertTrue(torch.isfinite(model(x)['logits']).all())
                model.cross_gate.copy_(torch.tensor([-1., 2., .5, 0.]))
                model.project_parameters()
                torch.testing.assert_close(model.cross_gate, torch.tensor([0., 1., .5, 0.]))

    def test_confirmation_gate(self):
        ref = {'clean_medium': {'regions': {'all': {'miou': .55}}}}
        def row(method, mean_gain, joint_gain, clean):
            return {'method': method, 'corruption_mean_gain_pp': mean_gain,
                    'joint_heavy_gain_pp': joint_gain, 'clean_miou': clean}
        self.assertIsNone(select_confirmation([row('F1', .49, .99, .55)], ref))
        self.assertIsNone(select_confirmation([row('F1', 1.5, 2., .54)], ref))
        selected = select_confirmation([row('F1', .3, 1.1, .55),
                                        row('F2', .7, .4, .55)], ref)
        self.assertEqual(selected['method'], 'F2')

    def test_training_runner_resumes_and_handles_r0(self):
        torch.manual_seed(9)
        source = {'args': {'decoder_dim': 32}, 'model': ToyModel().state_dict()}
        refs = {f'{f}_{s}': {'degradation_audit': [],
                            'regions': {'all': {'miou': .4}}}
                for f, s in runner.CONDITIONS}
        def metrics(*args):
            return {'degradation_audit': [], 'regions': {'all': {'miou': .4}}}
        scratch = Path(__file__).resolve().parents[2] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='fusion_test_', dir=scratch) as folder, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'FusionPilotCMX', ToyModel))
            stack.enter_context(patch.object(runner, 'CMXPaper', ToyModel))
            stack.enter_context(patch.object(runner, 'TrainingData', ToyData))
            stack.enter_context(patch.object(runner, 'DiagnosticData', lambda *args: None))
            stack.enter_context(patch.object(runner, 'regional', metrics))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            args = SimpleNamespace(output_dir=Path(folder), nyuv2_dir=Path(folder), device='cpu')
            protocol = {'purpose': 'synthetic'}
            r0 = runner.train_job(args, 'R0', 777, source, protocol, None)
            self.assertEqual(len(r0['scores']), 22)
            original = runner.atomic_checkpoint
            def interrupt(path, state):
                original(path, state)
                raise RuntimeError('synthetic interruption')
            with patch.object(runner, 'atomic_checkpoint', interrupt):
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    runner.train_job(args, 'F1', 42, source, protocol, refs)
            resumed = runner.train_job(args, 'F1', 42, source, protocol, refs)
            self.assertEqual(len(resumed['scores']), 22)
            self.assertEqual(resumed['corruption_mean_gain_pp'], 0.)
            self.assertEqual(runner.train_job(args, 'F1', 42, source, protocol, refs), resumed)

    def test_preflight_freezes_reference_and_manifests(self):
        scratch = Path(__file__).resolve().parents[2] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='fusion_preflight_', dir=scratch) as folder:
            root = Path(folder)
            phase1, nyuv2, reference, output = [root / name for name in
                                                  ('phase1', 'nyuv2', 'reference', 'output')]
            for seed in (42, 777):
                path = phase1 / f'seed_{seed}' / 'best.pth'
                path.parent.mkdir(parents=True)
                torch.save({'model': {}}, path)
            manifest = nyuv2 / 'manifests'
            manifest.mkdir(parents=True)
            for part in ('train', 'dev'):
                (manifest / f'{part}.jsonl').write_text('{}\n', encoding='utf-8')
            (reference / 'mixed' / 'curves').mkdir(parents=True)
            torch.save({'epoch': 20, 'history': [{} for _ in range(20)]},
                       reference / 'mixed' / 'last.pth')
            runner.save(reference / 'complete.json',
                        {'training_runs': 2, 'epochs_each': 20, 'evaluations': 44})
            runner.save(reference / 'protocol.json',
                        {'checkpoint_sha256': runner.digest(phase1/'seed_42'/'best.pth'),
                         'train_manifest_sha256': runner.digest(manifest/'train.jsonl'),
                         'dev_manifest_sha256': runner.digest(manifest/'dev.jsonl')})
            for family, severity in runner.CONDITIONS:
                runner.save(reference/'mixed'/'curves'/f'{family}_{severity}.json',
                            {'regime': 'mixed', 'epoch': 20, 'regions': {'all': {'miou': .4}}})
            output.mkdir()
            args = SimpleNamespace(phase1_dir=phase1, nyuv2_dir=nyuv2,
                                   reference_dir=reference, output_dir=output, budget_hours=4.5)
            protocol, curves = runner.preflight(args)
            self.assertEqual(len(curves), 22)
            self.assertEqual(protocol, runner.preflight(args)[0])
            (manifest/'dev.jsonl').write_text('changed\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                runner.preflight(args)


if __name__ == '__main__':
    unittest.main()
