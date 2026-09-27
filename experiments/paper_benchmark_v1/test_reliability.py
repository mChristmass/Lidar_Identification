"""Small synthetic CPU tests, not dataset training."""
import unittest
import contextlib
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np  # Load before torch for the local runtime's OpenMP setup.
import torch
from .model import CMXPaper
from .reliability_model import ReliabilityCMX
from .train_mixed_control import training_condition
from . import train_reliability as runner


class ToyData(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.epoch = 0

    def __len__(self):
        return 3  # Exercise the partial batch and accumulation window.

    def __getitem__(self, index):
        return torch.full((3, 4, 4), (index + self.epoch) / 25), torch.full((4, 4), index, dtype=torch.long)


class ToyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, 3, 1)
        self.depth_strength = torch.nn.Parameter(torch.zeros(1))

    def load_baseline(self, state):
        self.load_state_dict(state)

    def project_parameters(self):
        with torch.no_grad():
            self.depth_strength.clamp_(0, 1)

    def forward(self, x):
        return {'logits': self.backbone(x) * (1 + self.depth_strength)}


class ReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.base = CMXPaper('CMX_B0', 3, 64, 32).eval()
        cls.inputs = torch.rand(2, 3, 64, 64)
        cls.inputs[:, 2] = (cls.inputs[:, 2] > .5).float()

    def make_model(self, experiment):
        model = ReliabilityCMX(experiment, 'CMX_B0', 3, 64, 32).eval()
        model.load_baseline(self.base.state_dict())
        return model

    def test_identity_and_checkpoint_roundtrip(self):
        with torch.no_grad():
            expected = self.base(self.inputs)['logits']
            for name in ('R1', 'R2', 'R3'):
                model = self.make_model(name)
                actual = model(self.inputs)['logits']
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                restored = self.make_model(name)
                restored.load_state_dict(model.state_dict(), strict=True)
                torch.testing.assert_close(restored(self.inputs)['logits'], actual, rtol=0, atol=0)

    def test_gradients_and_extreme_validity(self):
        for name in ('R1', 'R2', 'R3'):
            model = self.make_model(name)
            if name != 'R1':
                with torch.no_grad():
                    model.depth_strength.fill_(.5)
                    if name == 'R3':
                        model.intensity_strength.fill_(.5)
            model(self.inputs)['logits'].square().mean().backward()
            for key, parameter in model.named_parameters():
                if not key.startswith(('backbone.', 'decode_head.')):
                    self.assertIsNotNone(parameter.grad, key)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), key)
            key = {'R1': 'validity_projection.weight', 'R2': 'depth_strength',
                   'R3': 'reliability.2.weight'}[name]
            self.assertGreater(dict(model.named_parameters())[key].grad.abs().sum().item(), 0)
            with torch.no_grad():
                for valid in (0., 1.):
                    x = self.inputs.clone()
                    x[:, 2] = valid
                    if valid == 0:
                        x[:, 1] = 0
                    self.assertTrue(torch.isfinite(model(x)['logits']).all())
                if name != 'R1':
                    model.depth_strength.copy_(torch.tensor([-1., 2., .5, 0.]))
                    model.project_parameters()
                    torch.testing.assert_close(model.depth_strength, torch.tensor([0., 1., .5, 0.]))

    def test_corruptions_are_paired(self):
        first = [training_condition(42, 1, str(i), 'mixed') for i in range(30)]
        self.assertEqual(first, [training_condition(42, 1, str(i), 'mixed') for i in range(30)])
        self.assertNotEqual(first, [training_condition(42, 2, str(i), 'mixed') for i in range(30)])

    def test_interrupted_resume_matches_continuous(self):
        # Only a tiny synthetic linear model; no images or real training involved.
        torch.manual_seed(11)
        source = {'args': {'decoder_dim': 32}, 'model': ToyModel().state_dict()}
        references = {f'{f}_{s}': {'degradation_audit': [], 'regions': {'all': {'miou': .4}}}
                      for f, s in runner.CONDITIONS}
        protocol = {'checkpoint_sha256': 'synthetic'}
        def metrics(*args):
            return {'degradation_audit': [], 'regions': {'all': {'miou': .4}}}
        scratch = Path(__file__).resolve().parents[2] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='reliability_test_', dir=scratch) as folder, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'ReliabilityCMX', ToyModel))
            stack.enter_context(patch.object(runner, 'TrainingData', ToyData))
            stack.enter_context(patch.object(runner, 'DiagnosticData', lambda *args: None))
            stack.enter_context(patch.object(runner, 'regional', metrics))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            args = SimpleNamespace(output_dir=Path(folder)/'full', nyuv2_dir=Path(folder), device='cpu', module_lr=1e-3)
            args.output_dir.mkdir()
            runner.train_group(args, 'R2', source, protocol, references)
            full = torch.load(args.output_dir/'R2/last.pth', weights_only=False)
            args.output_dir = Path(folder)/'resumed'
            args.output_dir.mkdir()
            original = runner.atomic_checkpoint
            def interrupt(path, state):
                original(path, state)
                raise RuntimeError('simulated interruption after checkpoint')
            with patch.object(runner, 'atomic_checkpoint', interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    runner.train_group(args, 'R2', source, protocol, references)
            runner.train_group(args, 'R2', source, protocol, references)
            resumed = torch.load(args.output_dir/'R2/last.pth', weights_only=False)
            self.assertEqual(full['history'], resumed['history'])
            for key in full['model']:
                torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
            with patch.object(runner, 'ReliabilityCMX', side_effect=AssertionError('must reuse')):
                result = runner.train_group(args, 'R2', source, protocol, references)
                self.assertEqual(len(result['scores']), 22)


if __name__ == '__main__':
    unittest.main()
