"""Synthetic CPU checks of fixed teachers, frozen buffers and recovery."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import run_noise_window as runner


class Base(torch.nn.Module):
    def __init__(self, *a, **kw):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))
        self.decode_head = torch.nn.Conv2d(4, 3, 1)

    def forward(self, x):
        return {'logits': self.decode_head(self.backbone(x))}


class Candidate(Base):
    def __init__(self, *a, **kw):
        super().__init__()
        self.restorer = torch.nn.Conv2d(3, 1, 1)
        torch.nn.init.zeros_(self.restorer.weight)
        torch.nn.init.zeros_(self.restorer.bias)
        self.noise_gate = torch.nn.Parameter(torch.zeros(1))

    def load_baseline(self, state):
        self.load_state_dict(state, strict=False)

    def project_parameters(self):
        pass

    def forward(self, x):
        delta = self.restorer(x)
        restored = x[:, :1]+delta*self.noise_gate.sigmoid()
        out = super().forward(torch.cat((restored, x[:, 1:]), 1))
        out.update(restored_intensity=restored, raw_restored_intensity=x[:, :1]+delta,
                   noise_gate_logits=self.noise_gate.expand(len(x), 1, 1, 1))
        return out


class Paired(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.epoch = 0
    def __len__(self):
        return 4
    def __getitem__(self, i):
        x = torch.full((3, 4, 4), .2+.1*i)
        clean = x.clone()
        if i%2:
            clean[0] += .1
        return x, torch.full((4, 4), i%3, dtype=torch.long), clean, torch.tensor(bool(i%2))


class PreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_mode_preserves_buffers_but_allows_input_gradients(self):
        model = Candidate()
        model.backbone.requires_grad_(False)
        model.decode_head.requires_grad_(False)
        runner.candidate_train_mode(model, True)
        before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
        model(torch.rand(2, 3, 4, 4))['logits'].square().mean().backward()
        self.assertGreater(model.restorer.weight.grad.abs().sum().item(), 0)
        for k, v in before.items():
            torch.testing.assert_close(v, model.backbone.state_dict()[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_teacher_frozen_training_and_resume(self):
        scratch = Path(__file__).resolve().parents[2]/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch, prefix='preserve_') as folder, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'NoiseCMX', Candidate))
            stack.enter_context(patch.object(runner, 'CMXPaper', Base))
            stack.enter_context(patch.object(runner, 'PairedIntensityData', Paired))
            stack.enter_context(patch.object(runner, 'EPOCHS', 2))
            stack.enter_context(patch.object(runner, 'DiagnosticData', lambda *a: None))
            row = dict(regions={'all': {'miou': .4}}, degradation_audit=[])
            stack.enter_context(patch.object(runner, 'regional', return_value=row))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            source = dict(args={'decoder_dim': 32}, model=Base().state_dict())
            refs = {f'{f}_{s}': row for f, s in runner.CONDITIONS}
            args = SimpleNamespace(output_dir=Path(folder), nyuv2_dir=Path(folder), device='cpu')
            for name, mode in (('P1', 'teacher'), ('P2', 'frozen')):
                args.group_prefix, args.preservation = name, mode
                result = runner.train_candidate(args, 'S2', 42, source, {'test': 1}, refs)
                state = torch.load(Path(folder)/f'{name}_seed42/last.pth', weights_only=False)
                if mode=='frozen':
                    for key, expected in source['model'].items():
                        torch.testing.assert_close(state['model'][key], expected, rtol=0, atol=0)
                else:
                    self.assertGreater(state['history'][-1]['losses']['preservation'], 0)
                self.assertEqual(result, runner.train_candidate(args, 'S2', 42, source, {'test': 1}, refs))
                # Remove only the synthetic result to exercise checkpoint recovery.
                (Path(folder)/f'{name}_seed42/result.json').unlink()
                self.assertEqual(result, runner.train_candidate(args, 'S2', 42, source, {'test': 1}, refs))


if __name__ == '__main__':
    unittest.main()
