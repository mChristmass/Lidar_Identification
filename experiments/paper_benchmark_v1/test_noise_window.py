"""Synthetic CPU checks for paired inputs, model identity and recovery."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np  # Must initialize before torch on this local runtime.
import torch
from PIL import Image

from . import run_noise_window as runner
from .model import CMXPaper
from .noise_model import NoiseCMX
from .paired_intensity import PairedIntensityData
from .train_mixed_control import TrainingData


class ToyPairedData(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.epoch = 0

    def __len__(self):
        return 3

    def __getitem__(self, index):
        x = torch.full((3, 4, 4), (index + self.epoch)/30)
        y = torch.full((4, 4), index, dtype=torch.long)
        clean = x.clone()
        clean[0] += .05
        return x, y, clean, torch.tensor(True)


class ToyNoiseModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, 3, 1)
        self.restorer = torch.nn.Conv2d(1, 1, 1)
        self.fusion_strength = torch.nn.Parameter(torch.zeros(1))

    def load_baseline(self, state):
        self.load_state_dict(state)

    def project_parameters(self):
        with torch.no_grad():
            self.fusion_strength.clamp_(0, 1)

    def forward(self, x):
        restoration = x[:, :1] + .01*self.restorer(x[:, :1])
        return {'logits': self.backbone(x)*(1+self.fusion_strength),
                'restored_intensity': restoration,
                'intensity_confidence': restoration.sigmoid()}


class NoiseWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        cls.base = CMXPaper('CMX_B0', 3, 64, 32).eval()
        cls.inputs = torch.rand(2, 3, 64, 64)
        cls.inputs[:, 2] = (cls.inputs[:, 2] > .5).float()

    def test_initial_identity_and_auxiliary_gradients(self):
        with torch.no_grad():
            expected = self.base(self.inputs)['logits']
            for name in ('C1', 'D1', 'D2'):
                model = NoiseCMX(name, 'CMX_B0', 3, 64, 32).eval()
                model.load_baseline(self.base.state_dict())
                torch.testing.assert_close(model(self.inputs)['logits'], expected,
                                           rtol=0, atol=0)
        model = NoiseCMX('D2', 'CMX_B0', 3, 64, 32)
        model.load_baseline(self.base.state_dict())
        with torch.no_grad():
            model.fusion_strength.fill_(.4)
        result = model(self.inputs)
        (result['logits'].square().mean() +
         result['restored_intensity'].square().mean() +
         result['intensity_confidence'].square().mean()).backward()
        for name in ('fusion_strength', 'restorer.4.weight', 'reliability.2.weight'):
            value = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(value)
            self.assertGreater(value.abs().sum().item(), 0)
        with torch.no_grad():
            for validity in (0., 1.):
                x = self.inputs.clone()
                x[:, 2] = validity
                if validity == 0:
                    x[:, 1] = 0
                self.assertTrue(torch.isfinite(model(x)['logits']).all())

    def test_consistency_only_on_noisy_samples(self):
        student = torch.randn(2, 3, 4, 4, requires_grad=True)
        teacher = torch.randn(2, 3, 4, 4)
        labels = torch.zeros(2, 4, 4, dtype=torch.long)
        self.assertEqual(runner.consistency(student, teacher, labels,
                                            torch.tensor([False, False])).item(), 0.)
        loss = runner.consistency(student, teacher, labels, torch.tensor([True, False]))
        loss.backward()
        self.assertGreater(student.grad[0].abs().sum().item(), 0)
        self.assertEqual(student.grad[1].abs().sum().item(), 0)

    def test_paired_inputs_match_existing_training_data(self):
        scratch = Path(__file__).resolve().parents[2] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='noise_pair_', dir=scratch) as folder:
            root = Path(folder)
            (root/'manifests').mkdir()
            Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(root/'a.png')
            Image.fromarray(np.full((8, 8), 128, np.uint16)).save(root/'d.png')
            Image.fromarray(np.zeros((8, 8), np.uint8)).save(root/'y.png')
            train = [{'sample_id': f'train{i}', 'intensity': 'a.png', 'depth': 'd.png',
                      'label': 'y.png'} for i in range(715)]
            dev = [{'sample_id': f'dev{i}', 'intensity': 'a.png', 'depth': 'd.png',
                    'label': 'y.png'} for i in range(80)]
            for name, rows in (('train', train), ('dev', dev)):
                (root/'manifests'/f'{name}.jsonl').write_text(
                    ''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
            original = TrainingData(root, 'mixed', 42)
            paired = PairedIntensityData(root, 42)
            for epoch in (1, 2):
                original.epoch = paired.epoch = epoch
                for index in (0, 5, 30):
                    x, y = original[index]
                    px, py, clean, active = paired[index]
                    torch.testing.assert_close(px, x, rtol=0, atol=0)
                    torch.testing.assert_close(py, y, rtol=0, atol=0)
                    self.assertEqual(clean.shape, x.shape)
                    self.assertIsInstance(bool(active), bool)

    def test_candidate_resume_and_gate(self):
        torch.manual_seed(7)
        source = {'args': {'decoder_dim': 32}, 'model': ToyNoiseModel().state_dict()}
        refs = {f'{f}_{s}': {'degradation_audit': [],
                            'regions': {'all': {'miou': .4}}}
                for f, s in runner.CONDITIONS}
        def metrics(*args):
            return {'degradation_audit': [], 'regions': {'all': {'miou': .4}}}
        scratch = Path(__file__).resolve().parents[2] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='noise_resume_', dir=scratch) as folder, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'NoiseCMX', ToyNoiseModel))
            stack.enter_context(patch.object(runner, 'PairedIntensityData', ToyPairedData))
            stack.enter_context(patch.object(runner, 'DiagnosticData', lambda *args: None))
            stack.enter_context(patch.object(runner, 'regional', metrics))
            stack.enter_context(patch.object(runner, 'confidence_audit',
                                             lambda *args: {'synthetic': {'predicted_mean': .5}}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            args = SimpleNamespace(output_dir=Path(folder), nyuv2_dir=Path(folder), device='cpu')
            protocol = {'test': True}
            original = runner.atomic_checkpoint
            def interrupt(path, state):
                original(path, state)
                raise RuntimeError('synthetic interruption')
            with patch.object(runner, 'atomic_checkpoint', interrupt):
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    runner.train_candidate(args, 'D2', 42, source, protocol, refs)
            result = runner.train_candidate(args, 'D2', 42, source, protocol, refs)
            self.assertEqual(len(result['scores']), 22)
            self.assertEqual(result['corruption_mean_gain_pp'], 0.)
            self.assertEqual(runner.train_candidate(args, 'D2', 42, source, protocol, refs), result)
            for method in ('C1', 'D1'):
                self.assertEqual(len(runner.train_candidate(
                    args, method, 42, source, protocol, refs)['scores']), 22)
            self.assertIsNone(runner.choose([result], refs))

    def test_confidence_audit_reads_only_matched_inputs(self):
        class FakeDiagnostic:
            def __init__(self, root, family, severity):
                self.family = family
            def __len__(self):
                return 2
            def __getitem__(self, index):
                x = torch.full((3, 4, 4), .2 + .1*index)
                if self.family != 'clean':
                    x[0, :2] += .15
                return {'sample_id': str(index), 'inputs': x}
        with patch.object(runner, 'DiagnosticData', FakeDiagnostic):
            values = runner.confidence_audit(ToyNoiseModel().eval(), Path('.'), 'cpu')
        self.assertEqual(len(values), 3)
        self.assertIsNone(values['clean_medium']['proxy_correlation'])
        self.assertGreater(values['clean_medium']['proxy_target_mean'],
                           values['photon_proxy_heavy']['proxy_target_mean'])

    def test_preflight_checks_reference_hashes(self):
        scratch = Path(__file__).resolve().parents[2] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='noise_preflight_', dir=scratch) as folder:
            root = Path(folder)
            phase1, nyuv2, reference, output = [root/name for name in
                                                  ('phase1', 'nyuv2', 'reference', 'output')]
            for seed in (42, 777):
                checkpoint = phase1/f'seed_{seed}'/'best.pth'
                checkpoint.parent.mkdir(parents=True)
                torch.save({'seed': seed}, checkpoint)
            manifest = nyuv2/'manifests'
            manifest.mkdir(parents=True)
            for part in ('train', 'dev'):
                (manifest/f'{part}.jsonl').write_text('{}\n', encoding='utf-8')
            (reference/'mixed'/'curves').mkdir(parents=True)
            runner.save(reference/'complete.json',
                        {'training_runs': 2, 'epochs_each': 20, 'evaluations': 44})
            runner.save(reference/'protocol.json',
                        {'checkpoint_sha256': runner.digest(phase1/'seed_42'/'best.pth'),
                         'train_manifest_sha256': runner.digest(manifest/'train.jsonl'),
                         'dev_manifest_sha256': runner.digest(manifest/'dev.jsonl')})
            for family, severity in runner.CONDITIONS:
                runner.save(reference/'mixed'/'curves'/f'{family}_{severity}.json',
                            {'regime': 'mixed', 'epoch': 20, 'regions': {'all': {'miou': .4}}})
            output.mkdir()
            args = SimpleNamespace(phase1_dir=phase1, nyuv2_dir=nyuv2,
                                   reference_dir=reference, output_dir=output, budget_hours=5.)
            protocol, refs = runner.preflight(args)
            self.assertEqual(len(refs), 22)
            self.assertEqual(protocol, runner.preflight(args)[0])
            (manifest/'train.jsonl').write_text('changed\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                runner.preflight(args)


if __name__ == '__main__':
    unittest.main()
