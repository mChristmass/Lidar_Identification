"""Synthetic CPU checks for inference-only repair diagnosis."""
import unittest
import contextlib
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from .diagnose_repair import FactorModel, segmentation_state, effects, private_audit
from .model import CMXPaper
from .run_repair_window import load_frontend
from . import diagnose_repair as runner


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_off_bypasses_nonzero_restorer_and_transfer_preserves_segmenter(self):
        base = CMXPaper('CMX_B0', 3, 64, 32).eval()
        model = FactorModel(False, backbone='CMX_B0', num_classes=3, image_size=64, decoder_dim=32).eval()
        model.load_baseline(base.state_dict())
        with torch.no_grad():
            model.restorer[-1].bias.fill_(.3)
            model.noise_gate[-1].bias.fill_(4)
            x = torch.rand(2, 3, 64, 64)
            torch.testing.assert_close(model(x)['logits'], base(x)['logits'], atol=0, rtol=0)
            model.enabled = True
            self.assertGreater((model(x)['logits']-base(x)['logits']).abs().sum().item(), 0)
            destination = FactorModel(True, backbone='CMX_B0', num_classes=2, image_size=64, decoder_dim=32)
            before = {k: v.clone() for k, v in segmentation_state(destination.state_dict()).items()}
            load_frontend(destination, model.state_dict())
            for key, value in before.items():
                torch.testing.assert_close(destination.state_dict()[key], value, atol=0, rtol=0)

    def test_factorial_contrasts(self):
        result = effects(dict(R0_off=.5, R0_on=.52, S2_off=.49, S2_on=.54))
        self.assertAlmostEqual(result['frontend_on_R0_pp'], 2)
        self.assertAlmostEqual(result['segmentation_change_off_pp'], -1)
        self.assertAlmostEqual(result['interaction_pp'], 3)

    def test_private_audit_empty_regions(self):
        class Frontend:
            def restore(self, x):
                return dict(noise_gate=torch.full((1, 1, 1, 1), .75),
                            restored_intensity=x[:, :1]+.1)
        data = [dict(inputs=torch.zeros(3, 4, 4), labels=torch.zeros(4, 4, dtype=torch.long), sample_id='dev1')]
        out = private_audit(Frontend(), data, 'cpu')
        self.assertIsNone(out['per_frame'][0]['foreground_correction_mae'])
        self.assertAlmostEqual(out['gate_mean'], .75)
        self.assertAlmostEqual(out['correction_mae_frame_mean'], .1, places=6)

    def test_full_schedule_resume_and_mismatch(self):
        scratch = Path(__file__).resolve().parents[2]/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch, prefix='repair_diagnosis_') as folder, contextlib.ExitStack() as stack:
            root = Path(folder)
            args = SimpleNamespace(output_dir=root/'out', reference_dir=root/'ref',
                                   repair_dir=root/'repair', private_checkpoint=root/'private.pth',
                                   private_dir=root/'private', nyuv2_dir=root/'nyu', device='cpu')
            row = dict(regions={'all': {'miou': .5}}, degradation_audit=[])
            for f, s in runner.CONDITIONS:
                for directory in (args.reference_dir/'mixed', args.repair_dir/'S2_seed42'):
                    runner.save(directory/'curves'/f'{f}_{s}.json', row)
            runner.save(args.repair_dir/'private_dev_transfer.json',
                        {'metrics': {k: {'foreground_iou': .7} for k in ('baseline', 'S2')}})
            def load(path, **kwargs):
                if path == args.private_checkpoint:
                    return dict(model={}, args=dict(dataset='private', model='CMX_B2', seed=42,
                                                     stage='tune', decoder_dim=256))
                return dict(model={}, epoch=20, group='S2_seed42')
            stack.enter_context(patch.object(runner, 'preflight', lambda a: None))
            stack.enter_context(patch.object(runner.torch, 'load', load))
            stack.enter_context(patch.object(runner, 'build_model', lambda *a: torch.nn.Identity()))
            stack.enter_context(patch.object(runner, 'DiagnosticData', lambda *a: []))
            regional = stack.enter_context(patch.object(runner, 'regional', return_value=row))
            stack.enter_context(patch.object(runner, 'PrivatePaperDataset', lambda *a: []))
            stack.enter_context(patch.object(runner, 'evaluate', return_value={'foreground_iou': .7}))
            stack.enter_context(patch.object(runner, 'private_audit', return_value={'gate_mean': .5}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            runner.execute(args)
            self.assertEqual(regional.call_count, 16)
            self.assertEqual(runner.read(args.output_dir/'complete.json')['evaluations'], 18)
            runner.execute(args)
            self.assertEqual(regional.call_count, 16)
            runner.save(args.output_dir/'curves/R0_off_clean_medium.json',
                        dict(regions={'all': {'miou': .1}}, degradation_audit=[]))
            with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                runner.execute(args)


if __name__ == '__main__':
    unittest.main()
