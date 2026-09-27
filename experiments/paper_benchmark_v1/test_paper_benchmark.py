from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .corruptions import CORRUPTIONS, apply_corruption
from .model import CMXPaper


def calibration() -> dict:
    return {
        "levels": {
            "light": {"target_invalid_fraction": 0.14, "target_largest_hole_fraction": 0.07},
            "medium": {"target_invalid_fraction": 0.19, "target_largest_hole_fraction": 0.11},
            "heavy": {"target_invalid_fraction": 0.25, "target_largest_hole_fraction": 0.17},
        },
        "appearance_proxy_defaults": {
            "light": {"poisson_peak": 64.0, "background_sigma": 0.01},
            "medium": {"poisson_peak": 32.0, "background_sigma": 0.02},
            "heavy": {"poisson_peak": 16.0, "background_sigma": 0.04},
        },
    }


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[:64, :64]
        self.intensity = (xx + yy).astype(np.float32)
        self.depth = (1000 + xx * 2 + yy).astype(np.float32)
        self.depth[:4, :] = 0

    def test_all_corruptions_are_deterministic(self):
        for name in CORRUPTIONS:
            first = apply_corruption(self.intensity, self.depth, name, "medium", calibration(),
                                     global_seed=42, split="test", sample_id="1")
            second = apply_corruption(self.intensity, self.depth, name, "medium", calibration(),
                                      global_seed=42, split="test", sample_id="1")
            for left, right in zip(first, second):
                np.testing.assert_array_equal(left, right)

    def test_missingness_severity_is_monotonic(self):
        for name in ("random_missing", "block_missing", "edge_missing", "low_signal_missing", "joint"):
            rates = []
            for severity in ("light", "medium", "heavy"):
                _, _, valid = apply_corruption(
                    self.intensity, self.depth, name, severity, calibration(),
                    global_seed=42, split="test", sample_id="1",
                )
                rates.append(float(1 - valid.mean()))
            self.assertTrue(rates[0] <= rates[1] <= rates[2], (name, rates))


class ModelTests(unittest.TestCase):
    def test_cmx_backbone_shapes(self):
        inputs = torch.rand(2, 3, 64, 64)
        for variant in ("CMX_B0", "CMX_B1", "CMX_B2"):
            model = CMXPaper(variant, num_classes=4, image_size=64, decoder_dim=32).eval()
            with torch.no_grad():
                output = model(inputs)["logits"]
            self.assertEqual(tuple(output.shape), (2, 4, 64, 64))


if __name__ == "__main__":
    unittest.main()

