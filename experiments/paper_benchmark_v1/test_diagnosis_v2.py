import unittest
import numpy as np
from .corruptions_v2 import corrupt, FAMILIES, RATES


class TestCorruptions(unittest.TestCase):
    def test_budget_with_large_natural_holes(self):
        a = np.arange(4096, dtype=np.float32).reshape(64, 64)
        d = np.ones((64, 64), np.float32)
        d[:40] = 0
        original = d > 0
        for family in FAMILIES:
            for severity, rate in RATES.items():
                x, z, valid, added = corrupt(a, d, family, severity, 'sample')
                repeat = corrupt(a, d, family, severity, 'sample')
                np.testing.assert_array_equal(x, repeat[0])
                np.testing.assert_array_equal(z, repeat[1])
                self.assertTrue(np.isfinite(x).all() and np.isfinite(z).all())
                self.assertFalse((valid & ~original).any())
                self.assertTrue((z[~valid] == 0).all())
                if family.endswith('missing') or family == 'joint':
                    self.assertEqual(int(added.sum()), round(rate * original.sum()))
        self.assertTrue((d[:40] == 0).all())

    def test_empty_depth(self):
        for family in FAMILIES:
            _, d, valid, added = corrupt(np.ones((16, 16)), np.zeros((16, 16)), family, 'heavy', 'empty')
            self.assertEqual(int(valid.sum()), 0)
            self.assertTrue(np.isfinite(d).all())
            self.assertEqual(int(added.sum()), 0)


if __name__ == '__main__':
    unittest.main()
