"""Diagnostic v2: added missingness is a fraction of originally valid pixels.

Rates are fixed stress-test settings, NOT physical SPL calibration. v1 is kept
unchanged for reproducing historical checkpoints.
"""
import numpy as np
from scipy import ndimage
from .corruptions import _normalize01, stable_seed

RATES = {'light': .10, 'medium': .25, 'heavy': .40}
FAMILIES = ('random_missing', 'block_missing', 'edge_missing',
            'low_signal_missing', 'depth_noise', 'photon_proxy', 'joint')


def corrupt(image, depth, family, severity, sample_id, seed=20260926):
    if family not in ('clean', *FAMILIES) or severity not in RATES:
        raise ValueError((family, severity))
    a = _normalize01(image)
    d = np.asarray(depth, np.float32).copy()
    original = np.isfinite(d) & (d > 0)
    valid = original.copy()
    # Same random field across severity permits nested missing masks.
    rng = np.random.default_rng(stable_seed(seed, 'dev_v2', str(sample_id), family, 'shared'))
    candidates = np.flatnonzero(original)
    n = int(round(RATES[severity] * len(candidates)))
    added = np.zeros(d.shape, bool)
    if family.endswith('missing') or family == 'joint':
        score = rng.random(d.shape)
        if family in ('block_missing', 'joint'):
            # Smooth spatial random field produces clustered holes; ranking
            # enforces exact added-pixel budgets independent of natural holes.
            score = ndimage.gaussian_filter(score, sigma=max(1., min(d.shape) / 24))
        elif family == 'edge_missing':
            filled = np.where(original, d, np.median(d[original]) if original.any() else 0)
            edge = np.hypot(ndimage.sobel(filled, 0), ndimage.sobel(filled, 1))
            score = score + _normalize01(edge)
        elif family == 'low_signal_missing':
            score = score + (1 - a) ** 2
        if family == 'joint':
            block_n = n // 2
            ordered = candidates[np.argsort(score.ravel()[candidates], kind='stable')]
            added.flat[ordered[-block_n:]] = True if block_n else False
            remaining = np.flatnonzero(original & ~added)
            signal_score = rng.random(d.shape) + (1 - a) ** 2
            chosen = remaining[np.argsort(signal_score.ravel()[remaining], kind='stable')]
            if n - block_n:
                added.flat[chosen[-(n - block_n):]] = True
        elif n:
            order = candidates[np.argsort(score.ravel()[candidates], kind='stable')]
            added.flat[order[-n:]] = True
        valid &= ~added
    if family == 'depth_noise' and valid.any():
        sigma = {'light': .005, 'medium': .015, 'heavy': .03}[severity]
        span = max(float(np.ptp(np.percentile(d[valid], [1, 99]))), 1e-6)
        d[valid] = np.maximum(d[valid] + rng.normal(size=int(valid.sum())) * span * sigma, 1e-6)
    if family in ('photon_proxy', 'joint'):
        peak = {'light': 64., 'medium': 32., 'heavy': 16.}[severity]
        sigma = {'light': .01, 'medium': .02, 'heavy': .04}[severity]
        a = np.clip(rng.poisson(a * peak) / peak + rng.normal(0, sigma, a.shape), 0, 1)
    d[~valid] = 0
    return a.astype(np.float32), d, valid, added
