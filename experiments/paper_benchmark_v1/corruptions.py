from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


CORRUPTIONS = (
    "clean",
    "random_missing",
    "block_missing",
    "edge_missing",
    "low_signal_missing",
    "depth_noise",
    "photon_proxy",
    "joint",
    "mixture",
)
SEVERITIES = ("light", "medium", "heavy")


def stable_seed(global_seed: int, split: str, sample_id: str, corruption: str, severity: str) -> int:
    payload = f"{global_seed}|{split}|{sample_id}|{corruption}|{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def load_calibration(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(value.get("levels", {})) != set(SEVERITIES):
        raise ValueError(f"Calibration must define {SEVERITIES}: {path}")
    return value


def _normalize01(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    result = np.zeros_like(array)
    if not finite.any():
        return result
    lo, hi = np.percentile(array[finite], [1, 99])
    if hi > lo:
        result[finite] = np.clip((array[finite] - lo) / (hi - lo), 0, 1)
    return result


def _add_random_missing(valid: np.ndarray, target: float, rng: np.random.Generator) -> np.ndarray:
    missing = ~valid
    needed = max(0, int(round(target * valid.size)) - int(missing.sum()))
    candidates = np.flatnonzero(valid)
    if needed and len(candidates):
        chosen = rng.choice(candidates, size=min(needed, len(candidates)), replace=False)
        missing.flat[chosen] = True
    return ~missing


def _add_blocks(valid: np.ndarray, target: float, largest: float, rng: np.random.Generator) -> np.ndarray:
    result = valid.copy()
    height, width = result.shape
    wanted = max(0, int(round(target * result.size)) - int((~result).sum()))
    max_area = max(1, int(round(largest * result.size)))
    attempts = 0
    while wanted > 0 and attempts < 256:
        area = min(wanted, max_area)
        aspect = float(np.exp(rng.uniform(np.log(0.35), np.log(2.85))))
        block_h = max(1, min(height, int(round(np.sqrt(area / aspect)))))
        block_w = max(1, min(width, int(round(area / block_h))))
        y = int(rng.integers(0, height - block_h + 1))
        x = int(rng.integers(0, width - block_w + 1))
        before = int((~result).sum())
        result[y:y + block_h, x:x + block_w] = False
        gained = int((~result).sum()) - before
        wanted -= gained
        attempts += 1
        if gained == 0:
            max_area = min(result.size, max_area * 2)
    return _add_random_missing(result, target, rng)


def _weighted_missing(valid: np.ndarray, weight: np.ndarray, target: float,
                      rng: np.random.Generator) -> np.ndarray:
    result = valid.copy()
    needed = max(0, int(round(target * valid.size)) - int((~valid).sum()))
    candidates = np.flatnonzero(valid)
    if not needed or not len(candidates):
        return result
    probabilities = np.asarray(weight, dtype=np.float64).ravel()[candidates]
    probabilities = np.maximum(probabilities, 1e-8)
    probabilities /= probabilities.sum()
    selected = rng.choice(candidates, size=min(needed, len(candidates)), replace=False, p=probabilities)
    result.flat[selected] = False
    return result


def apply_corruption(
    intensity: np.ndarray,
    depth: np.ndarray,
    corruption: str,
    severity: str,
    calibration: dict,
    *,
    global_seed: int,
    split: str,
    sample_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return intensity, depth and validity; input arrays are never modified."""
    if corruption not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption {corruption}; choose from {CORRUPTIONS}")
    if severity not in SEVERITIES:
        raise ValueError(f"Unknown severity {severity}; choose from {SEVERITIES}")
    if corruption == "mixture":
        selector = np.random.default_rng(stable_seed(global_seed, split, sample_id, "mixture", "mixed"))
        corruption = ("random_missing", "block_missing", "edge_missing", "low_signal_missing",
                      "depth_noise", "photon_proxy", "joint")[int(selector.integers(0, 7))]
        severity = SEVERITIES[int(selector.integers(0, 3))]
    rng = np.random.default_rng(stable_seed(global_seed, split, sample_id, corruption, severity))
    appearance = _normalize01(intensity)
    geometry = np.asarray(depth, dtype=np.float32).copy()
    valid = np.isfinite(geometry) & (geometry > 0)
    level = calibration["levels"][severity]
    target = float(level["target_invalid_fraction"])
    largest = float(level["target_largest_hole_fraction"])
    proxy = calibration["appearance_proxy_defaults"][severity]

    if corruption == "random_missing":
        valid = _add_random_missing(valid, target, rng)
    elif corruption == "block_missing":
        valid = _add_blocks(valid, target, largest, rng)
    elif corruption == "edge_missing":
        filled = geometry.copy()
        if valid.any():
            filled[~valid] = float(np.median(filled[valid]))
        edge = np.hypot(ndimage.sobel(filled, 0), ndimage.sobel(filled, 1))
        valid = _weighted_missing(valid, edge + np.percentile(edge, 25), target, rng)
    elif corruption == "low_signal_missing":
        valid = _weighted_missing(valid, (1.0 - appearance) ** 2 + 1e-3, target, rng)
    elif corruption == "depth_noise":
        if valid.any():
            scale = {"light": 0.005, "medium": 0.015, "heavy": 0.030}[severity]
            span = float(np.percentile(geometry[valid], 99) - np.percentile(geometry[valid], 1))
            geometry[valid] += rng.normal(0.0, max(span, 1e-6) * scale, int(valid.sum()))
    elif corruption == "photon_proxy":
        peak = float(proxy["poisson_peak"])
        appearance = rng.poisson(np.clip(appearance, 0, 1) * peak).astype(np.float32) / peak
        appearance += rng.normal(0.0, float(proxy["background_sigma"]), appearance.shape)
        appearance = np.clip(appearance, 0, 1)
    elif corruption == "joint":
        valid = _add_blocks(valid, target, largest, rng)
        valid = _weighted_missing(valid, (1.0 - appearance) ** 2 + 1e-3, target, rng)
        peak = float(proxy["poisson_peak"])
        appearance = rng.poisson(np.clip(appearance, 0, 1) * peak).astype(np.float32) / peak
        appearance += rng.normal(0.0, float(proxy["background_sigma"]), appearance.shape)
        appearance = np.clip(appearance, 0, 1)

    geometry[~valid] = 0.0
    return appearance.astype(np.float32), geometry.astype(np.float32), valid.astype(np.float32)
