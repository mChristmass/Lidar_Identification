from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def largest_component_fraction(mask: np.ndarray) -> tuple[int, float, int]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0, 0.0, 0
    sizes = np.bincount(labels.ravel())[1:]
    return int(count), float(sizes.max() / mask.size), int(np.median(sizes))


def boundary_band(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask)
    dilated = ndimage.binary_dilation(mask, iterations=radius)
    eroded = ndimage.binary_erosion(mask, iterations=radius)
    return dilated ^ eroded


def safe_rate(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(numerator[denominator].mean()) if denominator.any() else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit degradation using the frozen private TRAIN split only."
    )
    parser.add_argument("--merged-dir", type=Path, default=Path("data/new_data/merged"))
    parser.add_argument(
        "--split-dir", type=Path, default=Path("data/new_data/merged/paper_split_v1")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/new_data/merged/paper_split_v1/degradation_profile_v1"),
    )
    args = parser.parse_args()

    merged = args.merged_dir.resolve()
    split_dir = args.split_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    train_1based = np.load(split_dir / "train_indices.npy").astype(np.int64)
    if len(train_1based) != 970 or len(np.unique(train_1based)) != len(train_1based):
        raise RuntimeError("Expected the frozen paper_split_v1 train split with 970 unique frames")
    indices = train_1based - 1
    intensity = np.load(merged / "intensity.npy", mmap_mode="r")
    depth = np.load(merged / "depth.npy", mmap_mode="r")
    if intensity.shape != depth.shape or intensity.shape[0] != 1390:
        raise RuntimeError(f"Unexpected private arrays: {intensity.shape}, {depth.shape}")

    rows: list[dict] = []
    pooled_decile_invalid = np.zeros(10, dtype=np.float64)
    pooled_decile_count = np.zeros(10, dtype=np.int64)
    for frame_1based, array_index in zip(train_1based.tolist(), indices.tolist()):
        image = np.asarray(intensity[array_index], dtype=np.float32)
        geometry = np.asarray(depth[array_index], dtype=np.float32)
        label = np.asarray(Image.open(merged / "label" / f"{frame_1based}.png")) > 0
        invalid = ~np.isfinite(geometry) | (geometry <= 0)
        valid_intensity = np.isfinite(image)
        values = image[valid_intensity]
        p01, p10, p50, p90, p99 = np.percentile(values, [1, 10, 50, 90, 99])
        robust_range = max(float(p99 - p01), 1e-6)
        normalized = np.clip((image - p01) / robust_range, 0.0, 1.0)
        low_signal = normalized <= 0.10
        band = boundary_band(label, radius=2)
        background = ~label
        components, largest_fraction, median_component = largest_component_fraction(invalid)

        # Per-frame ranks make this independent of raw detector gain. Equal-valued
        # frames fall into the lowest available bin deterministically.
        rank_edges = np.percentile(values, np.arange(0, 101, 10))
        bins = np.clip(np.searchsorted(rank_edges[1:-1], image, side="right"), 0, 9)
        for decile in range(10):
            selected = valid_intensity & (bins == decile)
            pooled_decile_invalid[decile] += int((invalid & selected).sum())
            pooled_decile_count[decile] += int(selected.sum())

        rows.append({
            "frame_1based": frame_1based,
            "invalid_fraction": float(invalid.mean()),
            "valid_fraction": float((~invalid).mean()),
            "invalid_component_count": components,
            "largest_invalid_component_fraction": largest_fraction,
            "median_invalid_component_pixels": median_component,
            "invalid_rate_foreground": safe_rate(invalid, label),
            "invalid_rate_boundary_band_r2": safe_rate(invalid, band),
            "invalid_rate_background": safe_rate(invalid, background),
            "intensity_p01": float(p01),
            "intensity_p10": float(p10),
            "intensity_p50": float(p50),
            "intensity_p90": float(p90),
            "intensity_p99": float(p99),
            "intensity_robust_range": robust_range,
            "low_signal_fraction": float(low_signal.mean()),
            "joint_low_signal_invalid_fraction": float((low_signal & invalid).mean()),
        })

    write_csv(output / "frame_stats.csv", rows)
    numeric_keys = [key for key in rows[0] if key != "frame_1based"]
    aggregate = {
        "schema_version": 1,
        "scope": "private frozen training split only; dev and final_test arrays were not read",
        "frame_count": len(rows),
        "definitions": {
            "invalid_depth": "non-finite or <= 0",
            "boundary_band": "GT dilation XOR erosion with radius 2 pixels",
            "low_signal": "per-frame robust-normalized intensity <= 0.10",
            "warning": "Intensity statistics are empirical proxies, not a physical photon-SNR estimate.",
        },
        "statistics": {},
        "invalid_rate_by_within_frame_intensity_decile": [
            float(a / b) if b else None
            for a, b in zip(pooled_decile_invalid, pooled_decile_count)
        ],
    }
    for key in numeric_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        aggregate["statistics"][key] = {
            "mean": float(values.mean()),
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
        }

    invalid_fractions = [float(row["invalid_fraction"]) for row in rows]
    largest_holes = [float(row["largest_invalid_component_fraction"]) for row in rows]
    low_signal_joint = [float(row["joint_low_signal_invalid_fraction"]) for row in rows]
    levels = {}
    for name, q in (("light", 25), ("medium", 50), ("heavy", 75)):
        levels[name] = {
            "target_invalid_fraction": percentile(invalid_fractions, q),
            "target_largest_hole_fraction": percentile(largest_holes, q),
            "target_low_signal_invalid_fraction": percentile(low_signal_joint, q),
        }
    calibration = {
        "schema_version": 1,
        "calibration_source": "private paper_split_v1 train only",
        "seed_policy": "sha256(global_seed, split, sample_id, corruption, severity)",
        "levels": levels,
        "appearance_proxy_defaults": {
            "light": {"poisson_peak": 64.0, "background_sigma": 0.010},
            "medium": {"poisson_peak": 32.0, "background_sigma": 0.020},
            "heavy": {"poisson_peak": 16.0, "background_sigma": 0.040},
        },
        "interpretation": (
            "Depth missingness targets and hole scale are calibrated to private training data. "
            "Appearance parameters are fixed stress-test proxies and must not be called a physical SPL simulator."
        ),
    }

    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "calibration.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    provenance = {
        "script": str(Path(__file__).resolve()),
        "input_sha256": {
            "train_indices.npy": sha256(split_dir / "train_indices.npy"),
            "intensity.npy": sha256(merged / "intensity.npy"),
            "depth.npy": sha256(merged / "depth.npy"),
        },
        "outputs_sha256": {
            name: sha256(output / name)
            for name in ("frame_stats.csv", "aggregate.json", "calibration.json")
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "levels": levels}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
