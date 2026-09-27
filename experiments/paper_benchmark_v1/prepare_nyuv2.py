from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field(mat: dict, *names: str) -> np.ndarray:
    for name in names:
        if name in mat:
            return np.asarray(mat[name]).squeeze()
    raise KeyError(f"None of {names} found; available={sorted(k for k in mat if not k.startswith('__'))}")


def labels_for_index(labels: np.ndarray, index: int) -> np.ndarray:
    if labels.shape[-1] == 1449:
        value = labels[..., index]
    elif labels.shape[0] == 1449:
        value = labels[index]
    else:
        raise ValueError(f"Cannot identify sample axis in labels40 shape {labels.shape}")
    value = np.asarray(value).squeeze()
    if value.shape == (640, 480):
        value = value.T
    if value.shape != (480, 640):
        raise ValueError(f"Unexpected label shape {value.shape}")
    # Standard NYU40 files use 1..40 and 0=unlabelled. Store 0..39 and 255=ignore.
    output = np.full(value.shape, 255, dtype=np.uint8)
    selected = (value >= 1) & (value <= 40)
    output[selected] = value[selected].astype(np.uint8) - 1
    return output


def image_for_index(dataset: h5py.Dataset, index: int) -> np.ndarray:
    value = np.asarray(dataset[index])
    if value.shape == (3, 640, 480):
        value = value.transpose(2, 1, 0)
    elif value.shape == (480, 640, 3):
        pass
    else:
        raise ValueError(f"Unexpected RGB sample shape {value.shape}")
    return value.astype(np.uint8)


def depth_for_index(dataset: h5py.Dataset, index: int) -> np.ndarray:
    value = np.asarray(dataset[index], dtype=np.float32).squeeze()
    if value.shape == (640, 480):
        value = value.T
    if value.shape != (480, 640):
        raise ValueError(f"Unexpected rawDepths sample shape {value.shape}")
    # Millimetres in uint16 preserve zeros and all valid NYUv2 ranges.
    return np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0, 0, 65535).astype(np.uint16)


def write_jsonl(path: Path, indices_1based: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in indices_1based.tolist():
            stem = f"{int(sample):04d}"
            row = {
                "dataset": "NYUv2-40",
                "sample_id": stem,
                "intensity": f"intensity/{stem}.png",
                "rgb": f"rgb/{stem}.png",
                "depth": f"raw_depth_mm/{stem}.png",
                "label": f"label40/{stem}.png",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract NYUv2 grayscale/raw-depth/NYU40 paper benchmark.")
    parser.add_argument("--labeled-mat", type=Path, required=True)
    parser.add_argument("--splits-mat", type=Path, required=True)
    parser.add_argument("--labels40-mat", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-count", type=int, default=80)
    parser.add_argument("--dev-seed", type=int, default=20260922)
    args = parser.parse_args()

    for path in (args.labeled_mat, args.splits_mat, args.labels40_mat):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required non-empty file not found: {path}")
    output = args.output_dir.resolve()
    for name in ("rgb", "intensity", "raw_depth_mm", "label40", "manifests"):
        (output / name).mkdir(parents=True, exist_ok=True)

    split_mat = loadmat(args.splits_mat)
    train = field(split_mat, "trainNdxs", "trainIdxs").astype(np.int64)
    test = field(split_mat, "testNdxs", "testIdxs").astype(np.int64)
    if len(train) != 795 or len(test) != 654 or set(train) & set(test):
        raise RuntimeError(f"Expected standard NYUv2 split 795/654, got {len(train)}/{len(test)}")
    labels_mat = loadmat(args.labels40_mat)
    labels = field(labels_mat, "labels40")

    with h5py.File(args.labeled_mat, "r") as archive:
        if "images" not in archive or "rawDepths" not in archive:
            raise KeyError("Official labeled MAT must contain images and rawDepths")
        if archive["images"].shape[0] != 1449 or archive["rawDepths"].shape[0] != 1449:
            raise RuntimeError(f"Unexpected HDF5 shapes: {archive['images'].shape}, {archive['rawDepths'].shape}")
        for index in range(1449):
            stem = f"{index + 1:04d}"
            rgb = image_for_index(archive["images"], index)
            raw_depth = depth_for_index(archive["rawDepths"], index)
            label = labels_for_index(labels, index)
            intensity = np.rint(
                0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            ).astype(np.uint8)
            Image.fromarray(rgb, mode="RGB").save(output / "rgb" / f"{stem}.png", optimize=True)
            Image.fromarray(intensity, mode="L").save(output / "intensity" / f"{stem}.png", optimize=True)
            Image.fromarray(raw_depth, mode="I;16").save(output / "raw_depth_mm" / f"{stem}.png", optimize=True)
            Image.fromarray(label, mode="L").save(output / "label40" / f"{stem}.png", optimize=True)
            if (index + 1) % 100 == 0:
                print(f"EXTRACT_PROGRESS {index + 1}/1449", flush=True)

    rng = np.random.default_rng(args.dev_seed)
    shuffled = train.copy()
    rng.shuffle(shuffled)
    dev = np.sort(shuffled[:args.dev_count])
    tuning_train = np.sort(shuffled[args.dev_count:])
    manifests = {
        "train": tuning_train,
        "dev": dev,
        "train_full": np.sort(train),
        "test": np.sort(test),
    }
    for name, values in manifests.items():
        write_jsonl(output / "manifests" / f"{name}.jsonl", values)

    first_depth = np.asarray(Image.open(output / "raw_depth_mm" / "0001.png"))
    first_label = np.asarray(Image.open(output / "label40" / "0001.png"))
    audit = {
        "schema_version": 1,
        "dataset": "NYUv2 40-class semantic segmentation",
        "source": "official nyu_depth_v2_labeled.mat plus standard splits/labels40 metadata",
        "modalities": {
            "appearance": "RGB converted to one-channel ITU-R BT.601 luma proxy",
            "depth": "rawDepths (not inpainted depths), stored as uint16 millimetres; zero is invalid",
            "label": "NYU40, stored 0..39; 255 is ignored",
        },
        "counts": {name: int(len(values)) for name, values in manifests.items()},
        "internal_dev": {
            "purpose": "hyperparameter/model selection only",
            "seed": args.dev_seed,
            "count": args.dev_count,
            "final_reporting": "freeze recipe, retrain on train_full=795, evaluate test=654 once",
        },
        "checks": {
            "resolution": list(first_depth.shape),
            "first_depth_invalid_fraction": float((first_depth == 0).mean()),
            "first_label_values": sorted(np.unique(first_label).astype(int).tolist()),
        },
        "input_sha256": {
            "labeled_mat": sha256(args.labeled_mat),
            "splits_mat": sha256(args.splits_mat),
            "labels40_mat": sha256(args.labels40_mat),
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

