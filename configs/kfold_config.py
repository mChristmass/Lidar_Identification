import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
LABEL_DIR = DATA_ROOT / "labels"
KFOLD_INDEX_DIR = DATA_ROOT / "kfold_indices"
RUNS_DIR = DATA_ROOT / "runs"

KFOLD_SEEDS = [42, 777, 2025, 3407, 114514]

RAW_ITEMS = {
    "intensity": RAW_DIR / "intensity.npy",
    "depth": RAW_DIR / "depth.npy",
    "depth_edge": RAW_DIR / "depth_edge.npy",
}

STAGE1_INPUT_ITEM = "intensity"
STAGE1_INPUT_CHANNEL = 1

STAGE2_INPUT_ITEMS = ["intensity", "depth", "depth_edge", "prob"]
ROI_SIZE = (128, 128)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_str(path: Path) -> str:
    return os.fspath(path)


def fold_name(seed: int) -> str:
    return f"fold_seed{seed}"


def fold_run_dir(seed: int) -> Path:
    return RUNS_DIR / fold_name(seed)


def stage1_dir(seed: int) -> Path:
    return fold_run_dir(seed) / "stage1"


def stage2_dir(seed: int) -> Path:
    return fold_run_dir(seed) / "stage2"


def stage1_model_path(seed: int) -> Path:
    return stage1_dir(seed) / "best_model.pth"


def stage1_vis_dir(seed: int) -> Path:
    return stage1_dir(seed) / "visualizations"


def stage1_coarse_dir(seed: int) -> Path:
    return stage1_dir(seed) / "coarse_masks"


def stage1_logits_path(seed: int) -> Path:
    return stage1_coarse_dir(seed) / "coarse_masks.npy"


def stage1_prob_path(seed: int) -> Path:
    return stage1_coarse_dir(seed) / "prob.npy"


def stage1_png_dir(seed: int) -> Path:
    return stage1_coarse_dir(seed) / "png"


def stage2_roi_root(seed: int) -> Path:
    return stage2_dir(seed) / "ROI"


def stage2_model_path(seed: int) -> Path:
    return stage2_dir(seed) / "best_model.pth"


def stage2_vis_dir(seed: int) -> Path:
    return stage2_dir(seed) / "Final_mask"


def index_path(seed: int, split: str) -> Path:
    return KFOLD_INDEX_DIR / f"{split}_indices_seed{seed}.npy"


def load_split_indices(seed: int):
    import numpy as np

    all_indices = load_label_indices()
    indices = {"train": [], "val": [], "test": []}
    for split in ("val", "test"):
        path = index_path(seed, split)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing k-fold index file: {path}\n"
                f"Please place {split}_indices_seed{seed}.npy in {KFOLD_INDEX_DIR}."
            )
        indices[split] = np.load(path).astype(int).tolist()

    return build_train_from_val_test(seed, all_indices, indices["val"], indices["test"])


def load_label_indices():
    label_indices = sorted(
        int(path.stem)
        for path in LABEL_DIR.glob("*.png")
    )
    if not label_indices:
        raise FileNotFoundError(
            f"No label masks found in {LABEL_DIR}. "
            "Please place 1-based label masks such as 001.png in data/labels."
        )
    return label_indices


def build_train_from_val_test(seed: int, all_indices, val_indices, test_indices):
    """Use all labeled samples except val/test as train samples."""
    all_set = set(all_indices)
    test_set = set(test_indices)
    val_set = set(val_indices)

    missing_val = sorted(val_set - all_set)
    missing_test = sorted(test_set - all_set)
    if missing_val:
        raise ValueError(f"Val indices for seed {seed} not found in labels: {missing_val}")
    if missing_test:
        raise ValueError(f"Test indices for seed {seed} not found in labels: {missing_test}")

    val_test_overlap = val_set & test_set
    if val_test_overlap:
        raise ValueError(f"Overlap between val and test for seed {seed}: {sorted(val_test_overlap)}")

    train_set = all_set - val_set - test_set
    return sorted(train_set), sorted(val_set), sorted(test_set)


def ensure_data_layout() -> None:
    for path in (RAW_DIR, LABEL_DIR, KFOLD_INDEX_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)
