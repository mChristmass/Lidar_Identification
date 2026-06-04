import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.stage1.config_stage1 as stage1_conf
from configs import kfold_config as kfold
from engine.stage1.normalization_stage1 import data_normalization
from models.model_stage1 import UNet
from scripts.kfold_utils import save_json


class PredictDataset(Dataset):
    def __init__(self, npy_path):
        self.images = data_normalization(npy_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]).float(), idx


def softmax_numpy(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)


def load_selected_stage1(seed, runs_dir):
    fold_dir = Path(runs_dir) / kfold.fold_name(seed) / "stage1_only_strong"
    selected_path = fold_dir / "selected_metrics.json"
    if not selected_path.exists():
        raise FileNotFoundError(
            f"Missing selected Stage1-only metrics: {selected_path}\n"
            "Run scripts/train_stage1_only_strong.py first."
        )
    with selected_path.open("r", encoding="utf-8") as handle:
        selected = json.load(handle)
    model_path = fold_dir / selected["setting"] / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing selected Stage1 model: {model_path}")
    return selected, model_path


def bbox_from_mask(mask, margin):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return [0, 0, mask.shape[1], mask.shape[0]]

    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    w = int(stats[label, cv2.CC_STAT_WIDTH])
    h = int(stats[label, cv2.CC_STAT_HEIGHT])

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(mask.shape[1], x + w + margin)
    y2 = min(mask.shape[0], y + h + margin)
    return [x1, y1, x2, y2]


def build_roi_masks(coarse_masks, margin, dilate_iter):
    roi_masks = np.zeros_like(coarse_masks, dtype=np.uint8)
    bboxes = []
    kernel = np.ones((3, 3), dtype=np.uint8)

    for idx, mask in enumerate(coarse_masks):
        work_mask = mask.astype(np.uint8)
        if dilate_iter > 0:
            work_mask = cv2.dilate(work_mask, kernel, iterations=dilate_iter)
        x1, y1, x2, y2 = bbox_from_mask(work_mask, margin)
        roi_masks[idx, y1:y2, x1:x2] = 1
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])

    return roi_masks, bboxes


def predict_one_fold(seed, args):
    selected, model_path = load_selected_stage1(seed, args.runs_dir)
    threshold = float(selected["best_threshold"]) if args.threshold is None else float(args.threshold)
    out_dir = Path(args.runs_dir) / kfold.fold_name(seed) / "stage1_strong_priors"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PredictDataset(stage1_conf.NPY_PATH)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    raw_shape = np.load(stage1_conf.NPY_PATH).shape
    n, h, w = raw_shape[0], raw_shape[1], raw_shape[2]

    model = UNet(in_channels=stage1_conf.INPUT_CHANNEL, num_classes=2).to(stage1_conf.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=stage1_conf.DEVICE))
    model.eval()

    logits_all = np.zeros((n, 2, h, w), dtype=np.float32)
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(stage1_conf.DEVICE)
            logits = model(images).detach().cpu().numpy()
            for i, index in enumerate(indices.tolist()):
                logits_all[index] = logits[i]

    prob_all = softmax_numpy(logits_all, axis=1)[:, 1]
    coarse_masks = (prob_all > threshold).astype(np.uint8)
    roi_masks, bboxes = build_roi_masks(coarse_masks, args.roi_margin, args.roi_dilate_iter)

    np.save(out_dir / "logits.npy", logits_all)
    np.save(out_dir / "prob.npy", prob_all.astype(np.float32))
    np.save(out_dir / "coarse_mask.npy", coarse_masks)
    np.save(out_dir / "roi_mask.npy", roi_masks)
    save_json(
        out_dir / "meta.json",
        {
            "seed": int(seed),
            "stage1_setting": selected["setting"],
            "stage1_model_path": str(model_path),
            "threshold": threshold,
            "roi_margin": int(args.roi_margin),
            "roi_dilate_iter": int(args.roi_dilate_iter),
            "bboxes": bboxes,
        },
    )
    print(f"Seed {seed}: saved priors to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict full-scale priors from selected strong Stage1 models.")
    parser.add_argument("--runs-dir", type=Path, default=kfold.RUNS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=kfold.KFOLD_SEEDS)
    parser.add_argument("--batch-size", type=int, default=stage1_conf.BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--roi-margin", type=int, default=8)
    parser.add_argument("--roi-dilate-iter", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    for seed in args.seeds:
        predict_one_fold(seed, args)


if __name__ == "__main__":
    main()
