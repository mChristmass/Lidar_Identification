import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from configs import kfold_config as kfold
from scripts.kfold_utils import print_summary, save_json, summarize_metrics


def read_meta(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_full_gt(mask_dir, mask_name):
    mask_path = Path(mask_dir) / mask_name
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Missing or unreadable mask: {mask_path}")
    return (mask > 0).astype(np.uint8)


def reconstruct_roi_mask_to_full_scale(roi_mask, meta):
    roi_mask = roi_mask.astype(np.uint8)

    resize_meta = meta["resize_meta"]
    pad_top = int(resize_meta["pad_top"])
    pad_bottom = int(resize_meta["pad_bottom"])
    pad_left = int(resize_meta["pad_left"])
    pad_right = int(resize_meta["pad_right"])
    orig_crop_h, orig_crop_w = [int(v) for v in resize_meta["orig_shape"]]
    target_h, target_w = [int(v) for v in resize_meta["target_shape"]]

    h_end = target_h - pad_bottom
    w_end = target_w - pad_right
    roi_unpadded = roi_mask[pad_top:h_end, pad_left:w_end]
    roi_orig_size = cv2.resize(
        roi_unpadded,
        (orig_crop_w, orig_crop_h),
        interpolation=cv2.INTER_NEAREST,
    )

    full_h, full_w = [int(v) for v in meta["orig_image_shape"]]
    x1, y1, x2, y2 = [int(v) for v in meta["bbox_margin_xyxy"]]
    full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = roi_orig_size
    return full_mask


def compute_binary_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())

    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_area": int(pred.sum()),
        "gt_area": int(gt.sum()),
    }


def evaluate_roi_dir(roi_dir, mask_dir, require_oracle=False):
    roi_dir = Path(roi_dir)
    rows = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for sample_dir in sorted(path for path in roi_dir.iterdir() if path.is_dir()):
        meta_path = sample_dir / "meta.json"
        roi_gt_path = sample_dir / "gt_mask.npy"
        if not meta_path.exists() or not roi_gt_path.exists():
            continue

        meta = read_meta(meta_path)
        roi_source = meta.get("roi_source", "missing")
        if require_oracle and roi_source != "oracle":
            continue

        roi_gt = np.load(roi_gt_path)
        pred_full = reconstruct_roi_mask_to_full_scale(roi_gt, meta)
        gt_full = read_full_gt(mask_dir, meta["mask_name"])
        metrics = compute_binary_metrics(pred_full, gt_full)

        row = {
            "roi_id": meta.get("roi_id", sample_dir.name),
            "mask_name": meta["mask_name"],
            "roi_source": roi_source,
            "bbox_margin_xyxy": meta["bbox_margin_xyxy"],
            **metrics,
        }
        rows.append(row)
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

    if not rows:
        raise ValueError(f"No evaluable ROI samples found in {roi_dir}")

    global_metrics = compute_binary_metrics_from_counts(total_tp, total_fp, total_fn)
    mean_metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("iou", "precision", "recall", "dice")
    }
    worst_rows = sorted(rows, key=lambda row: row["iou"])[:10]
    return {
        "roi_dir": str(roi_dir),
        "num_samples": len(rows),
        "global": global_metrics,
        "sample_mean": mean_metrics,
        "worst_samples": worst_rows,
        "samples": rows,
    }


def compute_binary_metrics_from_counts(tp, fp, fn):
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def stage2_dir_for_run(runs_dir, seed):
    return Path(runs_dir) / kfold.fold_name(seed) / "stage2"


def evaluate_one_fold(seed, split, require_oracle, runs_dir):
    stage2_dir = stage2_dir_for_run(runs_dir, seed)
    roi_dir = stage2_dir / "ROI" / f"roi_{split}"
    result = evaluate_roi_dir(roi_dir, kfold.LABEL_DIR, require_oracle=require_oracle)
    result["seed"] = int(seed)
    result["split"] = split

    out_path = stage2_dir / f"oracle_roi_upper_bound_{split}.json"
    save_json(out_path, result)

    global_metrics = result["global"]
    print(
        f"Seed {seed} {split}: "
        f"IoU {global_metrics['iou']:.6f}  "
        f"Precision {global_metrics['precision']:.6f}  "
        f"Recall {global_metrics['recall']:.6f}  "
        f"Dice {global_metrics['dice']:.6f}  "
        f"samples {result['num_samples']}"
    )
    return {
        "seed": int(seed),
        "split": split,
        **global_metrics,
        "num_samples": result["num_samples"],
        "result_path": str(out_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the full-scale upper bound of saved oracle Stage2 ROI gt masks."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=kfold.KFOLD_SEEDS)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--runs-dir", type=Path, default=kfold.RUNS_DIR)
    parser.add_argument(
        "--require-oracle",
        action="store_true",
        help="Skip ROI samples whose meta.json roi_source is not oracle.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [evaluate_one_fold(seed, args.split, args.require_oracle, args.runs_dir) for seed in args.seeds]
    metric_keys = ["iou", "precision", "recall", "dice"]
    summary = summarize_metrics(rows, metric_keys)
    out_path = args.runs_dir / f"summary_oracle_roi_upper_bound_{args.split}.json"
    save_json(out_path, summary)
    print_summary(f"Oracle ROI Upper Bound {args.split}", summary, metric_keys)
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
