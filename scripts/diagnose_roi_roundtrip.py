import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from configs import kfold_config as kfold
from scripts.kfold_utils import save_json


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_gt(mask_dir, mask_name):
    path = Path(mask_dir) / mask_name
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return (mask > 0).astype(np.uint8)


def resize_with_meta(arr, resize_meta, is_mask=True):
    target_h, target_w = [int(v) for v in resize_meta["target_shape"]]
    new_h, new_w = [int(v) for v in resize_meta["resized_shape"]]
    pad_top = int(resize_meta["pad_top"])
    pad_bottom = int(resize_meta["pad_bottom"])
    pad_left = int(resize_meta["pad_left"])
    pad_right = int(resize_meta["pad_right"])

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(arr, (new_w, new_h), interpolation=interpolation)
    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )[:target_h, :target_w]


def restore_roi_to_crop(roi_mask, resize_meta):
    roi_mask = roi_mask.astype(np.uint8)
    target_h, target_w = [int(v) for v in resize_meta["target_shape"]]
    pad_top = int(resize_meta["pad_top"])
    pad_bottom = int(resize_meta["pad_bottom"])
    pad_left = int(resize_meta["pad_left"])
    pad_right = int(resize_meta["pad_right"])
    orig_h, orig_w = [int(v) for v in resize_meta["orig_shape"]]

    h_end = target_h - pad_bottom
    w_end = target_w - pad_right
    unpadded = roi_mask[pad_top:h_end, pad_left:w_end]
    return cv2.resize(unpadded, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def paste_crop_to_full(crop_mask, meta):
    full_h, full_w = [int(v) for v in meta["orig_image_shape"]]
    x1, y1, x2, y2 = [int(v) for v in meta["bbox_margin_xyxy"]]
    full = np.zeros((full_h, full_w), dtype=np.uint8)
    full[y1:y2, x1:x2] = crop_mask.astype(np.uint8)
    return full


def crop_by_bbox(arr, bbox_xyxy):
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    return arr[y1:y2, x1:x2]


def binary_metrics(pred, gt):
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


def shape_checks(meta, roi_shape):
    resize_meta = meta["resize_meta"]
    x1, y1, x2, y2 = [int(v) for v in meta["bbox_margin_xyxy"]]
    bbox_h = y2 - y1
    bbox_w = x2 - x1
    orig_h, orig_w = [int(v) for v in resize_meta["orig_shape"]]
    target_h, target_w = [int(v) for v in resize_meta["target_shape"]]
    resized_h, resized_w = [int(v) for v in resize_meta["resized_shape"]]
    pad_top = int(resize_meta["pad_top"])
    pad_bottom = int(resize_meta["pad_bottom"])
    pad_left = int(resize_meta["pad_left"])
    pad_right = int(resize_meta["pad_right"])
    return {
        "bbox_matches_orig_shape": bbox_h == orig_h and bbox_w == orig_w,
        "pad_matches_target_shape": resized_h + pad_top + pad_bottom == target_h
        and resized_w + pad_left + pad_right == target_w,
        "roi_matches_target_shape": list(roi_shape) == [target_h, target_w],
        "bbox_shape": [bbox_h, bbox_w],
        "orig_shape": [orig_h, orig_w],
        "resized_shape": [resized_h, resized_w],
        "target_shape": [target_h, target_w],
    }


def make_diff_rgb(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    image = np.zeros((*gt.shape, 3), dtype=np.float32)
    image[pred & gt] = [0.1, 0.8, 0.2]
    image[pred & ~gt] = [1.0, 0.2, 0.1]
    image[~pred & gt] = [0.1, 0.4, 1.0]
    return image


def save_visual(row, arrays, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_gt = arrays["full_gt"]
    restored_full = arrays["restored_full"]
    bbox = row["bbox_margin_xyxy"]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    full_with_bbox = np.stack([full_gt * 255] * 3, axis=-1).astype(np.uint8)
    cv2.rectangle(full_with_bbox, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 0), 1)

    panels = [
        ("full GT + bbox", full_with_bbox),
        ("full GT crop", arrays["full_crop"], "gray"),
        ("saved gt_mask.npy", arrays["saved_roi"], "gray"),
        ("regenerated ROI", arrays["regenerated_roi"], "gray"),
        ("restored full", restored_full, "gray"),
        ("diff TP/FP/FN", make_diff_rgb(restored_full, full_gt)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, panel in zip(axes.flat, panels):
        title = panel[0]
        image = panel[1]
        cmap = panel[2] if len(panel) > 2 else None
        ax.imshow(image, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"{row['seed']} {row['split']} {row['roi_id']} {row['mask_name']} | "
        f"full IoU={row['full_roundtrip_iou']:.4f}, "
        f"roi IoU={row['saved_vs_regenerated_roi_iou']:.4f}"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def diagnose_sample(sample_dir, mask_dir, seed, split):
    meta = read_json(sample_dir / "meta.json")
    saved_roi = np.load(sample_dir / "gt_mask.npy").astype(np.uint8)
    full_gt = read_gt(mask_dir, meta["mask_name"])
    full_crop = crop_by_bbox(full_gt, meta["bbox_margin_xyxy"])
    regenerated_roi = resize_with_meta(full_crop, meta["resize_meta"], is_mask=True)
    restored_crop_from_saved = restore_roi_to_crop(saved_roi, meta["resize_meta"])
    restored_full = paste_crop_to_full(restored_crop_from_saved, meta)
    bbox_only_full = paste_crop_to_full(full_crop, meta)

    saved_vs_regenerated = binary_metrics(saved_roi, regenerated_roi)
    crop_roundtrip = binary_metrics(restored_crop_from_saved, full_crop)
    full_roundtrip = binary_metrics(restored_full, full_gt)
    bbox_only = binary_metrics(bbox_only_full, full_gt)
    checks = shape_checks(meta, saved_roi.shape)

    row = {
        "seed": int(seed),
        "split": split,
        "roi_id": meta.get("roi_id", sample_dir.name),
        "mask_name": meta["mask_name"],
        "roi_source": meta.get("roi_source", "missing"),
        "bbox_margin_xyxy": meta["bbox_margin_xyxy"],
        "saved_vs_regenerated_roi_iou": saved_vs_regenerated["iou"],
        "crop_roundtrip_iou": crop_roundtrip["iou"],
        "full_roundtrip_iou": full_roundtrip["iou"],
        "bbox_only_iou": bbox_only["iou"],
        "full_roundtrip_precision": full_roundtrip["precision"],
        "full_roundtrip_recall": full_roundtrip["recall"],
        "full_roundtrip_dice": full_roundtrip["dice"],
        "full_tp": full_roundtrip["tp"],
        "full_fp": full_roundtrip["fp"],
        "full_fn": full_roundtrip["fn"],
        "saved_roi_area": int(saved_roi.sum()),
        "regenerated_roi_area": int(regenerated_roi.sum()),
        "full_gt_area": int(full_gt.sum()),
        **checks,
    }
    arrays = {
        "full_gt": full_gt,
        "full_crop": full_crop,
        "saved_roi": saved_roi,
        "regenerated_roi": regenerated_roi,
        "restored_full": restored_full,
    }
    return row, arrays


def aggregate(rows):
    keys = [
        "saved_vs_regenerated_roi_iou",
        "crop_roundtrip_iou",
        "full_roundtrip_iou",
        "bbox_only_iou",
        "full_roundtrip_precision",
        "full_roundtrip_recall",
        "full_roundtrip_dice",
    ]
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
            "min": float(np.min([row[key] for row in rows])),
        }
        for key in keys
    }


def stage2_dir_for_run(runs_dir, seed):
    return Path(runs_dir) / kfold.fold_name(seed) / "stage2"


def diagnose_fold(seed, split, runs_dir, require_oracle, max_visuals):
    stage2_dir = stage2_dir_for_run(runs_dir, seed)
    roi_dir = stage2_dir / "ROI" / f"roi_{split}"
    out_dir = stage2_dir / f"roi_roundtrip_diagnosis_{split}"
    visual_dir = out_dir / "worst_visuals"
    rows = []
    arrays_by_id = {}

    for sample_dir in sorted(path for path in roi_dir.iterdir() if path.is_dir()):
        if not (sample_dir / "meta.json").exists() or not (sample_dir / "gt_mask.npy").exists():
            continue
        row, arrays = diagnose_sample(sample_dir, kfold.LABEL_DIR, seed, split)
        if require_oracle and row["roi_source"] != "oracle":
            continue
        rows.append(row)
        arrays_by_id[row["roi_id"]] = arrays

    if not rows:
        raise ValueError(f"No diagnosable samples found in {roi_dir}")

    rows = sorted(rows, key=lambda row: row["full_roundtrip_iou"])
    for row in rows[:max_visuals]:
        save_visual(row, arrays_by_id[row["roi_id"]], visual_dir / f"{row['roi_id']}_{row['mask_name']}.png")

    result = {
        "seed": int(seed),
        "split": split,
        "roi_dir": str(roi_dir),
        "num_samples": len(rows),
        "aggregate": aggregate(rows),
        "worst_samples": rows[:20],
        "all_samples": rows,
    }
    save_json(out_dir / "diagnosis.json", result)
    print(
        f"Seed {seed} {split}: "
        f"bbox-only {result['aggregate']['bbox_only_iou']['mean']:.6f}  "
        f"saved-vs-regen {result['aggregate']['saved_vs_regenerated_roi_iou']['mean']:.6f}  "
        f"crop-roundtrip {result['aggregate']['crop_roundtrip_iou']['mean']:.6f}  "
        f"full-roundtrip {result['aggregate']['full_roundtrip_iou']['mean']:.6f}"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose Stage2 ROI GT round-trip consistency and loss.")
    parser.add_argument("--runs-dir", type=Path, default=kfold.RUNS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=kfold.KFOLD_SEEDS)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--require-oracle", action="store_true")
    parser.add_argument("--max-visuals", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    results = [
        diagnose_fold(seed, args.split, args.runs_dir, args.require_oracle, args.max_visuals)
        for seed in args.seeds
    ]
    summary_rows = []
    for result in results:
        row = {"seed": result["seed"], "num_samples": result["num_samples"]}
        for key, value in result["aggregate"].items():
            row[f"{key}_mean"] = value["mean"]
            row[f"{key}_min"] = value["min"]
        summary_rows.append(row)

    summary = {"folds": summary_rows}
    out_path = Path(args.runs_dir) / f"summary_roi_roundtrip_diagnosis_{args.split}.json"
    save_json(out_path, summary)
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
