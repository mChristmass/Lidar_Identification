import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.stage2_fixed_roi.config_stage2_fixed_roi as conf
from configs import kfold_config as kfold
from datasets.dataset_stage2_fixed_roi import Stage2FixedRoiDataset
from engine.stage1.loss_stage1 import SegmentationLoss
from models.model_stage1 import UNet
from scripts.kfold_utils import print_summary, save_json, set_seed, summarize_metrics, tee_stdout


DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def fold_dir(runs_dir, seed):
    return Path(runs_dir) / kfold.fold_name(seed)


def prepare_run(source_runs_dir, runs_dir, seeds):
    source_runs_dir = Path(source_runs_dir)
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        source_meta = fold_dir(source_runs_dir, seed) / "stage1_strong_priors" / "meta.json"
        target_dir = fold_dir(runs_dir, seed) / "stage1_roi_locator"
        target_dir.mkdir(parents=True, exist_ok=True)
        if not source_meta.exists():
            raise FileNotFoundError(f"Missing Stage1 locator metadata: {source_meta}")
        shutil.copy2(source_meta, target_dir / "meta.json")
        print(f"Prepared Stage1 ROI locator for seed {seed}: {target_dir}")

    summary_source = source_runs_dir / "summary_stage1_only_strong.json"
    if summary_source.exists():
        shutil.copy2(summary_source, runs_dir / summary_source.name)

    save_json(
        runs_dir / "stage2_fixed_roi_experiment.json",
        {
            "source_runs_dir": str(source_runs_dir),
            "runs_dir": str(runs_dir),
            "seeds": [int(seed) for seed in seeds],
            "roi_size": int(conf.ROI_SIZE),
            "input_items": ["intensity_roi", "local_depth_edge_roi"],
            "resize": False,
            "edge_normalization": "ROI-local Sobel, valid-mask erosion, percentile-99",
        },
    )


def make_dataset(indices, locator_meta_path, args):
    return Stage2FixedRoiDataset(
        raw_items=kfold.RAW_ITEMS,
        mask_dir=kfold.LABEL_DIR,
        indices=indices,
        locator_meta_path=locator_meta_path,
        roi_size=args.roi_size,
        edge_erode_iterations=args.edge_erode_iterations,
    )


def paste_roi_batch(pred_roi, placements, full_shape):
    batch_size = pred_roi.shape[0]
    full_pred = torch.zeros(
        (batch_size, full_shape[0], full_shape[1]),
        dtype=pred_roi.dtype,
        device=pred_roi.device,
    )
    for i in range(batch_size):
        src_x1, src_y1, src_x2, src_y2, dst_x1, dst_y1, dst_x2, dst_y2 = [
            int(v) for v in placements[i].tolist()
        ]
        full_pred[i, src_y1:src_y2, src_x1:src_x2] = pred_roi[
            i, dst_y1:dst_y2, dst_x1:dst_x2
        ]
    return full_pred


def counts_from_preds(pred, labels):
    pred = pred.bool()
    labels = labels.bool()
    return {
        "tp": int((pred & labels).sum().item()),
        "fp": int((pred & ~labels).sum().item()),
        "fn": int((~pred & labels).sum().item()),
        "pred_area": int(pred.sum().item()),
        "label_area": int(labels.sum().item()),
    }


def metrics_from_counts(row):
    tp, fp, fn = row["tp"], row["fp"], row["fn"]
    return {
        "iou": float(tp / (tp + fp + fn + 1e-6)),
        "precision": float(tp / (tp + fp + 1e-6)),
        "recall": float(tp / (tp + fn + 1e-6)),
        "dice": float(2 * tp / (2 * tp + fp + fn + 1e-6)),
        "coverage": float(tp / (tp + fn + 1e-6)),
        "pred_area": float(row["pred_area"]),
        "label_area": float(row["label_area"]),
    }


def evaluate_thresholds(model, loader, device, thresholds):
    model.eval()
    totals = {
        float(threshold): {"tp": 0, "fp": 0, "fn": 0, "pred_area": 0, "label_area": 0}
        for threshold in thresholds
    }
    with torch.no_grad():
        for x, _, full_gt, placements, _ in loader:
            x = x.to(device)
            full_gt = full_gt.to(device)
            probs_roi = torch.softmax(model(x), dim=1)[:, 1]
            for threshold in totals:
                pred_roi = probs_roi > threshold
                full_pred = paste_roi_batch(pred_roi, placements, full_gt.shape[-2:])
                counts = counts_from_preds(full_pred, full_gt)
                for key in totals[threshold]:
                    totals[threshold][key] += counts[key]
    return {threshold: metrics_from_counts(row) for threshold, row in totals.items()}


def select_best_threshold(metrics):
    return max(
        metrics.items(),
        key=lambda item: (item[1]["iou"], item[1]["dice"], item[1]["recall"]),
    )


def compute_bbox_coverage(dataset):
    inside = 0
    total = 0
    for idx in range(len(dataset)):
        _, gt_roi, full_gt, _, _ = dataset[idx]
        inside += int(gt_roi.sum().item())
        total += int(full_gt.sum().item())
    return float(inside / (total + 1e-6))


def train_one_fold(seed, args):
    set_seed(seed)
    train_indices, val_indices, test_indices = kfold.load_split_indices(seed)
    current_fold = fold_dir(args.runs_dir, seed)
    locator_meta_path = current_fold / "stage1_roi_locator" / "meta.json"
    out_dir = current_fold / "stage2_fixed_roi"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = make_dataset(train_indices, locator_meta_path, args)
    val_set = make_dataset(val_indices, locator_meta_path, args)
    test_set = make_dataset(test_indices, locator_meta_path, args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    with tee_stdout(out_dir / "train.log"):
        test_coverage = compute_bbox_coverage(test_set)
        print(f"\n===== Stage2 fixed ROI, seed {seed} =====")
        print("Input items: intensity ROI + ROI-local depth edge")
        print(f"ROI size: {args.roi_size}x{args.roi_size}, resize: False")
        print(f"Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}")
        print(f"Test GT coverage by fixed ROI: {test_coverage:.6f}")

        model = UNet(in_channels=2, num_classes=2).to(conf.DEVICE)
        criterion = SegmentationLoss(
            target_weight=args.target_weight,
            ce_weight=conf.CE_WEIGHT,
            dice_weight=args.dice_weight,
        ).to(conf.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_iou = -1.0
        best_threshold = None
        best_val_metrics = None
        best_model_path = out_dir / "best_model.pth"

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for x, gt_roi, _, _, _ in train_loader:
                x = x.to(conf.DEVICE)
                gt_roi = gt_roi.to(conf.DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), gt_roi)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_by_threshold = evaluate_thresholds(model, val_loader, conf.DEVICE, args.thresholds)
            threshold, val_metrics = select_best_threshold(val_by_threshold)
            print(
                f"Epoch [{epoch + 1}/{args.epochs}]  "
                f"Loss: {total_loss / max(1, len(train_loader)):.4f}  "
                f"Thr: {threshold:g}  "
                f"Val IoU: {val_metrics['iou']:.4f}  "
                f"Dice: {val_metrics['dice']:.4f}  "
                f"Precision: {val_metrics['precision']:.4f}  "
                f"Recall: {val_metrics['recall']:.4f}"
            )
            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                best_threshold = threshold
                best_val_metrics = val_metrics
                torch.save(model.state_dict(), best_model_path)
                save_json(out_dir / "val_threshold_metrics.json", val_by_threshold)

        model.load_state_dict(torch.load(best_model_path, map_location=conf.DEVICE))
        test_metrics = evaluate_thresholds(
            model,
            test_loader,
            conf.DEVICE,
            [best_threshold],
        )[best_threshold]
        result = {
            "seed": int(seed),
            "input_items": ["intensity_roi", "local_depth_edge_roi"],
            "roi_size": int(args.roi_size),
            "resize": False,
            "edge_erode_iterations": int(args.edge_erode_iterations),
            "test_gt_coverage": test_coverage,
            "target_weight": float(args.target_weight),
            "dice_weight": float(args.dice_weight),
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "best_threshold": float(best_threshold),
            "best_val_iou": float(best_val_iou),
            "best_val_metrics": best_val_metrics,
            **test_metrics,
        }
        save_json(out_dir / "metrics.json", result)
        print(
            f"Test IoU: {result['iou']:.4f}  Dice: {result['dice']:.4f}  "
            f"Precision: {result['precision']:.4f}  Recall: {result['recall']:.4f}"
        )
        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-size, no-resize Stage2 ROI k-fold experiment.")
    parser.add_argument("--stage", choices=["prepare", "train", "all"], default="all")
    parser.add_argument("--source-runs-dir", type=Path, default=kfold.DATA_ROOT / "runs/run8")
    parser.add_argument("--runs-dir", type=Path, default=kfold.DATA_ROOT / "runs/run11")
    parser.add_argument("--seeds", type=int, nargs="+", default=kfold.KFOLD_SEEDS)
    parser.add_argument("--roi-size", type=int, default=conf.ROI_SIZE)
    parser.add_argument("--epochs", type=int, default=conf.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=conf.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=conf.LR)
    parser.add_argument("--target-weight", type=float, default=conf.TARGET_WEIGHT)
    parser.add_argument("--dice-weight", type=float, default=conf.DICE_WEIGHT)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--edge-erode-iterations", type=int, default=conf.EDGE_ERODE_ITERATIONS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.roi_size % 16 != 0:
        raise ValueError("ROI size must be divisible by 16 for the current U-Net.")
    if args.stage in {"prepare", "all"}:
        prepare_run(args.source_runs_dir, args.runs_dir, args.seeds)
    if args.stage in {"train", "all"}:
        results = [train_one_fold(seed, args) for seed in args.seeds]
        metric_keys = ["iou", "precision", "recall", "dice", "coverage", "pred_area"]
        summary = summarize_metrics(results, metric_keys)
        save_json(Path(args.runs_dir) / "summary_stage2_fixed_roi.json", summary)
        print_summary("Stage2 Fixed ROI K-Fold Summary", summary, metric_keys)


if __name__ == "__main__":
    main()
