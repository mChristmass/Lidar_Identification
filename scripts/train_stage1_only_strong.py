import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.stage1.config_stage1 as conf
from configs import kfold_config as kfold
from datasets.dataset_stage1 import LidarSegDataset
from engine.stage1.loss_stage1 import SegmentationLoss
from engine.stage1.visualize_stage1 import visualize_sample
from models.model_stage1 import UNet
from scripts.kfold_utils import print_summary, save_json, set_seed, summarize_metrics, tee_stdout


DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def setting_name(target_weight, dice_weight, lr):
    return f"tw{target_weight:g}_dw{dice_weight:g}_lr{lr:g}"


def compute_confusion_from_probs(probs, labels, threshold):
    preds = probs > threshold
    labels = labels.bool()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    pred_area = preds.sum().item()
    label_area = labels.sum().item()
    return tp, fp, fn, pred_area, label_area


def metrics_from_confusion(tp, fp, fn, pred_area, label_area):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "coverage": float(recall),
        "pred_area": float(pred_area),
        "label_area": float(label_area),
    }


def evaluate_thresholds(model, loader, device, thresholds):
    model.eval()
    totals = {
        float(threshold): {"tp": 0, "fp": 0, "fn": 0, "pred_area": 0, "label_area": 0}
        for threshold in thresholds
    }

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            probs = torch.softmax(model(imgs), dim=1)[:, 1, :, :]

            for threshold in totals:
                tp, fp, fn, pred_area, label_area = compute_confusion_from_probs(probs, labels, threshold)
                totals[threshold]["tp"] += tp
                totals[threshold]["fp"] += fp
                totals[threshold]["fn"] += fn
                totals[threshold]["pred_area"] += pred_area
                totals[threshold]["label_area"] += label_area

    metrics = {}
    for threshold, row in totals.items():
        metrics[threshold] = metrics_from_confusion(
            row["tp"],
            row["fp"],
            row["fn"],
            row["pred_area"],
            row["label_area"],
        )
    return metrics


def select_best_threshold(threshold_metrics, key="iou"):
    return max(
        threshold_metrics.items(),
        key=lambda item: (item[1][key], item[1]["dice"], item[1]["recall"]),
    )


def validate_indices(train_indices, val_indices, test_indices, mask_dir):
    label_indices = sorted(int(path.stem) for path in Path(mask_dir).glob("*.png"))
    label_set = set(label_indices)
    split_sets = {
        "train": set(train_indices),
        "val": set(val_indices),
        "test": set(test_indices),
    }

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise ValueError(f"Overlap between {left} and {right}: {sorted(overlap)}")

    for split_name, split_set in split_sets.items():
        missing = sorted(split_set - label_set)
        if missing:
            raise ValueError(f"{split_name} indices not found in mask dir: {missing}")


def visualize_test_samples(model, test_loader, out_dir, device, threshold, num_visualize):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            probs = torch.softmax(model(data), dim=1)[:, 1, :, :]
            pred_masks = (probs > threshold).long()

            for i in range(data.size(0)):
                if count >= num_visualize:
                    return
                intensity = data[i, 0].detach().cpu().numpy()
                mask = pred_masks[i].detach().cpu().numpy()
                visualize_sample(
                    intensity=intensity,
                    mask=mask,
                    save_path=str(out_dir / f"sample_{count}.png"),
                    title=f"Stage1-only threshold={threshold:g}",
                )
                count += 1


def train_one_setting(
    seed,
    train_indices,
    val_indices,
    test_indices,
    target_weight,
    dice_weight,
    lr,
    epochs,
    batch_size,
    thresholds,
    out_dir,
):
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = LidarSegDataset(conf.NPY_PATH, conf.MASK_DIR, train_indices)
    val_set = LidarSegDataset(conf.NPY_PATH, conf.MASK_DIR, val_indices)
    test_set = LidarSegDataset(conf.NPY_PATH, conf.MASK_DIR, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = UNet(in_channels=conf.INPUT_CHANNEL, num_classes=2).to(conf.DEVICE)
    criterion = SegmentationLoss(
        target_weight=target_weight,
        ce_weight=conf.CE_WEIGHT,
        dice_weight=dice_weight,
    ).to(conf.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_iou = -1.0
    best_threshold = None
    best_val_metrics = None
    best_model_path = out_dir / "best_model.pth"

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(conf.DEVICE)
            labels = labels.to(conf.DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(train_loader))
        val_by_threshold = evaluate_thresholds(model, val_loader, conf.DEVICE, thresholds)
        epoch_threshold, epoch_val_metrics = select_best_threshold(val_by_threshold, key="iou")

        print(
            f"Epoch [{epoch + 1}/{epochs}]  "
            f"Loss: {avg_loss:.4f}  "
            f"ValBestThr: {epoch_threshold:g}  "
            f"Val IoU: {epoch_val_metrics['iou']:.4f}  "
            f"Val Dice: {epoch_val_metrics['dice']:.4f}  "
            f"Val Precision: {epoch_val_metrics['precision']:.4f}  "
            f"Val Recall: {epoch_val_metrics['recall']:.4f}"
        )

        if epoch_val_metrics["iou"] > best_val_iou:
            best_val_iou = epoch_val_metrics["iou"]
            best_threshold = epoch_threshold
            best_val_metrics = epoch_val_metrics
            torch.save(model.state_dict(), best_model_path)
            save_json(out_dir / "val_threshold_metrics.json", val_by_threshold)
            print(f">>> Saved best model. Val IoU = {best_val_iou:.4f}, threshold = {best_threshold:g}")

    model.load_state_dict(torch.load(best_model_path, map_location=conf.DEVICE))
    test_by_threshold = evaluate_thresholds(model, test_loader, conf.DEVICE, [best_threshold])
    test_metrics = test_by_threshold[best_threshold]
    visualize_test_samples(
        model,
        test_loader,
        out_dir / "visualizations",
        conf.DEVICE,
        threshold=best_threshold,
        num_visualize=conf.VISUALIZE_NUM,
    )

    result = {
        "seed": seed,
        "setting": setting_name(target_weight, dice_weight, lr),
        "target_weight": float(target_weight),
        "dice_weight": float(dice_weight),
        "lr": float(lr),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "best_threshold": float(best_threshold),
        "best_val_iou": float(best_val_iou),
        "best_val_metrics": best_val_metrics,
        **test_metrics,
    }
    save_json(out_dir / "metrics.json", result)
    return result


def train_one_fold(seed, args):
    train_indices, val_indices, test_indices = kfold.load_split_indices(seed)
    validate_indices(train_indices, val_indices, test_indices, conf.MASK_DIR)

    fold_dir = kfold.fold_run_dir(seed) / "stage1_only_strong"
    log_path = fold_dir / "train.log"
    fold_dir.mkdir(parents=True, exist_ok=True)

    with tee_stdout(log_path):
        print(f"\n===== Strong Stage1-only fold seed {seed} =====")
        print(f"Train samples: {len(train_indices)}")
        print(f"Val samples  : {len(val_indices)}")
        print(f"Test samples : {len(test_indices)}")
        print(f"Thresholds   : {args.thresholds}")
        print(f"Target weights: {args.target_weights}")
        print(f"Dice weights  : {args.dice_weights}")
        print(f"LRs           : {args.lrs}")

        setting_results = []
        for target_weight in args.target_weights:
            for dice_weight in args.dice_weights:
                for lr in args.lrs:
                    name = setting_name(target_weight, dice_weight, lr)
                    print(f"\n----- Setting {name} -----")
                    result = train_one_setting(
                        seed=seed,
                        train_indices=train_indices,
                        val_indices=val_indices,
                        test_indices=test_indices,
                        target_weight=target_weight,
                        dice_weight=dice_weight,
                        lr=lr,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        thresholds=args.thresholds,
                        out_dir=fold_dir / name,
                    )
                    setting_results.append(result)

        selected = max(
            setting_results,
            key=lambda row: (
                row["best_val_iou"],
                row["best_val_metrics"]["dice"],
                row["best_val_metrics"]["recall"],
            ),
        )
        save_json(fold_dir / "all_setting_metrics.json", {"settings": setting_results})
        save_json(fold_dir / "selected_metrics.json", selected)

        print("\n===== Selected Setting =====")
        print(
            f"{selected['setting']}  "
            f"Val IoU: {selected['best_val_iou']:.4f}  "
            f"Thr: {selected['best_threshold']:.3f}  "
            f"Test IoU: {selected['iou']:.4f}  "
            f"Test Dice: {selected['dice']:.4f}  "
            f"Test Precision: {selected['precision']:.4f}  "
            f"Test Recall: {selected['recall']:.4f}"
        )
        return selected


def train_all_folds(args):
    kfold.ensure_data_layout()
    results = [train_one_fold(seed, args) for seed in args.seeds]
    metric_keys = ["iou", "precision", "recall", "dice", "coverage", "pred_area"]
    summary = summarize_metrics(results, metric_keys)
    save_json(kfold.RUNS_DIR / "summary_stage1_only_strong.json", summary)
    print_summary("Strong Stage1-only K-Fold Summary", summary, metric_keys)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tuned full-scale Stage1-only segmentation baseline.")
    parser.add_argument("--seeds", type=int, nargs="+", default=kfold.KFOLD_SEEDS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=conf.BATCH_SIZE)
    parser.add_argument("--target-weights", type=float, nargs="+", default=[3.0, 5.0, 8.0])
    parser.add_argument("--dice-weights", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    return parser.parse_args()


if __name__ == "__main__":
    train_all_folds(parse_args())
