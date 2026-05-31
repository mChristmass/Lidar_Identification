import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.stage1.config_stage1 as conf
from configs import kfold_config as kfold
from datasets.dataset_stage1 import LidarSegDataset
from engine.stage1.visualize_stage1 import visualize_sample
from engine.stage1.loss_stage1 import SegmentationLoss
from models.model_stage1 import UNet
from scripts.kfold_utils import save_json, set_seed, summarize_metrics, tee_stdout, print_summary


def compute_stage1_metrics(logits, labels, threshold=0.3, valid_mask=None):
    probs = torch.softmax(logits, dim=1)[:, 1, :, :]
    preds = (probs > threshold).long()

    if valid_mask is not None:
        preds = preds * valid_mask.long()
        labels = labels * valid_mask.long()

    preds_f = preds.float()
    labels_f = labels.float()

    intersection = (preds_f * labels_f).sum(dim=(1, 2))
    pred_sum = preds_f.sum(dim=(1, 2))
    label_sum = labels_f.sum(dim=(1, 2))
    union = pred_sum + label_sum - intersection

    iou = (intersection / (union + 1e-6)).mean().item()
    precision = (intersection / (pred_sum + 1e-6)).mean().item()
    recall = (intersection / (label_sum + 1e-6)).mean().item()
    pred_area = pred_sum.mean().item()

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "coverage": recall,
        "pred_area": pred_area,
    }


def evaluate(model, loader, device, threshold=0.3):
    model.eval()
    totals = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "coverage": 0.0, "pred_area": 0.0}
    n = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            metrics = compute_stage1_metrics(model(imgs), labels, threshold=threshold)
            for key in totals:
                totals[key] += metrics[key]
            n += 1

    if n == 0:
        raise ValueError("Cannot evaluate an empty loader.")
    return {key: value / n for key, value in totals.items()}


def validate_indices(train_indices, val_indices, test_indices, mask_dir):
    label_indices = sorted(int(f.replace(".png", "")) for f in os.listdir(mask_dir) if f.endswith(".png"))
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


def visualize_test_samples(model, test_loader, out_dir, device, num_visualize):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            pred = model(data)
            pred_mask = torch.argmax(pred, dim=1)

            for i in range(data.size(0)):
                if count >= num_visualize:
                    return
                intensity = data[i, 0].cpu().numpy()
                mask = pred_mask[i].cpu().numpy()
                visualize_sample(
                    intensity=intensity,
                    mask=mask,
                    save_path=str(out_dir / f"sample_{count}.png"),
                    title=f"Test Sample {count}",
                )
                count += 1


def train_one_fold(seed, train_indices, val_indices, test_indices, paths=None):
    set_seed(seed)
    paths = paths or {
        "save_path": Path(conf.SAVE_PATH),
        "vis_dir": Path(conf.out_path),
        "log_path": Path(conf.out_path).parent / "train.log",
        "metrics_path": Path(conf.out_path).parent / "metrics.json",
    }

    for path_key in ("save_path", "vis_dir", "log_path", "metrics_path"):
        paths[path_key] = Path(paths[path_key])
    paths["save_path"].parent.mkdir(parents=True, exist_ok=True)

    with tee_stdout(paths["log_path"]):
        print(f"\n===== Stage1 fold seed {seed} =====")
        print(f"Train samples: {len(train_indices)}")
        print(f"Val samples  : {len(val_indices)}")
        print(f"Test samples : {len(test_indices)}")

        validate_indices(train_indices, val_indices, test_indices, conf.MASK_DIR)

        train_set = LidarSegDataset(conf.NPY_PATH, conf.MASK_DIR, train_indices)
        val_set = LidarSegDataset(conf.NPY_PATH, conf.MASK_DIR, val_indices)
        test_set = LidarSegDataset(conf.NPY_PATH, conf.MASK_DIR, test_indices)

        train_loader = DataLoader(train_set, batch_size=conf.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        model = UNet(in_channels=conf.INPUT_CHANNEL, num_classes=2).to(conf.DEVICE)
        criterion = SegmentationLoss(
            target_weight=conf.TARGET_WEIGHT,
            ce_weight=conf.CE_WEIGHT,
            dice_weight=conf.DICE_WEIGHT,
        ).to(conf.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=conf.LR)

        best_val_recall = -1.0
        for epoch in range(conf.EPOCHS):
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
            val_metrics = evaluate(model, val_loader, conf.DEVICE, threshold=conf.EVALUATE_threshold)

            print(
                f"Epoch [{epoch + 1}/{conf.EPOCHS}]  "
                f"Loss: {avg_loss:.4f}  "
                f"Val IoU: {val_metrics['iou']:.3f}  "
                f"Val Precision: {val_metrics['precision']:.3f}  "
                f"Val Recall: {val_metrics['recall']:.3f}  "
                f"Val coverage: {val_metrics['coverage']:.3f}  "
                f"Val pred_area: {val_metrics['pred_area']:.3f}"
            )

            if val_metrics["recall"] > best_val_recall:
                best_val_recall = val_metrics["recall"]
                torch.save(model.state_dict(), paths["save_path"])
                print(f"Best model saved. Val Recall = {best_val_recall:.3f}")

        print("\nLoading best model for final test evaluation...")
        model.load_state_dict(torch.load(paths["save_path"], map_location=conf.DEVICE))
        test_metrics = evaluate(model, test_loader, conf.DEVICE, threshold=conf.EVALUATE_threshold)

        print("\n===== Final Test Result =====")
        print(
            f"Test IoU: {test_metrics['iou']:.3f}  "
            f"Test Precision: {test_metrics['precision']:.3f}  "
            f"Test Recall: {test_metrics['recall']:.3f}  "
            f"Test coverage: {test_metrics['coverage']:.3f}  "
            f"Test pred_area: {test_metrics['pred_area']:.3f}"
        )

        visualize_test_samples(model, test_loader, paths["vis_dir"], conf.DEVICE, conf.VISUALIZE_NUM)

        result = {"seed": seed, **test_metrics, "best_val_recall": best_val_recall}
        save_json(paths["metrics_path"], result)
        print(f"\nTraining finished. Best model saved to:\n{paths['save_path']}")
        return result


def train_all_folds():
    kfold.ensure_data_layout()
    results = []
    for seed in kfold.KFOLD_SEEDS:
        train_indices, val_indices, test_indices = kfold.load_split_indices(seed)
        results.append(
            train_one_fold(
                seed,
                train_indices,
                val_indices,
                test_indices,
                paths={
                    "save_path": kfold.stage1_model_path(seed),
                    "vis_dir": kfold.stage1_vis_dir(seed),
                    "log_path": kfold.stage1_dir(seed) / "train.log",
                    "metrics_path": kfold.stage1_dir(seed) / "metrics.json",
                },
            )
        )

    summary = summarize_metrics(results, ["iou", "precision", "recall", "coverage", "pred_area"])
    save_json(kfold.RUNS_DIR / "summary_stage1.json", summary)
    print_summary("Stage1 K-Fold Summary", summary, ["iou", "precision", "recall", "coverage", "pred_area"])
    return summary


if __name__ == "__main__":
    train_all_folds()
