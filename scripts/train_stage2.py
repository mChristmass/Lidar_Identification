import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.stage2.config_stage2 as conf
from configs import kfold_config as kfold
from datasets.dataset_stage2 import Stage2Dataset
from models.model_stage2 import Stage2UNet
from scripts.kfold_utils import print_summary, save_json, set_seed, summarize_metrics, tee_stdout


DEVICE = conf.DEVICE


def compute_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1).view(-1)
    labels = labels.view(-1)

    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return iou.item(), precision.item(), recall.item(), dice.item()


def reconstruct_to_full_scale(roi_pred, meta):
    if isinstance(roi_pred, torch.Tensor):
        roi_pred = roi_pred.cpu().numpy().astype(np.uint8)

    pad_top = meta["resize_meta"]["pad_top"]
    pad_bottom = meta["resize_meta"]["pad_bottom"]
    pad_left = meta["resize_meta"]["pad_left"]
    pad_right = meta["resize_meta"]["pad_right"]
    orig_crop_h, orig_crop_w = meta["resize_meta"]["orig_shape"]
    full_h, full_w = meta["orig_image_shape"]
    x1, y1, x2, y2 = meta["bbox_margin_xyxy"]

    target_h, target_w = meta["resize_meta"]["target_shape"]
    h_end = target_h - pad_bottom
    w_end = target_w - pad_right
    roi_unpadded = roi_pred[pad_top:h_end, pad_left:w_end]
    roi_orig_size = cv2.resize(roi_unpadded, (orig_crop_w, orig_crop_h), interpolation=cv2.INTER_NEAREST)

    full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = roi_orig_size
    return full_mask


def evaluate(model, loader, device):
    model.eval()
    totals = np.zeros(4, dtype=np.float64)
    n = 0

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            totals += np.array(compute_metrics(model(x), y), dtype=np.float64)
            n += 1

    if n == 0:
        raise ValueError("Cannot evaluate an empty loader.")
    return tuple((totals / n).tolist())


def evaluate_full_scale(model, loader, device, mask_dir, raw_intensity_path, save_dir=None):
    model.eval()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    raw_intensity_all = np.load(raw_intensity_path)
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for x, _, metas in loader:
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1).cpu().numpy()

            for i in range(preds.shape[0]):
                meta = extract_meta(metas, i)
                full_pred = reconstruct_to_full_scale(preds[i], meta)
                mask_path = os.path.join(mask_dir, meta["mask_name"])
                full_gt = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

                total_tp += ((full_pred == 1) & (full_gt == 1)).sum()
                total_fp += ((full_pred == 1) & (full_gt == 0)).sum()
                total_fn += ((full_pred == 0) & (full_gt == 1)).sum()

                if save_dir:
                    intensity_full = raw_intensity_all[int(meta["mask_name"].split(".")[0]) - 1, :, :]
                    save_path = os.path.join(save_dir, f"result_{meta['mask_name']}")
                    save_stage2_visualization(intensity_full, full_gt, full_pred, meta, save_path)

    iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-6)
    return float(iou), float(precision), float(recall), float(dice)


def extract_meta(metas, i):
    return {
        "mask_name": metas["mask_name"][i],
        "orig_image_shape": [
            int(metas["orig_image_shape"][0][i]),
            int(metas["orig_image_shape"][1][i]),
        ],
        "bbox_margin_xyxy": [
            int(metas["bbox_margin_xyxy"][0][i]),
            int(metas["bbox_margin_xyxy"][1][i]),
            int(metas["bbox_margin_xyxy"][2][i]),
            int(metas["bbox_margin_xyxy"][3][i]),
        ],
        "resize_meta": {
            "pad_top": int(metas["resize_meta"]["pad_top"][i]),
            "pad_bottom": int(metas["resize_meta"]["pad_bottom"][i]),
            "pad_left": int(metas["resize_meta"]["pad_left"][i]),
            "pad_right": int(metas["resize_meta"]["pad_right"][i]),
            "orig_shape": [
                int(metas["resize_meta"]["orig_shape"][0][i]),
                int(metas["resize_meta"]["orig_shape"][1][i]),
            ],
            "target_shape": conf.ROI_size,
        },
    }


def save_stage2_visualization(intensity_full, gt_full, pred_full, meta, save_path):
    plt.figure(figsize=(16, 9))

    plt.subplot(2, 2, 1)
    plt.imshow(intensity_full, cmap="gray")
    overlay = np.zeros((*pred_full.shape, 4))
    overlay[pred_full == 1] = [1, 0, 0, 0.4]
    plt.imshow(overlay)
    plt.title(f"Full-Scale Prediction Overlay (Frame: {meta['mask_name']})")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    error_vis = np.zeros((*pred_full.shape, 3))
    error_vis[(pred_full == 1) & (gt_full == 1)] = [0, 1, 0]
    error_vis[(pred_full == 1) & (gt_full == 0)] = [1, 0, 0]
    error_vis[(pred_full == 0) & (gt_full == 1)] = [0, 0, 1]
    plt.imshow(error_vis)
    plt.title("Error Analysis: TP(Green), FP(Red), FN(Blue)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    x1, y1, x2, y2 = meta["bbox_margin_xyxy"]
    roi_view = intensity_full[max(0, y1 - 5):min(y2 + 5, 128), max(0, x1 - 5):min(x2 + 5, 128)]
    plt.imshow(roi_view, cmap="gray")
    plt.title("ROI Region Detail")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    info_text = (
        f"Frame Index: {int(meta['mask_name'].split('.')[0]) - 1}\n"
        f"BBox (xyxy): {meta['bbox_margin_xyxy']}\n"
        f"ROI Orig Shape: {meta['resize_meta']['orig_shape']}\n"
        f"Stage 2 Status: Refinement Complete"
    )
    plt.text(0.1, 0.5, info_text, fontsize=12, family="monospace")
    plt.axis("off")
    plt.title("Metadata & Status")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_one_fold(seed, paths=None):
    set_seed(seed)
    paths = paths or {
        "train_path": Path(conf.TRAIN_PATH),
        "val_path": Path(conf.VAL_PATH),
        "test_path": Path(conf.TEST_PATH),
        "save_path": Path(conf.SAVE_PATH),
        "vis_dir": Path(conf.TEST_VIS_PATH),
        "log_path": Path(conf.SAVE_PATH).with_suffix(".log"),
        "metrics_path": Path(conf.SAVE_PATH).with_name("metrics.json"),
    }
    paths = {key: Path(value) for key, value in paths.items()}
    paths["save_path"].parent.mkdir(parents=True, exist_ok=True)

    with tee_stdout(paths["log_path"]):
        print(f"\n===== Stage2 fold seed {seed} =====")
        train_set = Stage2Dataset(str(paths["train_path"]), conf.INPUT_ITEMS)
        val_set = Stage2Dataset(str(paths["val_path"]), conf.INPUT_ITEMS)
        test_set = Stage2Dataset(str(paths["test_path"]), conf.INPUT_ITEMS)

        print(f"Train ROIs: {len(train_set)}")
        print(f"Val ROIs  : {len(val_set)}")
        print(f"Test ROIs : {len(test_set)}")

        train_loader = DataLoader(train_set, batch_size=conf.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=conf.BATCH_SIZE)
        test_loader = DataLoader(test_set, batch_size=conf.BATCH_SIZE)

        model = Stage2UNet().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.LR)
        criterion = torch.nn.CrossEntropyLoss()

        best_iou = -1.0
        for epoch in range(conf.EPOCHS):
            model.train()
            total_loss = 0.0

            for x, y, _ in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_iou, val_precision, val_recall, val_dice = evaluate(model, val_loader, DEVICE)

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), paths["save_path"])
                print(">>> Saved best model")

            print(
                f"Epoch [{epoch + 1}/{conf.EPOCHS}]  "
                f"Loss: {total_loss:.4f}  "
                f"Val IoU: {val_iou:.3f}  "
                f"Val Precision: {val_precision:.3f}  "
                f"Val Recall: {val_recall:.3f}  "
                f"Val Dice: {val_dice:.3f}"
            )

        print("\nLoading best model for TEST...")
        model = Stage2UNet().to(DEVICE)
        model.load_state_dict(torch.load(paths["save_path"], map_location=DEVICE))
        model.eval()

        full_iou, full_precision, full_recall, full_dice = evaluate_full_scale(
            model,
            test_loader,
            DEVICE,
            conf.MASK_DIR,
            conf.RAW_INTENSITY_PATH,
            save_dir=str(paths["vis_dir"]),
        )

        print("\n===== FINAL TEST RESULT =====")
        print("GLOBAL ACCURACY:")
        print(f"IoU: {full_iou:.4f}")
        print(f"Precision: {full_precision:.4f}")
        print(f"Recall: {full_recall:.4f}")
        print(f"Dice: {full_dice:.4f}")

        result = {
            "seed": seed,
            "iou": full_iou,
            "precision": full_precision,
            "recall": full_recall,
            "dice": full_dice,
            "best_val_iou": best_iou,
        }
        save_json(paths["metrics_path"], result)
        return result


def train_all_folds():
    results = []
    for seed in kfold.KFOLD_SEEDS:
        roi_root = kfold.stage2_roi_root(seed)
        results.append(
            train_one_fold(
                seed,
                paths={
                    "train_path": roi_root / "roi_train",
                    "val_path": roi_root / "roi_val",
                    "test_path": roi_root / "roi_test",
                    "save_path": kfold.stage2_model_path(seed),
                    "vis_dir": kfold.stage2_vis_dir(seed),
                    "log_path": kfold.stage2_dir(seed) / "train.log",
                    "metrics_path": kfold.stage2_dir(seed) / "metrics.json",
                },
            )
        )

    summary = summarize_metrics(results, ["iou", "precision", "recall", "dice"])
    save_json(kfold.RUNS_DIR / "summary_stage2.json", summary)
    print_summary("Stage2 K-Fold Summary", summary, ["iou", "precision", "recall", "dice"])
    return summary


if __name__ == "__main__":
    train_all_folds()
