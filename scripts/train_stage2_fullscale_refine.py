import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.stage2_fullscale.config_stage2_fullscale as conf
from configs import kfold_config as kfold
from datasets.dataset_stage2_fullscale import Stage2FullScaleDataset
from engine.stage2.loss_refine import ErrorFocusedRefineLoss
from models.model_stage2_refine import Stage2RefineUNet
from scripts.kfold_utils import print_summary, save_json, set_seed, summarize_metrics, tee_stdout


DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def compute_metrics_from_logits(logits, labels, threshold):
    probs = torch.softmax(logits, dim=1)[:, 1]
    preds = probs > threshold
    labels = labels.bool()

    tp = int((preds & labels).sum().item())
    fp = int((preds & ~labels).sum().item())
    fn = int((~preds & labels).sum().item())
    pred_area = int(preds.sum().item())
    label_area = int(labels.sum().item())
    return tp, fp, fn, pred_area, label_area


def metrics_from_counts(tp, fp, fn, pred_area, label_area):
    return {
        "iou": float(tp / (tp + fp + fn + 1e-6)),
        "precision": float(tp / (tp + fp + 1e-6)),
        "recall": float(tp / (tp + fn + 1e-6)),
        "dice": float(2 * tp / (2 * tp + fp + fn + 1e-6)),
        "coverage": float(tp / (tp + fn + 1e-6)),
        "pred_area": float(pred_area),
        "label_area": float(label_area),
    }


def refine_logits(model, x, base_logits, roi_mask):
    delta_logits = model(x)
    if conf.GATE_RESIDUAL_WITH_ROI:
        delta_logits = delta_logits * roi_mask.unsqueeze(1)
    return base_logits + delta_logits, delta_logits


def evaluate_thresholds(model, loader, device, thresholds):
    model.eval()
    totals = {
        float(threshold): {"tp": 0, "fp": 0, "fn": 0, "pred_area": 0, "label_area": 0}
        for threshold in thresholds
    }

    with torch.no_grad():
        for x, y, base_logits, roi_mask, _ in loader:
            x = x.to(device)
            y = y.to(device)
            base_logits = base_logits.to(device)
            roi_mask = roi_mask.to(device)
            final_logits, _ = refine_logits(model, x, base_logits, roi_mask)
            for threshold in totals:
                tp, fp, fn, pred_area, label_area = compute_metrics_from_logits(final_logits, y, threshold)
                totals[threshold]["tp"] += tp
                totals[threshold]["fp"] += fp
                totals[threshold]["fn"] += fn
                totals[threshold]["pred_area"] += pred_area
                totals[threshold]["label_area"] += label_area

    return {
        threshold: metrics_from_counts(
            row["tp"],
            row["fp"],
            row["fn"],
            row["pred_area"],
            row["label_area"],
        )
        for threshold, row in totals.items()
    }


def select_best_threshold(threshold_metrics):
    return max(
        threshold_metrics.items(),
        key=lambda item: (item[1]["iou"], item[1]["dice"], item[1]["recall"]),
    )


def make_dataset(indices, prior_dir):
    return Stage2FullScaleDataset(
        raw_items=conf.RAW_ITEMS,
        prior_dir=prior_dir,
        mask_dir=conf.MASK_DIR,
        indices=indices,
        input_items=conf.INPUT_ITEMS,
    )


def load_prior_threshold(prior_dir):
    meta_path = Path(prior_dir) / "meta.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        return float(json.load(handle)["threshold"])


def diagnose_refinement(model, loader, device, threshold):
    totals = {
        "changed_pixels": 0,
        "repaired_pixels": 0,
        "repaired_fn": 0,
        "repaired_fp": 0,
        "damaged_pixels": 0,
        "damaged_tp": 0,
        "damaged_tn": 0,
        "changed_inside_roi": 0,
        "changed_outside_roi": 0,
    }

    model.eval()
    with torch.no_grad():
        for x, y, base_logits, roi_mask, _ in loader:
            x = x.to(device)
            y = y.to(device).bool()
            base_logits = base_logits.to(device)
            roi_mask = roi_mask.to(device).bool()
            final_logits, _ = refine_logits(model, x, base_logits, roi_mask.float())

            base_pred = torch.softmax(base_logits, dim=1)[:, 1] > threshold
            final_pred = torch.softmax(final_logits, dim=1)[:, 1] > threshold
            changed = base_pred != final_pred
            base_correct = base_pred == y
            final_correct = final_pred == y

            repaired = (~base_correct) & final_correct
            damaged = base_correct & (~final_correct)
            totals["changed_pixels"] += int(changed.sum().item())
            totals["repaired_pixels"] += int(repaired.sum().item())
            totals["repaired_fn"] += int((repaired & y).sum().item())
            totals["repaired_fp"] += int((repaired & ~y).sum().item())
            totals["damaged_pixels"] += int(damaged.sum().item())
            totals["damaged_tp"] += int((damaged & y).sum().item())
            totals["damaged_tn"] += int((damaged & ~y).sum().item())
            totals["changed_inside_roi"] += int((changed & roi_mask).sum().item())
            totals["changed_outside_roi"] += int((changed & ~roi_mask).sum().item())

    totals["net_repaired_pixels"] = totals["repaired_pixels"] - totals["damaged_pixels"]
    return totals


def visualize_predictions(model, loader, device, threshold, out_dir, max_count=8):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    count = 0

    with torch.no_grad():
        for x, y, base_logits, roi_mask, label_indices in loader:
            x = x.to(device)
            base_logits = base_logits.to(device)
            roi_mask_device = roi_mask.to(device)
            final_logits, _ = refine_logits(model, x, base_logits, roi_mask_device)
            stage1_pred = (torch.softmax(base_logits, dim=1)[:, 1] > threshold).long().cpu().numpy()
            final_pred = (torch.softmax(final_logits, dim=1)[:, 1] > threshold).long().cpu().numpy()
            x_np = x.cpu().numpy()
            y_np = y.numpy()
            roi_np = roi_mask.numpy()

            for i in range(x_np.shape[0]):
                if count >= max_count:
                    return
                fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                panels = [
                    ("intensity", x_np[i, 0], "gray"),
                    ("roi", roi_np[i], "gray"),
                    ("stage1", stage1_pred[i], "gray"),
                    ("stage2", final_pred[i], "gray"),
                    ("gt", y_np[i], "gray"),
                ]
                for ax, (title, image, cmap) in zip(axes, panels):
                    ax.imshow(image, cmap=cmap, interpolation="nearest")
                    ax.set_title(title)
                    ax.axis("off")
                plt.tight_layout()
                plt.savefig(out_dir / f"sample_{int(label_indices[i]):03d}.png", dpi=160)
                plt.close(fig)
                count += 1


def train_one_fold(seed, args):
    set_seed(seed)
    train_indices, val_indices, test_indices = kfold.load_split_indices(seed)
    fold_dir = Path(args.runs_dir) / kfold.fold_name(seed)
    prior_dir = fold_dir / "stage1_strong_priors"
    out_dir = fold_dir / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with tee_stdout(out_dir / "train.log"):
        print(f"\n===== Stage2 full-scale refine fold seed {seed} =====")
        print(f"Train samples: {len(train_indices)}")
        print(f"Val samples  : {len(val_indices)}")
        print(f"Test samples : {len(test_indices)}")
        print(f"Prior dir    : {prior_dir}")
        print(f"Input items  : {conf.INPUT_ITEMS}")
        base_threshold = load_prior_threshold(prior_dir)
        print(f"Stage1 threshold: {base_threshold:g}")

        train_loader = DataLoader(make_dataset(train_indices, prior_dir), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(make_dataset(val_indices, prior_dir), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(make_dataset(test_indices, prior_dir), batch_size=args.batch_size, shuffle=False)

        model = Stage2RefineUNet(
            in_channels=conf.INPUT_CHANNEL,
            num_classes=2,
            zero_init_head=conf.ZERO_INIT_RESIDUAL_HEAD,
        ).to(conf.DEVICE)
        criterion = ErrorFocusedRefineLoss(
            target_weight=conf.TARGET_WEIGHT,
            ce_weight=conf.CE_WEIGHT,
            dice_weight=conf.DICE_WEIGHT,
            error_pixel_weight=conf.ERROR_PIXEL_WEIGHT,
            correct_pixel_weight=conf.CORRECT_PIXEL_WEIGHT,
            correct_delta_reg_weight=conf.CORRECT_DELTA_REG_WEIGHT,
        ).to(conf.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_iou = -1.0
        best_threshold = None
        best_val_metrics = None
        best_model_path = out_dir / "best_model.pth"

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0

            for x, y, base_logits, roi_mask, _ in train_loader:
                x = x.to(conf.DEVICE)
                y = y.to(conf.DEVICE)
                base_logits = base_logits.to(conf.DEVICE)
                roi_mask = roi_mask.to(conf.DEVICE)

                final_logits, delta_logits = refine_logits(model, x, base_logits, roi_mask)
                loss = criterion(
                    final_logits,
                    delta_logits,
                    base_logits,
                    y,
                    roi_mask,
                    base_threshold,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            val_by_threshold = evaluate_thresholds(model, val_loader, conf.DEVICE, args.thresholds)
            epoch_threshold, epoch_val_metrics = select_best_threshold(val_by_threshold)
            avg_loss = epoch_loss / max(1, len(train_loader))

            print(
                f"Epoch [{epoch + 1}/{args.epochs}]  "
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
        refinement_diagnostics = diagnose_refinement(
            model,
            test_loader,
            conf.DEVICE,
            best_threshold,
        )
        visualize_predictions(
            model,
            test_loader,
            conf.DEVICE,
            best_threshold,
            out_dir / "visualizations",
            max_count=args.visualize_num,
        )

        result = {
            "seed": int(seed),
            "best_threshold": float(best_threshold),
            "best_val_iou": float(best_val_iou),
            "best_val_metrics": best_val_metrics,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "input_items": list(conf.INPUT_ITEMS),
            "gate_residual_with_roi": bool(conf.GATE_RESIDUAL_WITH_ROI),
            "zero_init_residual_head": bool(conf.ZERO_INIT_RESIDUAL_HEAD),
            "stage1_threshold": float(base_threshold),
            "error_pixel_weight": float(conf.ERROR_PIXEL_WEIGHT),
            "correct_pixel_weight": float(conf.CORRECT_PIXEL_WEIGHT),
            "correct_delta_reg_weight": float(conf.CORRECT_DELTA_REG_WEIGHT),
            "refinement_diagnostics": refinement_diagnostics,
            **test_metrics,
        }
        save_json(out_dir / "metrics.json", result)
        print("\n===== Final Test Result =====")
        print(
            f"IoU: {result['iou']:.4f}  "
            f"Dice: {result['dice']:.4f}  "
            f"Precision: {result['precision']:.4f}  "
            f"Recall: {result['recall']:.4f}"
        )
        print(
            "Refinement pixels: "
            f"repaired={refinement_diagnostics['repaired_pixels']}  "
            f"damaged={refinement_diagnostics['damaged_pixels']}  "
            f"net={refinement_diagnostics['net_repaired_pixels']}  "
            f"outside_roi={refinement_diagnostics['changed_outside_roi']}"
        )
        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train full-scale Stage2 residual refinement.")
    parser.add_argument("--runs-dir", type=Path, default=kfold.RUNS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=kfold.KFOLD_SEEDS)
    parser.add_argument("--epochs", type=int, default=conf.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=conf.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=conf.LR)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--visualize-num", type=int, default=8)
    parser.add_argument("--output-name", default="stage2_fullscale_refine_error_focused")
    return parser.parse_args()


def main():
    args = parse_args()
    results = [train_one_fold(seed, args) for seed in args.seeds]
    metric_keys = ["iou", "precision", "recall", "dice", "coverage", "pred_area"]
    summary = summarize_metrics(results, metric_keys)
    save_json(Path(args.runs_dir) / f"summary_{args.output_name}.json", summary)
    print_summary("Stage2 Full-Scale Refine K-Fold Summary", summary, metric_keys)


if __name__ == "__main__":
    main()
