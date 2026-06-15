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

import configs.new_data.config_new_data as conf
from datasets.dataset_new_data import NewDataSegmentationDataset
from engine.stage1.loss_stage1 import SegmentationLoss
from models.model_dual_branch import DualBranchGatedUNet
from models.model_stage1 import UNet
from scripts.new_data_test_reporting import evaluate_and_report_test_set
from scripts.kfold_utils import print_summary, save_json, set_seed, summarize_metrics, tee_stdout
from scripts.train_stage1_only_strong import evaluate_thresholds, select_best_threshold


DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
EXPERIMENTS = ("I0", "ID", "IDE", "C1", "C1L", "D1", "D3")


def load_fold_indices(index_dir, fold):
    index_dir = Path(index_dir)
    return tuple(
        np.load(index_dir / f"fold_{fold}_{split}_indices.npy").astype(int).tolist()
        for split in ("train", "val", "test")
    )


def make_dataset(indices, experiment):
    return NewDataSegmentationDataset(
        raw_items=conf.RAW_ITEMS,
        label_dir=conf.LABEL_DIR,
        indices=indices,
        experiment=experiment,
    )


def make_model(experiment, input_channels, args):
    if experiment == "C1L":
        return UNet(
            in_channels=input_channels,
            num_classes=2,
            base_channels=args.light_unet_base_channels,
        )
    if experiment in {"I0", "ID", "IDE", "C1"}:
        return UNet(in_channels=input_channels, num_classes=2)
    fusion_mode = "no_gate" if experiment == "D1" else "multiscale_gate"
    return DualBranchGatedUNet(
        num_classes=2,
        intensity_base_channels=args.intensity_base_channels,
        edge_base_channels=args.edge_base_channels,
        fusion_mode=fusion_mode,
    )


def train_one_fold(fold, experiment, args):
    seed = conf.FOLD_SEEDS[fold]
    set_seed(seed)
    train_indices, val_indices, test_indices = load_fold_indices(args.index_dir, fold)
    out_dir = Path(args.runs_dir) / f"fold_{fold}" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = make_dataset(train_indices, experiment)
    val_set = make_dataset(val_indices, experiment)
    test_set = make_dataset(test_indices, experiment)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    with tee_stdout(out_dir / "train.log"):
        print(f"\n===== New data {experiment}, fold {fold}, seed {seed} =====")
        print(f"Input items: {train_set.input_items}")
        print(f"Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}")

        model = make_model(experiment, train_set.input_channels, args).to(conf.DEVICE)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"Parameters: {parameter_count:,}")
        criterion = SegmentationLoss(
            target_weight=args.target_weight,
            ce_weight=conf.CE_WEIGHT,
            dice_weight=args.dice_weight,
        ).to(conf.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_iou = -1.0
        best_threshold = None
        best_val_metrics = None
        best_epoch = None
        epochs_without_improvement = 0
        best_model_path = out_dir / "best_model.pth"

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x = x.to(conf.DEVICE, non_blocking=True)
                y = y.to(conf.DEVICE, non_blocking=True)
                if experiment in {"D1", "D3"} and args.edge_dropout > 0:
                    drop = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < args.edge_dropout
                    x = x.clone()
                    x[:, 1:2] = x[:, 1:2] * (~drop)

                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_by_threshold = evaluate_thresholds(
                model, val_loader, conf.DEVICE, args.thresholds
            )
            threshold, val_metrics = select_best_threshold(val_by_threshold, key="iou")
            print(
                f"Epoch [{epoch + 1}/{args.epochs}]  "
                f"Loss: {total_loss / max(1, len(train_loader)):.4f}  "
                f"Thr: {threshold:g}  Val IoU: {val_metrics['iou']:.4f}  "
                f"Dice: {val_metrics['dice']:.4f}  "
                f"Precision: {val_metrics['precision']:.4f}  "
                f"Recall: {val_metrics['recall']:.4f}"
            )
            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                best_threshold = threshold
                best_val_metrics = val_metrics
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                save_json(out_dir / "val_threshold_metrics.json", val_by_threshold)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    break

        model.load_state_dict(torch.load(best_model_path, map_location=conf.DEVICE))
        test_metrics = evaluate_thresholds(
            model, test_loader, conf.DEVICE, [best_threshold]
        )[best_threshold]
        detailed_test_summary = evaluate_and_report_test_set(
            model=model,
            dataset=test_set,
            raw_items=conf.RAW_ITEMS,
            label_dir=conf.LABEL_DIR,
            device=conf.DEVICE,
            threshold=best_threshold,
            output_dir=out_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_visualizations=not args.skip_test_visualizations,
        )
        result = {
            "fold": int(fold),
            "seed": int(seed),
            "experiment": experiment,
            "input_items": train_set.input_items,
            "parameters": int(parameter_count),
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "test_samples": len(test_set),
            "target_weight": float(args.target_weight),
            "dice_weight": float(args.dice_weight),
            "edge_dropout": float(args.edge_dropout if experiment in {"D1", "D3"} else 0.0),
            "lr": float(args.lr),
            "max_epochs": int(args.epochs),
            "best_epoch": int(best_epoch),
            "best_threshold": float(best_threshold),
            "best_val_iou": float(best_val_iou),
            "best_val_metrics": best_val_metrics,
            "detailed_test_summary": detailed_test_summary,
            **test_metrics,
        }
        save_json(out_dir / "metrics.json", result)
        print(
            f"Test IoU: {result['iou']:.4f}  Dice: {result['dice']:.4f}  "
            f"Precision: {result['precision']:.4f}  Recall: {result['recall']:.4f}"
        )
        return result


def run_experiment(experiment, args):
    results = [train_one_fold(fold, experiment, args) for fold in args.folds]
    metric_keys = ["iou", "precision", "recall", "dice", "coverage", "pred_area"]
    summary = summarize_metrics(results, metric_keys)
    detailed_keys = [
        "foreground_mean_image_iou",
        "foreground_median_image_iou",
        "foreground_iou_std",
        "background_false_positive_frame_rate",
        "background_mean_pred_area",
        "background_pixel_false_positive_rate",
    ]
    summary["detailed_mean"] = {
        key: float(
            np.mean([row["detailed_test_summary"][key] for row in results])
        )
        for key in detailed_keys
    }
    summary["detailed_std"] = {
        key: float(
            np.std([row["detailed_test_summary"][key] for row in results])
        )
        for key in detailed_keys
    }
    save_json(Path(args.runs_dir) / f"summary_{experiment}.json", summary)
    print_summary(f"New Data {experiment}", summary, metric_keys)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the most useful baselines and dual-branch models on new data."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=[*EXPERIMENTS, "all", "core"],
        default=["core"],
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(conf.NUM_FOLDS)))
    parser.add_argument("--index-dir", type=Path, default=conf.INDEX_DIR)
    parser.add_argument("--runs-dir", type=Path, default=conf.RUNS_DIR)
    parser.add_argument("--epochs", type=int, default=conf.EPOCHS)
    parser.add_argument("--patience", type=int, default=conf.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=conf.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=conf.LR)
    parser.add_argument("--target-weight", type=float, default=conf.TARGET_WEIGHT)
    parser.add_argument("--dice-weight", type=float, default=conf.DICE_WEIGHT)
    parser.add_argument("--edge-dropout", type=float, default=conf.EDGE_DROPOUT)
    parser.add_argument("--intensity-base-channels", type=int, default=conf.INTENSITY_BASE_CHANNELS)
    parser.add_argument("--edge-base-channels", type=int, default=conf.EDGE_BASE_CHANNELS)
    parser.add_argument(
        "--light-unet-base-channels",
        type=int,
        default=conf.LIGHT_UNET_BASE_CHANNELS,
    )
    parser.add_argument(
        "--skip-test-visualizations",
        action="store_true",
        help="Skip six-panel PNGs; predictions and per-image metrics are still saved.",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(fold < 0 or fold >= conf.NUM_FOLDS for fold in args.folds):
        raise ValueError(f"Folds must be in [0, {conf.NUM_FOLDS - 1}].")
    if "all" in args.experiments:
        experiments = list(EXPERIMENTS)
    elif "core" in args.experiments:
        experiments = ["I0", "C1", "D1", "D3"]
    else:
        experiments = args.experiments
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        run_experiment(experiment, args)


if __name__ == "__main__":
    main()
