import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.stage1.config_stage1 as conf
from configs import kfold_config as kfold
from datasets.dataset_stage1_input_ablation import Stage1InputAblationDataset
from engine.stage1.loss_stage1 import SegmentationLoss
from models.model_stage1 import UNet
from scripts.kfold_utils import print_summary, save_json, set_seed, summarize_metrics, tee_stdout
from scripts.train_stage1_only_strong import evaluate_thresholds, select_best_threshold


DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def make_dataset(indices, experiment, edge_erode_iterations):
    return Stage1InputAblationDataset(
        raw_items=kfold.RAW_ITEMS,
        mask_dir=kfold.LABEL_DIR,
        indices=indices,
        experiment=experiment,
        edge_erode_iterations=edge_erode_iterations,
    )


def train_one_fold(seed, experiment, args):
    set_seed(seed)
    train_indices, val_indices, test_indices = kfold.load_split_indices(seed)
    fold_dir = Path(args.runs_dir) / kfold.fold_name(seed)
    out_dir = fold_dir / "stage1_input_ablation" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = make_dataset(train_indices, experiment, args.edge_erode_iterations)
    val_set = make_dataset(val_indices, experiment, args.edge_erode_iterations)
    test_set = make_dataset(test_indices, experiment, args.edge_erode_iterations)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    with tee_stdout(out_dir / "train.log"):
        print(f"\n===== Stage1 input ablation {experiment}, seed {seed} =====")
        print(f"Input items: {train_set.input_items}")
        print(f"Input channels: {train_set.input_channels}")
        print(f"Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}")
        if experiment == "B":
            print(f"Depth valid-mask erosion iterations: {args.edge_erode_iterations}")

        model = UNet(in_channels=train_set.input_channels, num_classes=2).to(conf.DEVICE)
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
            for x, y in train_loader:
                x = x.to(conf.DEVICE)
                y = y.to(conf.DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_by_threshold = evaluate_thresholds(model, val_loader, conf.DEVICE, args.thresholds)
            threshold, val_metrics = select_best_threshold(val_by_threshold, key="iou")
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
            "experiment": experiment,
            "input_items": train_set.input_items,
            "input_channels": train_set.input_channels,
            "target_weight": float(args.target_weight),
            "dice_weight": float(args.dice_weight),
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "edge_erode_iterations": int(args.edge_erode_iterations),
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


def run_experiment(experiment, args):
    results = [train_one_fold(seed, experiment, args) for seed in args.seeds]
    metric_keys = ["iou", "precision", "recall", "dice", "coverage", "pred_area"]
    summary = summarize_metrics(results, metric_keys)
    out_path = Path(args.runs_dir) / f"summary_stage1_input_ablation_{experiment}.json"
    save_json(out_path, summary)
    print_summary(f"Stage1 Input Ablation {experiment}", summary, metric_keys)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-hyperparameter Stage1 input ablations A and B.")
    parser.add_argument("--experiment", choices=["A", "B", "all"], default="all")
    parser.add_argument("--runs-dir", type=Path, default=kfold.RUNS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=kfold.KFOLD_SEEDS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-weight", type=float, default=5.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--edge-erode-iterations", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    experiments = ["A", "B"] if args.experiment == "all" else [args.experiment]
    for experiment in experiments:
        run_experiment(experiment, args)


if __name__ == "__main__":
    main()
