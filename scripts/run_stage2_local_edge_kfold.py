import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.stage2_local_edge.config_stage2_local_edge as conf
from configs import kfold_config as kfold
from datasets.dataset_stage2_local_edge import Stage2LocalEdgeDataset
from engine.stage1.loss_stage1 import SegmentationLoss
from models.model_stage1 import UNet
from scripts.kfold_utils import print_summary, save_json, set_seed, summarize_metrics, tee_stdout
from scripts.train_stage1_only_strong import evaluate_thresholds, select_best_threshold


DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def fold_dir(runs_dir, seed):
    return Path(runs_dir) / kfold.fold_name(seed)


def prepare_run(source_runs_dir, runs_dir, seeds):
    source_runs_dir = Path(source_runs_dir)
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        source_fold = fold_dir(source_runs_dir, seed)
        target_fold = fold_dir(runs_dir, seed)
        target_fold.mkdir(parents=True, exist_ok=True)
        for name in ("stage1_only_strong", "stage1_strong_priors"):
            source = source_fold / name
            target = target_fold / name
            if not source.exists():
                raise FileNotFoundError(f"Missing source artifact: {source}")
            shutil.copytree(source, target, dirs_exist_ok=True)
        print(f"Prepared fixed Stage1 artifacts for seed {seed}: {target_fold}")

    summary_source = source_runs_dir / "summary_stage1_only_strong.json"
    if summary_source.exists():
        shutil.copy2(summary_source, runs_dir / summary_source.name)

    save_json(
        runs_dir / "stage2_local_edge_experiment.json",
        {
            "source_runs_dir": str(source_runs_dir),
            "runs_dir": str(runs_dir),
            "seeds": [int(seed) for seed in seeds],
            "experiments": {
                "C1": ["intensity", "local_depth_edge"],
                "C2": ["intensity", "local_depth_edge", "prob", "roi_mask"],
                "C3": [
                    "intensity",
                    "local_depth_edge",
                    "local_depth_edge_roi",
                    "prob",
                    "roi_mask",
                ],
            },
        },
    )


def make_dataset(indices, experiment, prior_dir, edge_erode_iterations):
    return Stage2LocalEdgeDataset(
        raw_items=kfold.RAW_ITEMS,
        mask_dir=kfold.LABEL_DIR,
        indices=indices,
        experiment=experiment,
        prior_dir=prior_dir,
        edge_erode_iterations=edge_erode_iterations,
    )


def compute_loss(logits, labels, roi_mask, criterion, experiment, roi_loss_weight):
    full_loss = criterion(logits, labels)
    if experiment == "C1" or roi_loss_weight <= 0:
        return full_loss
    roi_loss = criterion(logits, labels, valid_mask=roi_mask)
    return full_loss + roi_loss_weight * roi_loss


def train_one_fold(seed, experiment, args):
    set_seed(seed)
    train_indices, val_indices, test_indices = kfold.load_split_indices(seed)
    current_fold_dir = fold_dir(args.runs_dir, seed)
    prior_dir = current_fold_dir / "stage1_strong_priors"
    out_dir = current_fold_dir / "stage2_local_edge" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_arg = prior_dir if experiment in {"C2", "C3"} else None
    train_set = make_dataset(train_indices, experiment, prior_arg, args.edge_erode_iterations)
    val_set = make_dataset(val_indices, experiment, prior_arg, args.edge_erode_iterations)
    test_set = make_dataset(test_indices, experiment, prior_arg, args.edge_erode_iterations)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    with tee_stdout(out_dir / "train.log"):
        print(f"\n===== Stage2 local-edge {experiment}, seed {seed} =====")
        print(f"Input items: {train_set.input_items}")
        print(f"Input channels: {train_set.input_channels}")
        print(f"Uses Stage1 priors: {train_set.uses_stage1_priors}")
        print(f"Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}")
        print(f"ROI loss weight: {0.0 if experiment == 'C1' else args.roi_loss_weight}")

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
            for x, y, roi_mask in train_loader:
                x = x.to(conf.DEVICE)
                y = y.to(conf.DEVICE)
                roi_mask = roi_mask.to(conf.DEVICE)

                optimizer.zero_grad()
                logits = model(x)
                loss = compute_loss(
                    logits,
                    y,
                    roi_mask,
                    criterion,
                    experiment,
                    args.roi_loss_weight,
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_by_threshold = evaluate_thresholds(model, val_loader_without_roi(val_loader), conf.DEVICE, args.thresholds)
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
            val_loader_without_roi(test_loader),
            conf.DEVICE,
            [best_threshold],
        )[best_threshold]
        result = {
            "seed": int(seed),
            "experiment": experiment,
            "input_items": train_set.input_items,
            "input_channels": train_set.input_channels,
            "uses_stage1_priors": train_set.uses_stage1_priors,
            "target_weight": float(args.target_weight),
            "dice_weight": float(args.dice_weight),
            "roi_loss_weight": float(0.0 if experiment == "C1" else args.roi_loss_weight),
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


def val_loader_without_roi(loader):
    for x, y, _ in loader:
        yield x, y


def run_experiment(experiment, args):
    results = [train_one_fold(seed, experiment, args) for seed in args.seeds]
    metric_keys = ["iou", "precision", "recall", "dice", "coverage", "pred_area"]
    summary = summarize_metrics(results, metric_keys)
    out_path = Path(args.runs_dir) / f"summary_stage2_local_edge_{experiment}.json"
    save_json(out_path, summary)
    print_summary(f"Stage2 Local Edge {experiment}", summary, metric_keys)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run C1/C2/C3 local-depth-edge Stage2 experiments.")
    parser.add_argument("--stage", choices=["prepare", "train", "all"], default="all")
    parser.add_argument("--experiment", choices=["C1", "C2", "C3", "all"], default="all")
    parser.add_argument("--source-runs-dir", type=Path, default=kfold.DATA_ROOT / "runs/run8")
    parser.add_argument("--runs-dir", type=Path, default=kfold.DATA_ROOT / "runs/run10")
    parser.add_argument("--seeds", type=int, nargs="+", default=kfold.KFOLD_SEEDS)
    parser.add_argument("--epochs", type=int, default=conf.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=conf.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=conf.LR)
    parser.add_argument("--target-weight", type=float, default=conf.TARGET_WEIGHT)
    parser.add_argument("--dice-weight", type=float, default=conf.DICE_WEIGHT)
    parser.add_argument("--roi-loss-weight", type=float, default=conf.ROI_LOSS_WEIGHT)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--edge-erode-iterations", type=int, default=conf.EDGE_ERODE_ITERATIONS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage in {"prepare", "all"}:
        prepare_run(args.source_runs_dir, args.runs_dir, args.seeds)
    if args.stage in {"train", "all"}:
        experiments = ["C1", "C2", "C3"] if args.experiment == "all" else [args.experiment]
        for experiment in experiments:
            run_experiment(experiment, args)


if __name__ == "__main__":
    main()
