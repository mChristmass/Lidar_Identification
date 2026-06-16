import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.reliability_gate.config_reliability_gate as conf
from datasets.dataset_reliability_gate import ReliabilityGateDataset
from engine.stage1.loss_stage1 import SegmentationLoss
from models.model_reliability_gate import ReliabilityGuidedEdgeUNet
from models.model_stage1 import UNet
from scripts.kfold_utils import print_summary, save_json, set_seed, summarize_metrics, tee_stdout
from scripts.reliability_gate_reporting import (
    estimate_conv_flops,
    evaluate_and_report_reliability_model,
    measure_latency,
)
from scripts.train_stage1_only_strong import (
    compute_confusion_from_probs,
    metrics_from_confusion,
    select_best_threshold,
)


DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
EXPERIMENTS = ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7")
EXPERIMENT_SPECS = {
    "R0": {
        "description": "C1L parameter-matched baseline",
        "gate_supervision": False,
        "use_raw_depth": False,
        "edge_keep_alpha": 1.0,
        "modulation_mode": "suppress",
        "gate_beta": 0.0,
    },
    "R1": {
        "description": "Reliability gate without explicit gate supervision",
        "gate_supervision": False,
        "use_raw_depth": True,
        "edge_keep_alpha": conf.EDGE_KEEP_ALPHA,
        "modulation_mode": "suppress",
        "gate_beta": 0.0,
    },
    "R2": {
        "description": "Supervised raw-depth edge reliability gate",
        "gate_supervision": True,
        "use_raw_depth": True,
        "edge_keep_alpha": conf.EDGE_KEEP_ALPHA,
        "modulation_mode": "suppress",
        "gate_beta": 0.0,
    },
    "R3": {
        "description": "Supervised reliability gate without raw depth",
        "gate_supervision": True,
        "use_raw_depth": False,
        "edge_keep_alpha": conf.EDGE_KEEP_ALPHA,
        "modulation_mode": "suppress",
        "gate_beta": 0.0,
    },
    "R4": {
        "description": "Supervised reliability gate without minimum edge retention",
        "gate_supervision": True,
        "use_raw_depth": True,
        "edge_keep_alpha": 0.0,
        "modulation_mode": "suppress",
        "gate_beta": 0.0,
    },
    "R5": {
        "description": "Supervised raw-depth residual edge modulation, beta=0.2",
        "gate_supervision": True,
        "use_raw_depth": True,
        "edge_keep_alpha": 1.0,
        "modulation_mode": "residual",
        "gate_beta": 0.2,
    },
    "R6": {
        "description": "Supervised raw-depth residual edge modulation, beta=0.3",
        "gate_supervision": True,
        "use_raw_depth": True,
        "edge_keep_alpha": 1.0,
        "modulation_mode": "residual",
        "gate_beta": 0.3,
    },
    "R7": {
        "description": "Unsupervised raw-depth residual edge modulation, beta=0.2",
        "gate_supervision": False,
        "use_raw_depth": True,
        "edge_keep_alpha": 1.0,
        "modulation_mode": "residual",
        "gate_beta": 0.2,
    },
}


class C1LBaselineAdapter(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        self.model = UNet(in_channels=2, num_classes=2, base_channels=base_channels)

    def forward(self, x):
        return self.model(torch.cat([x[:, 0:1], x[:, 3:4]], dim=1))


def load_fold_indices(index_dir, fold):
    index_dir = Path(index_dir)
    return tuple(
        np.load(index_dir / f"fold_{fold}_{split}_indices.npy").astype(int).tolist()
        for split in ("train", "val", "test")
    )


def make_dataset(indices, args):
    return ReliabilityGateDataset(
        raw_items=conf.RAW_ITEMS,
        label_dir=conf.LABEL_DIR,
        indices=indices,
        boundary_radius=args.boundary_radius,
        edge_threshold=args.edge_threshold,
    )


def make_model(experiment, args):
    spec = EXPERIMENT_SPECS[experiment]
    if experiment == "R0":
        return C1LBaselineAdapter(args.light_unet_base_channels)
    return ReliabilityGuidedEdgeUNet(
        num_classes=2,
        base_channels=args.light_unet_base_channels,
        reliability_base_channels=args.reliability_base_channels,
        edge_keep_alpha=spec["edge_keep_alpha"],
        use_raw_depth=spec["use_raw_depth"],
        modulation_mode=spec["modulation_mode"],
        gate_beta=spec["gate_beta"],
    )


def masked_focal_gate_loss(logits, targets, valid_mask, gamma):
    valid_mask = valid_mask.float()
    valid_count = valid_mask.sum()
    if valid_count.item() == 0:
        return logits.sum() * 0.0

    positives = (targets * valid_mask).sum()
    negatives = ((1.0 - targets) * valid_mask).sum()
    positive_weight = (negatives / (positives + 1e-6)).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
        pos_weight=positive_weight,
    )
    probabilities = torch.sigmoid(logits)
    pt = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
    focal = (1.0 - pt).pow(gamma) * bce
    return (focal * valid_mask).sum() / (valid_count + 1e-6)


def balanced_boundary_loss(logits, targets):
    targets = targets.float()
    positives = targets.sum()
    negatives = targets.numel() - positives
    positive_weight = (negatives / (positives + 1e-6)).clamp(1.0, 20.0)
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
    )


def evaluate_thresholds(model, loader, device, thresholds):
    model.eval()
    totals = {
        float(threshold): {"tp": 0, "fp": 0, "fn": 0, "pred_area": 0, "label_area": 0}
        for threshold in thresholds
    }
    with torch.no_grad():
        for x, labels, _, _, _ in loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probabilities = torch.softmax(model(x), dim=1)[:, 1]
            for threshold in totals:
                values = compute_confusion_from_probs(probabilities, labels, threshold)
                for key, value in zip(
                    ("tp", "fp", "fn", "pred_area", "label_area"),
                    values,
                ):
                    totals[threshold][key] += value
    return {
        threshold: metrics_from_confusion(
            row["tp"], row["fp"], row["fn"], row["pred_area"], row["label_area"]
        )
        for threshold, row in totals.items()
    }


def train_one_fold(fold, experiment, args):
    seed = conf.FOLD_SEEDS[fold]
    set_seed(seed)
    spec = EXPERIMENT_SPECS[experiment]
    train_indices, val_indices, test_indices = load_fold_indices(args.index_dir, fold)
    out_dir = Path(args.runs_dir) / f"fold_{fold}" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = make_dataset(train_indices, args)
    val_set = make_dataset(val_indices, args)
    test_set = make_dataset(test_indices, args)
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
        print(f"\n===== Reliability gate {experiment}, fold {fold}, seed {seed} =====")
        print(spec["description"])
        print(f"Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}")
        print(
            f"Gate supervision: {spec['gate_supervision']}  "
            f"Raw depth for gate: {spec['use_raw_depth']}  "
            f"Edge keep alpha: {spec['edge_keep_alpha']}  "
            f"Mode: {spec['modulation_mode']}  "
            f"Beta: {spec['gate_beta']}"
        )

        model = make_model(experiment, args).to(conf.DEVICE)
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
            totals = {"loss": 0.0, "seg": 0.0, "gate": 0.0, "boundary": 0.0}
            for x, labels, gate_target, gate_valid, boundary_target in train_loader:
                x = x.to(conf.DEVICE, non_blocking=True)
                labels = labels.to(conf.DEVICE, non_blocking=True)
                gate_target = gate_target.to(conf.DEVICE, non_blocking=True)
                gate_valid = gate_valid.to(conf.DEVICE, non_blocking=True)
                boundary_target = boundary_target.to(conf.DEVICE, non_blocking=True)

                optimizer.zero_grad()
                if experiment == "R0":
                    seg_loss = criterion(model(x), labels)
                    gate_loss = seg_loss.new_zeros(())
                    boundary_loss = seg_loss.new_zeros(())
                else:
                    output = model(x, return_aux=True)
                    seg_loss = criterion(output["logits"], labels)
                    boundary_loss = balanced_boundary_loss(
                        output["boundary_logits"], boundary_target
                    )
                    if spec["gate_supervision"]:
                        gate_loss = masked_focal_gate_loss(
                            output["reliability_logits"],
                            gate_target,
                            gate_valid,
                            args.gate_focal_gamma,
                        )
                    else:
                        gate_loss = seg_loss.new_zeros(())

                loss = seg_loss
                if experiment != "R0":
                    loss = loss + args.boundary_loss_weight * boundary_loss
                    loss = loss + args.gate_loss_weight * gate_loss
                loss.backward()
                optimizer.step()

                totals["loss"] += loss.item()
                totals["seg"] += seg_loss.item()
                totals["gate"] += gate_loss.item()
                totals["boundary"] += boundary_loss.item()

            val_by_threshold = evaluate_thresholds(
                model, val_loader, conf.DEVICE, args.thresholds
            )
            threshold, val_metrics = select_best_threshold(val_by_threshold, key="iou")
            batches = max(1, len(train_loader))
            print(
                f"Epoch [{epoch + 1}/{args.epochs}]  "
                f"Loss {totals['loss'] / batches:.4f}  "
                f"Seg {totals['seg'] / batches:.4f}  "
                f"Gate {totals['gate'] / batches:.4f}  "
                f"Boundary {totals['boundary'] / batches:.4f}  "
                f"Thr {threshold:g}  Val IoU {val_metrics['iou']:.4f}"
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
        detailed = evaluate_and_report_reliability_model(
            model=model,
            dataset=test_set,
            raw_items=conf.RAW_ITEMS,
            device=conf.DEVICE,
            threshold=best_threshold,
            output_dir=out_dir,
            batch_size=args.batch_size,
            save_visualizations=not args.skip_test_visualizations,
            is_baseline=experiment == "R0",
        )

        sample = test_set[0][0].unsqueeze(0).to(conf.DEVICE)
        flops = estimate_conv_flops(model, sample)
        latency_ms = measure_latency(
            model,
            sample,
            conf.DEVICE,
            warmup=args.latency_warmup,
            repeats=args.latency_repeats,
        )
        result = {
            "fold": int(fold),
            "seed": int(seed),
            "experiment": experiment,
            "description": spec["description"],
            "parameters": int(parameter_count),
            "estimated_flops": int(flops),
            "latency_ms_batch1": float(latency_ms),
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "test_samples": len(test_set),
            "gate_supervision": bool(spec["gate_supervision"]),
            "use_raw_depth": bool(spec["use_raw_depth"]),
            "edge_keep_alpha": float(spec["edge_keep_alpha"]),
            "modulation_mode": spec["modulation_mode"],
            "gate_beta": float(spec["gate_beta"]),
            "gate_loss_weight": float(args.gate_loss_weight if spec["gate_supervision"] else 0.0),
            "boundary_loss_weight": float(args.boundary_loss_weight if experiment != "R0" else 0.0),
            "boundary_radius": int(args.boundary_radius),
            "edge_threshold": float(args.edge_threshold),
            "lr": float(args.lr),
            "max_epochs": int(args.epochs),
            "best_epoch": int(best_epoch),
            "best_threshold": float(best_threshold),
            "best_val_iou": float(best_val_iou),
            "best_val_metrics": best_val_metrics,
            "detailed_test_summary": detailed,
            "boundary_f1": float(detailed["boundary_f1"]),
            "boundary_iou": float(detailed["boundary_iou"]),
            "boundary_precision": float(detailed["boundary_precision"]),
            "boundary_recall": float(detailed["boundary_recall"]),
            "mean_reliability": float(detailed["mean_reliability"]),
            "reliability_positive_mean": float(detailed["reliability_positive_mean"]),
            "reliability_negative_mean": float(detailed["reliability_negative_mean"]),
            "reliability_separation": float(detailed["reliability_separation"]),
            **test_metrics,
        }
        save_json(out_dir / "metrics.json", result)
        print(
            f"Test IoU {result['iou']:.4f}  Dice {result['dice']:.4f}  "
            f"Boundary F1 {result['boundary_f1']:.4f}  "
            f"Latency {latency_ms:.3f} ms"
        )
        return result


def run_experiment(experiment, args):
    results = [train_one_fold(fold, experiment, args) for fold in args.folds]
    metric_keys = [
        "iou",
        "precision",
        "recall",
        "dice",
        "boundary_f1",
        "boundary_iou",
        "boundary_precision",
        "boundary_recall",
        "mean_reliability",
        "latency_ms_batch1",
    ]
    summary = summarize_metrics(results, metric_keys)
    summary["experiment"] = experiment
    summary["description"] = EXPERIMENT_SPECS[experiment]["description"]
    summary["parameters"] = int(results[0]["parameters"])
    summary["estimated_flops"] = int(results[0]["estimated_flops"])
    save_json(Path(args.runs_dir) / f"summary_{experiment}.json", summary)
    print_summary(f"Reliability Gate {experiment}", summary, metric_keys)
    return summary


def load_per_image_metrics(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return {
            int(row["label_index"]): row
            for row in csv.DictReader(handle)
        }


def summarize_paired_comparisons(args, experiments):
    primary = next(
        (name for name in ("R5", "R6", "R7", "R2") if name in experiments),
        None,
    )
    if primary is None:
        return
    comparisons = {}
    for baseline in experiments:
        if baseline == primary:
            continue
        differences = []
        wins = ties = losses = 0
        compared_images = 0
        for fold in args.folds:
            primary_path = (
                Path(args.runs_dir)
                / f"fold_{fold}"
                / primary
                / "test_per_image_metrics.csv"
            )
            baseline_path = (
                Path(args.runs_dir)
                / f"fold_{fold}"
                / baseline
                / "test_per_image_metrics.csv"
            )
            if not primary_path.exists() or not baseline_path.exists():
                continue
            primary_rows = load_per_image_metrics(primary_path)
            baseline_rows = load_per_image_metrics(baseline_path)
            for label_index in sorted(set(primary_rows) & set(baseline_rows)):
                primary_iou = primary_rows[label_index]["iou"]
                baseline_iou = baseline_rows[label_index]["iou"]
                if primary_iou in ("", None) or baseline_iou in ("", None):
                    continue
                difference = float(primary_iou) - float(baseline_iou)
                differences.append(difference)
                compared_images += 1
                if difference > 1e-9:
                    wins += 1
                elif difference < -1e-9:
                    losses += 1
                else:
                    ties += 1
        if differences:
            values = np.asarray(differences, dtype=np.float64)
            comparisons[f"{primary}_vs_{baseline}"] = {
                "foreground_images": compared_images,
                "mean_image_iou_difference": float(values.mean()),
                "median_image_iou_difference": float(np.median(values)),
                "improved_images": wins,
                "tied_images": ties,
                "worse_images": losses,
                "win_rate": float(wins / compared_images),
            }
    if comparisons:
        save_json(Path(args.runs_dir) / "paired_comparisons.json", comparisons)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run reliability-guided local depth edge experiments."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=[*EXPERIMENTS, "all", "core"],
        default=["all"],
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
    parser.add_argument("--light-unet-base-channels", type=int, default=conf.LIGHT_UNET_BASE_CHANNELS)
    parser.add_argument("--reliability-base-channels", type=int, default=conf.RELIABILITY_BASE_CHANNELS)
    parser.add_argument("--gate-loss-weight", type=float, default=conf.GATE_LOSS_WEIGHT)
    parser.add_argument("--boundary-loss-weight", type=float, default=conf.BOUNDARY_LOSS_WEIGHT)
    parser.add_argument("--boundary-radius", type=int, default=conf.BOUNDARY_RADIUS)
    parser.add_argument("--edge-threshold", type=float, default=conf.EDGE_THRESHOLD)
    parser.add_argument("--gate-focal-gamma", type=float, default=conf.GATE_FOCAL_GAMMA)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--skip-test-visualizations", action="store_true")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(fold < 0 or fold >= conf.NUM_FOLDS for fold in args.folds):
        raise ValueError(f"Folds must be in [0, {conf.NUM_FOLDS - 1}].")
    if "all" in args.experiments:
        experiments = list(EXPERIMENTS)
    elif "core" in args.experiments:
        experiments = ["R0", "R1", "R2"]
    else:
        experiments = args.experiments
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.runs_dir / "experiment_config.json",
        {
            "experiments": experiments,
            "folds": args.folds,
            "runs_dir": str(args.runs_dir),
            "index_dir": str(args.index_dir),
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "target_weight": args.target_weight,
            "dice_weight": args.dice_weight,
            "light_unet_base_channels": args.light_unet_base_channels,
            "reliability_base_channels": args.reliability_base_channels,
            "gate_loss_weight": args.gate_loss_weight,
            "boundary_loss_weight": args.boundary_loss_weight,
            "boundary_radius": args.boundary_radius,
            "edge_threshold": args.edge_threshold,
            "gate_focal_gamma": args.gate_focal_gamma,
            "experiment_specs": EXPERIMENT_SPECS,
        },
    )
    for experiment in experiments:
        run_experiment(experiment, args)
    summarize_paired_comparisons(args, experiments)


if __name__ == "__main__":
    main()
