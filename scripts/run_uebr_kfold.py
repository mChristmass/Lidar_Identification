import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.uebr.config_uebr as conf
from datasets.dataset_reliability_gate import ReliabilityGateDataset
from engine.stage1.loss_stage1 import SegmentationLoss
from models.model_stage1 import UNet
from models.model_uebr import UncertaintyEdgeBoundaryRefinementNet
from scripts.kfold_utils import print_summary, save_json, set_seed, summarize_metrics, tee_stdout
from scripts.reliability_gate_reporting import estimate_conv_flops, measure_latency
from scripts.run_reliability_gate_kfold import balanced_boundary_loss, evaluate_thresholds, load_fold_indices
from scripts.train_stage1_only_strong import select_best_threshold
from scripts.uebr_reporting import evaluate_and_report_uebr


DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
EXPERIMENTS = ("U0", "U1", "U2", "U3", "U4")
EXPERIMENT_SPECS = {
    "U0": ("C1L baseline", None, False, False),
    "U1": ("C1L with boundary auxiliary supervision", "boundary_only", False, True),
    "U2": ("Full-image edge-guided residual correction", "full_residual", True, True),
    "U3": ("Uncertainty-guided edge boundary refinement", "uncertainty_residual", True, True),
    "U4": ("U3 without explicit edge refinement features", "uncertainty_residual", False, True),
}


class C1LBaselineAdapter(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        self.model = UNet(in_channels=2, num_classes=2, base_channels=base_channels)

    def forward(self, x):
        return self.model(torch.cat([x[:, 0:1], x[:, 3:4]], dim=1))


def spec_dict(experiment):
    description, mode, explicit_edge, boundary = EXPERIMENT_SPECS[experiment]
    return {
        "description": description,
        "refinement_mode": mode,
        "use_explicit_edge": explicit_edge,
        "boundary_supervision": boundary,
    }


def make_dataset(indices, args):
    return ReliabilityGateDataset(
        conf.RAW_ITEMS,
        conf.LABEL_DIR,
        indices,
        boundary_radius=args.boundary_radius,
        edge_threshold=conf.EDGE_THRESHOLD,
    )


def make_model(experiment, args):
    spec = spec_dict(experiment)
    if experiment == "U0":
        return C1LBaselineAdapter(args.light_unet_base_channels)
    return UncertaintyEdgeBoundaryRefinementNet(
        base_channels=args.light_unet_base_channels,
        edge_feature_channels=args.edge_feature_channels,
        refinement_channels=args.refinement_channels,
        refinement_mode=spec["refinement_mode"],
        use_explicit_edge=spec["use_explicit_edge"],
        refinement_scale=args.refinement_scale,
    )


def train_one_fold(fold, experiment, args):
    seed = conf.FOLD_SEEDS[fold]
    set_seed(seed)
    spec = spec_dict(experiment)
    splits = load_fold_indices(args.index_dir, fold)
    out_dir = Path(args.runs_dir) / f"fold_{fold}" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    train_set, val_set, test_set = [make_dataset(indices, args) for indices in splits]
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    with tee_stdout(out_dir / "train.log"):
        print(f"\n===== UEBR {experiment}, fold {fold}, seed {seed} =====")
        print(spec["description"])
        print(f"Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}")
        print(f"Mode: {spec['refinement_mode']}  Explicit edge: {spec['use_explicit_edge']}")
        model = make_model(experiment, args).to(conf.DEVICE)
        parameter_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {parameter_count:,}")
        criterion = SegmentationLoss(
            target_weight=args.target_weight,
            ce_weight=conf.CE_WEIGHT,
            dice_weight=args.dice_weight,
        ).to(conf.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_iou, best_threshold, best_metrics, best_epoch = -1.0, None, None, None
        stale = 0
        best_path = out_dir / "best_model.pth"

        for epoch in range(args.epochs):
            model.train()
            totals = dict(loss=0.0, final=0.0, base=0.0, boundary=0.0, delta=0.0)
            for x, labels, _, _, boundary_target in train_loader:
                x = x.to(conf.DEVICE)
                labels = labels.to(conf.DEVICE)
                boundary_target = boundary_target.to(conf.DEVICE)
                optimizer.zero_grad()
                if experiment == "U0":
                    final_loss = criterion(model(x), labels)
                    base_loss = boundary_loss = delta_loss = final_loss.new_zeros(())
                    loss = final_loss
                else:
                    output = model(x, return_aux=True)
                    final_loss = criterion(output["logits"], labels)
                    base_loss = criterion(output["base_logits"], labels)
                    boundary_loss = balanced_boundary_loss(output["boundary_logits"], boundary_target)
                    delta_loss = output["delta_logits"].abs().mean()
                    loss = final_loss + args.boundary_loss_weight * boundary_loss
                    if experiment in {"U2", "U3", "U4"}:
                        loss = loss + args.base_loss_weight * base_loss
                        loss = loss + args.delta_l1_weight * delta_loss
                loss.backward()
                optimizer.step()
                for key, value in (
                    ("loss", loss), ("final", final_loss), ("base", base_loss),
                    ("boundary", boundary_loss), ("delta", delta_loss),
                ):
                    totals[key] += value.item()

            val_by_threshold = evaluate_thresholds(model, val_loader, conf.DEVICE, args.thresholds)
            threshold, metrics = select_best_threshold(val_by_threshold, key="iou")
            n = max(len(train_loader), 1)
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] Loss {totals['loss']/n:.4f} "
                f"Final {totals['final']/n:.4f} Base {totals['base']/n:.4f} "
                f"Boundary {totals['boundary']/n:.4f} Delta {totals['delta']/n:.4f} "
                f"Thr {threshold:g} Val IoU {metrics['iou']:.4f}"
            )
            if metrics["iou"] > best_iou:
                best_iou, best_threshold, best_metrics = metrics["iou"], threshold, metrics
                best_epoch, stale = epoch + 1, 0
                torch.save(model.state_dict(), best_path)
                save_json(out_dir / "val_threshold_metrics.json", val_by_threshold)
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    break

        model.load_state_dict(torch.load(best_path, map_location=conf.DEVICE))
        test_metrics = evaluate_thresholds(model, test_loader, conf.DEVICE, [best_threshold])[best_threshold]
        detailed = evaluate_and_report_uebr(
            model, test_set, conf.RAW_ITEMS, conf.DEVICE, best_threshold, out_dir,
            args.batch_size, not args.skip_test_visualizations, experiment == "U0",
        )
        sample = test_set[0][0].unsqueeze(0).to(conf.DEVICE)
        flops = estimate_conv_flops(model, sample)
        latency = measure_latency(model, sample, conf.DEVICE, args.latency_warmup, args.latency_repeats)
        refine_experiments = {"U2", "U3", "U4"}
        result = {
            "fold": fold,
            "seed": seed,
            "experiment": experiment,
            **spec,
            "parameters": parameter_count,
            "estimated_flops": flops,
            "latency_ms_batch1": latency,
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "test_samples": len(test_set),
            "base_loss_weight": args.base_loss_weight if experiment in refine_experiments else 0.0,
            "boundary_loss_weight": args.boundary_loss_weight if experiment != "U0" else 0.0,
            "delta_l1_weight": args.delta_l1_weight if experiment in refine_experiments else 0.0,
            "refinement_scale": args.refinement_scale,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "best_val_iou": best_iou,
            "best_val_metrics": best_metrics,
            "detailed_test_summary": detailed,
            "boundary_f1": detailed["boundary_f1"],
            "boundary_iou": detailed["boundary_iou"],
            "base_to_final_image_iou": detailed["mean_image_iou_improvement"],
            "refinement_win_rate": detailed["improvement_win_rate"],
            **test_metrics,
        }
        save_json(out_dir / "metrics.json", result)
        print(
            f"Test IoU {result['iou']:.4f} Dice {result['dice']:.4f} "
            f"Boundary F1 {result['boundary_f1']:.4f} "
            f"Base-to-final {result['base_to_final_image_iou']:+.4f}"
        )
        return result


def run_experiment(experiment, args):
    results = [train_one_fold(fold, experiment, args) for fold in args.folds]
    keys = [
        "iou", "precision", "recall", "dice", "boundary_f1", "boundary_iou",
        "base_to_final_image_iou", "refinement_win_rate", "latency_ms_batch1",
    ]
    summary = summarize_metrics(results, keys)
    summary.update(
        experiment=experiment,
        description=spec_dict(experiment)["description"],
        parameters=results[0]["parameters"],
        estimated_flops=results[0]["estimated_flops"],
    )
    save_json(Path(args.runs_dir) / f"summary_{experiment}.json", summary)
    print_summary(f"UEBR {experiment}", summary, keys)


def load_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return {int(row["label_index"]): row for row in csv.DictReader(handle)}


def summarize_paired(args, experiments):
    if "U3" not in experiments:
        return
    comparisons = {}
    for baseline in experiments:
        if baseline == "U3":
            continue
        differences = []
        for fold in args.folds:
            primary_path = Path(args.runs_dir) / f"fold_{fold}/U3/test_per_image_metrics.csv"
            baseline_path = Path(args.runs_dir) / f"fold_{fold}/{baseline}/test_per_image_metrics.csv"
            if not primary_path.exists() or not baseline_path.exists():
                continue
            primary, other = load_rows(primary_path), load_rows(baseline_path)
            for index in set(primary) & set(other):
                a, b = primary[index]["iou"], other[index]["iou"]
                if a not in ("", None) and b not in ("", None):
                    differences.append(float(a) - float(b))
        if differences:
            values = np.asarray(differences)
            comparisons[f"U3_vs_{baseline}"] = {
                "foreground_images": len(values),
                "mean_image_iou_difference": float(values.mean()),
                "median_image_iou_difference": float(np.median(values)),
                "improved_images": int((values > 1e-9).sum()),
                "tied_images": int((np.abs(values) <= 1e-9).sum()),
                "worse_images": int((values < -1e-9).sum()),
                "win_rate": float((values > 1e-9).mean()),
            }
    save_json(Path(args.runs_dir) / "paired_comparisons.json", comparisons)


def parse_args():
    parser = argparse.ArgumentParser(description="Run UEBR U0-U4 experiments.")
    parser.add_argument("--experiments", nargs="+", choices=[*EXPERIMENTS, "all"], default=["all"])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1])
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
    parser.add_argument("--edge-feature-channels", type=int, default=conf.EDGE_FEATURE_CHANNELS)
    parser.add_argument("--refinement-channels", type=int, default=conf.REFINEMENT_CHANNELS)
    parser.add_argument("--base-loss-weight", type=float, default=conf.BASE_LOSS_WEIGHT)
    parser.add_argument("--boundary-loss-weight", type=float, default=conf.BOUNDARY_LOSS_WEIGHT)
    parser.add_argument("--delta-l1-weight", type=float, default=conf.DELTA_L1_WEIGHT)
    parser.add_argument("--refinement-scale", type=float, default=conf.REFINEMENT_SCALE)
    parser.add_argument("--boundary-radius", type=int, default=conf.BOUNDARY_RADIUS)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--skip-test-visualizations", action="store_true")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(fold < 0 or fold >= conf.NUM_FOLDS for fold in args.folds):
        raise ValueError(f"Folds must be in [0, {conf.NUM_FOLDS - 1}].")
    experiments = list(EXPERIMENTS) if "all" in args.experiments else args.experiments
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
            "edge_feature_channels": args.edge_feature_channels,
            "refinement_channels": args.refinement_channels,
            "base_loss_weight": args.base_loss_weight,
            "boundary_loss_weight": args.boundary_loss_weight,
            "delta_l1_weight": args.delta_l1_weight,
            "refinement_scale": args.refinement_scale,
            "boundary_radius": args.boundary_radius,
            "experiment_specs": {name: spec_dict(name) for name in experiments},
        },
    )
    for experiment in experiments:
        run_experiment(experiment, args)
    summarize_paired(args, experiments)


if __name__ == "__main__":
    main()
