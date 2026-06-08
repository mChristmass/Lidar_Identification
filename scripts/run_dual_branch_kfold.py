import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.dual_branch.config_dual_branch as conf
from configs import kfold_config as kfold
from datasets.dataset_dual_branch import DualBranchDataset
from engine.stage1.loss_stage1 import SegmentationLoss
from models.model_dual_branch import DualBranchGatedUNet
from scripts.kfold_utils import print_summary, save_json, set_seed, summarize_metrics, tee_stdout
from scripts.train_stage1_only_strong import evaluate_thresholds, select_best_threshold


DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
EXPERIMENT_MODES = {
    "D1": "no_gate",
    "D2": "bottleneck_gate",
    "D3": "multiscale_gate",
}


def make_dataset(indices, edge_erode_iterations):
    return DualBranchDataset(
        raw_items=kfold.RAW_ITEMS,
        mask_dir=kfold.LABEL_DIR,
        indices=indices,
        edge_erode_iterations=edge_erode_iterations,
    )


def build_boundary_target(labels):
    labels = labels.float().unsqueeze(1)
    dilated = F.max_pool2d(labels, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-labels, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0).squeeze(1)


def gate_means(gates):
    return [float(gate.detach().mean().cpu()) for gate in gates]


def train_one_fold(seed, experiment, args):
    set_seed(seed)
    train_indices, val_indices, test_indices = kfold.load_split_indices(seed)
    out_dir = Path(args.runs_dir) / kfold.fold_name(seed) / "dual_branch" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = make_dataset(train_indices, args.edge_erode_iterations)
    val_set = make_dataset(val_indices, args.edge_erode_iterations)
    test_set = make_dataset(test_indices, args.edge_erode_iterations)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    fusion_mode = EXPERIMENT_MODES[experiment]
    with tee_stdout(out_dir / "train.log"):
        print(f"\n===== Dual-branch {experiment}, seed {seed} =====")
        print(f"Fusion mode: {fusion_mode}")
        print(f"Input items: {train_set.input_items}")
        print(f"Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}")
        print(f"Boundary loss weight: {args.boundary_loss_weight}")
        print(f"Edge input dropout: {args.edge_dropout}")

        model = DualBranchGatedUNet(
            num_classes=2,
            intensity_base_channels=args.intensity_base_channels,
            edge_base_channels=args.edge_base_channels,
            fusion_mode=fusion_mode,
        ).to(conf.DEVICE)
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
        best_model_path = out_dir / "best_model.pth"

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            epoch_gate_means = []
            for x, y in train_loader:
                x = x.to(conf.DEVICE)
                y = y.to(conf.DEVICE)
                if args.edge_dropout > 0:
                    drop = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < args.edge_dropout
                    x = x.clone()
                    x[:, 1:2] = x[:, 1:2] * (~drop)

                optimizer.zero_grad()
                output = model(x, return_aux=True)
                loss = criterion(output["logits"], y)
                if args.boundary_loss_weight > 0:
                    boundary_target = build_boundary_target(y)
                    boundary_loss = F.binary_cross_entropy_with_logits(
                        output["boundary_logits"],
                        boundary_target,
                    )
                    loss = loss + args.boundary_loss_weight * boundary_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                epoch_gate_means.append(gate_means(output["gates"]))

            val_by_threshold = evaluate_thresholds(model, val_loader, conf.DEVICE, args.thresholds)
            threshold, val_metrics = select_best_threshold(val_by_threshold, key="iou")
            mean_gates = torch.tensor(epoch_gate_means).mean(dim=0).tolist()
            print(
                f"Epoch [{epoch + 1}/{args.epochs}]  "
                f"Loss: {total_loss / max(1, len(train_loader)):.4f}  "
                f"Thr: {threshold:g}  Val IoU: {val_metrics['iou']:.4f}  "
                f"Dice: {val_metrics['dice']:.4f}  "
                f"Gates: {[round(value, 3) for value in mean_gates]}"
            )
            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                best_threshold = threshold
                best_val_metrics = val_metrics
                torch.save(model.state_dict(), best_model_path)
                save_json(out_dir / "val_threshold_metrics.json", val_by_threshold)

        model.load_state_dict(torch.load(best_model_path, map_location=conf.DEVICE))
        test_metrics = evaluate_thresholds(model, test_loader, conf.DEVICE, [best_threshold])[best_threshold]
        result = {
            "seed": int(seed),
            "experiment": experiment,
            "fusion_mode": fusion_mode,
            "input_items": train_set.input_items,
            "parameters": int(parameter_count),
            "intensity_base_channels": int(args.intensity_base_channels),
            "edge_base_channels": int(args.edge_base_channels),
            "edge_dropout": float(args.edge_dropout),
            "boundary_loss_weight": float(args.boundary_loss_weight),
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
    output_path = Path(args.runs_dir) / f"summary_dual_branch_{experiment}.json"
    save_json(output_path, summary)
    print_summary(f"Dual Branch {experiment}", summary, metric_keys)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run lightweight dual-branch local-edge ablations.")
    parser.add_argument("--experiment", choices=["D1", "D2", "D3", "all"], default="D3")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=kfold.DATA_ROOT / f"runs/{conf.RUNS_DIR_NAME}",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=kfold.KFOLD_SEEDS)
    parser.add_argument("--epochs", type=int, default=conf.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=conf.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=conf.LR)
    parser.add_argument("--target-weight", type=float, default=conf.TARGET_WEIGHT)
    parser.add_argument("--dice-weight", type=float, default=conf.DICE_WEIGHT)
    parser.add_argument("--boundary-loss-weight", type=float, default=conf.BOUNDARY_LOSS_WEIGHT)
    parser.add_argument("--intensity-base-channels", type=int, default=conf.INTENSITY_BASE_CHANNELS)
    parser.add_argument("--edge-base-channels", type=int, default=conf.EDGE_BASE_CHANNELS)
    parser.add_argument("--edge-dropout", type=float, default=conf.EDGE_DROPOUT)
    parser.add_argument("--edge-erode-iterations", type=int, default=conf.EDGE_ERODE_ITERATIONS)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    return parser.parse_args()


def main():
    args = parse_args()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    experiments = list(EXPERIMENT_MODES) if args.experiment == "all" else [args.experiment]
    for experiment in experiments:
        run_experiment(experiment, args)


if __name__ == "__main__":
    main()
