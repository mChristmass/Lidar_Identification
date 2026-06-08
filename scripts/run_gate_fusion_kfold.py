import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.gate_fusion.config_gate_fusion as conf
from configs import kfold_config as kfold
from datasets.dataset_gate_fusion import GateFusionDataset
from datasets.dataset_stage2_local_edge import Stage2LocalEdgeDataset
from models.model_gate_fusion import PixelGateNet
from models.model_stage1 import UNet
from scripts.kfold_utils import print_summary, save_json, set_seed, summarize_metrics, tee_stdout


DEFAULT_ALPHAS = [round(value, 1) for value in np.linspace(0.0, 1.0, 11)]
DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def fold_dir(runs_dir, seed):
    return Path(runs_dir) / kfold.fold_name(seed)


def binary_counts(prob, label, threshold):
    pred = prob > threshold
    label = label.bool()
    return {
        "tp": int((pred & label).sum().item()),
        "fp": int((pred & ~label).sum().item()),
        "fn": int((~pred & label).sum().item()),
        "pred_area": int(pred.sum().item()),
        "label_area": int(label.sum().item()),
    }


def metrics_from_counts(row):
    tp, fp, fn = row["tp"], row["fp"], row["fn"]
    return {
        "iou": float(tp / (tp + fp + fn + 1e-6)),
        "precision": float(tp / (tp + fp + 1e-6)),
        "recall": float(tp / (tp + fn + 1e-6)),
        "dice": float(2 * tp / (2 * tp + fp + fn + 1e-6)),
        "coverage": float(tp / (tp + fn + 1e-6)),
        "pred_area": float(row["pred_area"]),
        "label_area": float(row["label_area"]),
    }


def add_counts(total, row):
    for key in total:
        total[key] += row[key]


def predict_c1(seed, source_c1_dir, batch_size):
    all_indices = kfold.load_label_indices()
    dataset = Stage2LocalEdgeDataset(
        raw_items=kfold.RAW_ITEMS,
        mask_dir=kfold.LABEL_DIR,
        indices=all_indices,
        experiment="C1",
        prior_dir=None,
        edge_erode_iterations=2,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = UNet(in_channels=2, num_classes=2).to(conf.DEVICE)
    model.load_state_dict(torch.load(source_c1_dir / "best_model.pth", map_location=conf.DEVICE))
    model.eval()

    sample_count = len(np.load(kfold.RAW_ITEMS["intensity"], mmap_mode="r"))
    probs = np.zeros((sample_count, 128, 128), dtype=np.float32)
    with torch.no_grad():
        offset = 0
        for x, _, _ in loader:
            x = x.to(conf.DEVICE)
            batch_prob = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            probs[offset:offset + len(batch_prob)] = batch_prob
            offset += len(batch_prob)
    return probs


def prepare_run(source_stage1_dir, source_c1_dir, runs_dir, seeds, batch_size):
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        source_stage1_fold = fold_dir(source_stage1_dir, seed) / "stage1_strong_priors"
        source_c1_fold = fold_dir(source_c1_dir, seed) / "stage2_local_edge" / "C1"
        target_dir = fold_dir(runs_dir, seed) / "gate_fusion" / "expert_predictions"
        target_dir.mkdir(parents=True, exist_ok=True)

        for name, target_name in (
            ("prob.npy", "stage1_prob.npy"),
            ("roi_mask.npy", "roi_mask.npy"),
        ):
            source = source_stage1_fold / name
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, target_dir / target_name)

        c1_prob = predict_c1(seed, source_c1_fold, batch_size)
        np.save(target_dir / "c1_prob.npy", c1_prob)
        shutil.copy2(source_c1_fold / "metrics.json", target_dir / "c1_metrics.json")
        shutil.copy2(source_stage1_fold / "meta.json", target_dir / "stage1_meta.json")
        print(f"Prepared frozen expert predictions for seed {seed}: {target_dir}")

    save_json(
        runs_dir / "gate_fusion_experiment.json",
        {
            "source_stage1_dir": str(source_stage1_dir),
            "source_c1_dir": str(source_c1_dir),
            "runs_dir": str(runs_dir),
            "seeds": [int(seed) for seed in seeds],
            "experts": ["intensity-only Stage1", "intensity + local depth edge C1"],
            "fusion": ["fixed validation-selected alpha", "pixel-wise Gate-Net"],
        },
    )


def evaluate_fixed(dataset, alphas, thresholds):
    loader = DataLoader(dataset, batch_size=conf.BATCH_SIZE, shuffle=False)
    totals = {
        (float(alpha), float(threshold)): {"tp": 0, "fp": 0, "fn": 0, "pred_area": 0, "label_area": 0}
        for alpha in alphas
        for threshold in thresholds
    }
    for features, labels, _ in loader:
        p1 = features[:, 0]
        p2 = features[:, 2]
        for alpha in alphas:
            fused = (1.0 - alpha) * p1 + alpha * p2
            for threshold in thresholds:
                add_counts(totals[(float(alpha), float(threshold))], binary_counts(fused, labels, threshold))
    return {key: metrics_from_counts(row) for key, row in totals.items()}


def select_best_fixed(metrics):
    return max(
        metrics.items(),
        key=lambda item: (item[1]["iou"], item[1]["dice"], item[1]["recall"]),
    )


def run_fixed_fold(seed, args):
    _, val_indices, test_indices = kfold.load_split_indices(seed)
    expert_dir = fold_dir(args.runs_dir, seed) / "gate_fusion" / "expert_predictions"
    out_dir = fold_dir(args.runs_dir, seed) / "gate_fusion" / "fixed"
    out_dir.mkdir(parents=True, exist_ok=True)
    val_set = GateFusionDataset(expert_dir, kfold.LABEL_DIR, val_indices)
    test_set = GateFusionDataset(expert_dir, kfold.LABEL_DIR, test_indices)

    val_metrics = evaluate_fixed(val_set, args.alphas, args.thresholds)
    (alpha, threshold), best_val = select_best_fixed(val_metrics)
    test_metrics = evaluate_fixed(test_set, [alpha], [threshold])[(alpha, threshold)]
    result = {
        "seed": int(seed),
        "alpha_c1": float(alpha),
        "alpha_stage1": float(1.0 - alpha),
        "threshold": float(threshold),
        "best_val_metrics": best_val,
        **test_metrics,
    }
    save_json(out_dir / "metrics.json", result)
    print(
        f"Fixed seed {seed}: alpha_c1={alpha:.1f}, threshold={threshold:g}, "
        f"test IoU={result['iou']:.4f}"
    )
    return result


def split_gate_indices(val_indices, seed, train_ratio):
    rng = np.random.default_rng(seed)
    shuffled = np.array(val_indices, dtype=np.int64)
    rng.shuffle(shuffled)
    split = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_ratio))))
    return shuffled[:split].tolist(), shuffled[split:].tolist()


def gate_forward(model, features, training=False, prior_dropout=0.0):
    gate_features = features.clone()
    if training and prior_dropout > 0:
        drop = (torch.rand(features.shape[0], 1, 1, device=features.device) < prior_dropout).float()
        gate_features[:, 0] *= 1.0 - drop
        gate_features[:, 1] *= 1.0 - drop
        gate_features[:, 4] *= 1.0 - drop
    gate = model(gate_features)
    p1 = features[:, 0]
    p2 = features[:, 2]
    fused = (1.0 - gate) * p1 + gate * p2
    return fused.clamp(1e-6, 1.0 - 1e-6), gate


def gate_loss(fused, gate, labels, reg_weight):
    bce = F.binary_cross_entropy(fused, labels)
    intersection = (fused * labels).sum(dim=(1, 2))
    dice = (2.0 * intersection + 1e-6) / (
        fused.sum(dim=(1, 2)) + labels.sum(dim=(1, 2)) + 1e-6
    )
    preserve_c1 = ((1.0 - gate) ** 2).mean()
    return bce + (1.0 - dice.mean()) + reg_weight * preserve_c1


def evaluate_gate(model, dataset, thresholds):
    loader = DataLoader(dataset, batch_size=conf.BATCH_SIZE, shuffle=False)
    totals = {
        float(threshold): {"tp": 0, "fp": 0, "fn": 0, "pred_area": 0, "label_area": 0}
        for threshold in thresholds
    }
    gate_sum = 0.0
    gate_pixels = 0
    model.eval()
    with torch.no_grad():
        for features, labels, _ in loader:
            features = features.to(conf.DEVICE)
            labels = labels.to(conf.DEVICE)
            fused, gate = gate_forward(model, features)
            gate_sum += float(gate.sum().item())
            gate_pixels += gate.numel()
            for threshold in thresholds:
                add_counts(totals[float(threshold)], binary_counts(fused, labels, threshold))
    metrics = {threshold: metrics_from_counts(row) for threshold, row in totals.items()}
    return metrics, gate_sum / max(gate_pixels, 1)


def run_gate_fold(seed, args):
    set_seed(seed)
    _, val_indices, test_indices = kfold.load_split_indices(seed)
    gate_train_indices, gate_val_indices = split_gate_indices(
        val_indices,
        seed,
        args.gate_train_ratio,
    )
    expert_dir = fold_dir(args.runs_dir, seed) / "gate_fusion" / "expert_predictions"
    out_dir = fold_dir(args.runs_dir, seed) / "gate_fusion" / "pixel_gate"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = GateFusionDataset(expert_dir, kfold.LABEL_DIR, gate_train_indices)
    val_set = GateFusionDataset(expert_dir, kfold.LABEL_DIR, gate_val_indices)
    test_set = GateFusionDataset(expert_dir, kfold.LABEL_DIR, test_indices)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    with tee_stdout(out_dir / "train.log"):
        print(f"\n===== Pixel Gate-Net seed {seed} =====")
        print(f"Gate train/val/test: {len(train_set)}/{len(val_set)}/{len(test_set)}")
        print(f"Prior dropout: {args.prior_dropout}")
        model = PixelGateNet(
            in_channels=5,
            hidden_channels=args.hidden_channels,
            initial_c1_weight=args.initial_c1_weight,
        ).to(conf.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_val_iou = -1.0
        best_threshold = None
        best_val_metrics = None
        best_model_path = out_dir / "best_model.pth"

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for features, labels, _ in train_loader:
                features = features.to(conf.DEVICE)
                labels = labels.to(conf.DEVICE)
                fused, gate = gate_forward(
                    model,
                    features,
                    training=True,
                    prior_dropout=args.prior_dropout,
                )
                loss = gate_loss(fused, gate, labels, args.gate_reg_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_by_threshold, mean_gate = evaluate_gate(model, val_set, args.thresholds)
            threshold, val_metrics = max(
                val_by_threshold.items(),
                key=lambda item: (item[1]["iou"], item[1]["dice"], item[1]["recall"]),
            )
            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                best_threshold = threshold
                best_val_metrics = val_metrics
                torch.save(model.state_dict(), best_model_path)
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] Loss: "
                f"{total_loss / max(1, len(train_loader)):.4f}  "
                f"Val IoU: {val_metrics['iou']:.4f}  Thr: {threshold:g}  "
                f"Mean C1 gate: {mean_gate:.3f}"
            )

        model.load_state_dict(torch.load(best_model_path, map_location=conf.DEVICE))
        test_by_threshold, mean_test_gate = evaluate_gate(model, test_set, [best_threshold])
        test_metrics = test_by_threshold[best_threshold]
        result = {
            "seed": int(seed),
            "gate_train_samples": len(train_set),
            "gate_val_samples": len(val_set),
            "best_threshold": float(best_threshold),
            "best_val_iou": float(best_val_iou),
            "best_val_metrics": best_val_metrics,
            "mean_test_c1_gate": float(mean_test_gate),
            "prior_dropout": float(args.prior_dropout),
            "gate_reg_weight": float(args.gate_reg_weight),
            **test_metrics,
        }
        save_json(out_dir / "metrics.json", result)
        print(
            f"Test IoU: {result['iou']:.4f}  Dice: {result['dice']:.4f}  "
            f"Mean C1 gate: {mean_test_gate:.3f}"
        )
        return result


def summarize(results, name, runs_dir):
    keys = ["iou", "precision", "recall", "dice", "coverage", "pred_area"]
    summary = summarize_metrics(results, keys)
    save_json(Path(runs_dir) / f"summary_gate_fusion_{name}.json", summary)
    print_summary(f"Gate Fusion {name}", summary, keys)


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed and pixel-wise fusion of Stage1 and C1 experts.")
    parser.add_argument("--stage", choices=["prepare", "fixed", "gate", "all"], default="all")
    parser.add_argument("--source-stage1-dir", type=Path, default=kfold.DATA_ROOT / "runs/run8")
    parser.add_argument("--source-c1-dir", type=Path, default=kfold.DATA_ROOT / "runs/run10")
    parser.add_argument("--runs-dir", type=Path, default=kfold.DATA_ROOT / "runs/run12")
    parser.add_argument("--seeds", type=int, nargs="+", default=kfold.KFOLD_SEEDS)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--batch-size", type=int, default=conf.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=conf.EPOCHS)
    parser.add_argument("--lr", type=float, default=conf.LR)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--prior-dropout", type=float, default=conf.PRIOR_DROPOUT)
    parser.add_argument("--gate-train-ratio", type=float, default=conf.GATE_TRAIN_RATIO)
    parser.add_argument("--initial-c1-weight", type=float, default=conf.GATE_INIT_C1_WEIGHT)
    parser.add_argument("--gate-reg-weight", type=float, default=conf.GATE_REG_WEIGHT)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage in {"prepare", "all"}:
        prepare_run(
            args.source_stage1_dir,
            args.source_c1_dir,
            args.runs_dir,
            args.seeds,
            args.batch_size,
        )
    if args.stage in {"fixed", "all"}:
        summarize(
            [run_fixed_fold(seed, args) for seed in args.seeds],
            "fixed",
            args.runs_dir,
        )
    if args.stage in {"gate", "all"}:
        summarize(
            [run_gate_fold(seed, args) for seed in args.seeds],
            "pixel_gate",
            args.runs_dir,
        )


if __name__ == "__main__":
    main()
