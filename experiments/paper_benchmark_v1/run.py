from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

from experiments.t8_transfer_statistics.model_transfer import T8TransferNet
from .corruptions import CORRUPTIONS, SEVERITIES
from .data import NYUv2PaperDataset, PrivatePaperDataset
from .model import BACKBONES, CMXPaper, load_imagenet


MODELS = ("T8", *BACKBONES)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def build_datasets(args):
    if args.stage == "final" and not args.confirm_final_protocol:
        raise RuntimeError(
            "Final evaluation is locked. Freeze the selected model, corruption suite, learning recipe, "
            "and epoch count first, then pass --confirm-final-protocol."
        )
    if args.dataset == "private":
        split = Path(args.private_split_dir)
        train = PrivatePaperDataset(args.private_data_dir, split / "train_indices.npy", training=True, seed=args.seed)
        dev = PrivatePaperDataset(args.private_data_dir, split / "dev_indices.npy", seed=args.seed)
        if args.stage == "tune":
            return train, dev, None, 2
        final_train = ConcatDataset([
            PrivatePaperDataset(args.private_data_dir, split / "train_indices.npy", training=True, seed=args.seed),
            PrivatePaperDataset(args.private_data_dir, split / "dev_indices.npy", training=True, seed=args.seed),
        ])
        test = PrivatePaperDataset(args.private_data_dir, split / "final_test_indices.npy", seed=args.seed)
        return final_train, None, test, 2

    root = Path(args.nyuv2_dir)
    manifests = root / "manifests"
    common = dict(
        root=root, calibration=args.calibration, corruption=args.corruption,
        severity=args.severity, image_size=(args.height, args.width), seed=args.seed,
    )
    if args.stage == "tune":
        train = NYUv2PaperDataset(manifest=manifests / "train.jsonl", split="train", training=True, **common)
        dev = NYUv2PaperDataset(manifest=manifests / "dev.jsonl", split="dev", **common)
        return train, dev, None, 40
    train = NYUv2PaperDataset(manifest=manifests / "train_full.jsonl", split="train_full", training=True, **common)
    test = NYUv2PaperDataset(manifest=manifests / "test.jsonl", split="test", **common)
    return train, None, test, 40


def build_model(args, num_classes: int) -> nn.Module:
    if args.model == "T8":
        return T8TransferNet("T0_R1_REPRO", base_channels=args.t8_channels, num_classes=num_classes)
    model = CMXPaper(args.model, num_classes, image_size=max(args.height, args.width), decoder_dim=args.decoder_dim)
    if args.pretrained:
        report = load_imagenet(model, args.pretrained)
        print("PRETRAINED " + json.dumps(report, ensure_ascii=False), flush=True)
    elif not args.allow_random_init:
        raise ValueError("CMX requires --pretrained, or pass --allow-random-init deliberately for a smoke test")
    return model


def loader(dataset, args, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
        pin_memory=args.pin_memory, persistent_workers=args.workers > 0,
    )


@torch.no_grad()
def evaluate(model, data_loader, device, classes: int) -> dict:
    model.eval()
    matrix = torch.zeros(classes, classes, dtype=torch.int64, device=device)
    boundary_pred_matched = boundary_true_matched = 0
    boundary_pred_total = boundary_true_total = 0
    background_frames = background_false_positive = 0
    for batch in data_loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        target = batch["labels"].to(device, non_blocking=True)
        prediction = model(inputs)["logits"].argmax(1)
        selected = target != 255
        encoded = target[selected] * classes + prediction[selected]
        matrix += torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)
        if classes == 2:
            predicted_fg = prediction == 1
            true_fg = target == 1
            pred_eroded = 1 - F.max_pool2d((~predicted_fg).float()[:, None], 3, 1, 1)[:, 0]
            true_eroded = 1 - F.max_pool2d((~true_fg).float()[:, None], 3, 1, 1)[:, 0]
            pred_boundary = predicted_fg & (pred_eroded < 0.5)
            true_boundary = true_fg & (true_eroded < 0.5)
            true_tolerance = F.max_pool2d(true_boundary.float()[:, None], 3, 1, 1)[:, 0] > 0
            pred_tolerance = F.max_pool2d(pred_boundary.float()[:, None], 3, 1, 1)[:, 0] > 0
            # Symmetric tolerance matching, accumulated as precision/recall counts.
            boundary_pred_matched += int((pred_boundary & true_tolerance).sum())
            boundary_true_matched += int((true_boundary & pred_tolerance).sum())
            boundary_pred_total += int(pred_boundary.sum())
            boundary_true_total += int(true_boundary.sum())
            empty = ~true_fg.flatten(1).any(1)
            background_frames += int(empty.sum())
            background_false_positive += int((empty & predicted_fg.flatten(1).any(1)).sum())
    matrix = matrix.cpu().numpy().astype(np.float64)
    tp = np.diag(matrix)
    union = matrix.sum(0) + matrix.sum(1) - tp
    present = union > 0
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=present)
    accuracy = tp.sum() / max(matrix.sum(), 1)
    result = {
        "miou": float(np.nanmean(iou)), "pixel_accuracy": float(accuracy),
        "class_iou": [None if not np.isfinite(x) else float(x) for x in iou],
        "confusion_matrix": matrix.astype(np.int64).tolist(),
    }
    if classes == 2:
        fg_tp, fg_union = tp[1], union[1]
        precision = fg_tp / max(matrix[:, 1].sum(), 1)
        recall = fg_tp / max(matrix[1, :].sum(), 1)
        boundary_precision = boundary_pred_matched / max(boundary_pred_total, 1)
        boundary_recall = boundary_true_matched / max(boundary_true_total, 1)
        boundary_f1 = 2 * boundary_precision * boundary_recall / max(
            boundary_precision + boundary_recall, 1e-12
        )
        result.update({
            "foreground_iou": float(fg_tp / max(fg_union, 1)),
            "foreground_dice": float(2 * fg_tp / max(matrix[:, 1].sum() + matrix[1, :].sum(), 1)),
            "foreground_precision": float(precision),
            "foreground_recall": float(recall),
            "boundary_precision_1px": float(boundary_precision),
            "boundary_recall_1px": float(boundary_recall),
            "boundary_f1_1px": float(boundary_f1),
            "background_fp_frame_rate": float(background_false_positive / max(background_frames, 1)),
        })
    return result


def train(args) -> None:
    seed_all(args.seed)
    train_data, dev_data, test_data, classes = build_datasets(args)
    device = torch.device(args.device)
    model = build_model(args, classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    train_loader = loader(train_data, args, True)
    dev_loader = loader(dev_data, args, False) if dev_data is not None else None
    destination = Path(args.runs_dir) / args.dataset / args.stage / args.model / f"seed_{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, 1):
            inputs = batch["inputs"].to(device, non_blocking=True)
            target = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss = criterion(model(inputs)["logits"], target) / args.grad_accum
            scaler.scale(loss).backward()
            if batch_index % args.grad_accum == 0 or batch_index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * args.grad_accum
        factor = (1.0 - epoch / args.epochs) ** args.lr_power
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
        record = {"epoch": epoch, "loss": total_loss / max(1, len(train_loader))}
        if dev_loader is not None:
            record.update({f"dev_{key}": value for key, value in evaluate(model, dev_loader, device, classes).items()})
            score = record["dev_miou"]
        else:
            score = float(epoch)
        history.append(record)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, destination / "best.pth")
        save_json(destination / "history.json", {"epochs": history})
        elapsed = time.time() - started
        eta = elapsed / epoch * (args.epochs - epoch)
        print(
            f"EPOCH_SUMMARY dataset={args.dataset} stage={args.stage} model={args.model} "
            f"epoch={epoch}/{args.epochs} loss={record['loss']:.6f} "
            f"dev_miou={record.get('dev_miou', float('nan')):.6f} eta_seconds={eta:.0f}",
            flush=True,
        )

    checkpoint = torch.load(destination / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {
        "dataset": args.dataset, "stage": args.stage, "model": args.model,
        "seed": args.seed, "best_epoch": checkpoint["epoch"], "best_selection_score": best,
    }
    if test_data is not None:
        result["test"] = evaluate(model, loader(test_data, args, False), device, classes)
        print(f"FINAL_TEST_SUMMARY model={args.model} miou={result['test']['miou']:.6f}", flush=True)
    save_json(destination / "result.json", result)


def evaluate_checkpoint(args) -> None:
    if not args.checkpoint:
        raise ValueError("evaluate requires --checkpoint")
    if args.stage == "final" and not args.confirm_final_protocol:
        raise RuntimeError("Final evaluation remains locked; pass --confirm-final-protocol only after freezing the protocol")
    device = torch.device(args.device)
    classes = 2 if args.dataset == "private" else 40
    if args.model == "T8":
        model = T8TransferNet("T0_R1_REPRO", base_channels=args.t8_channels, num_classes=classes)
    else:
        model = CMXPaper(args.model, classes, image_size=max(args.height, args.width), decoder_dim=args.decoder_dim)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    if args.dataset == "private":
        name = "dev_indices.npy" if args.stage == "tune" else "final_test_indices.npy"
        dataset = PrivatePaperDataset(args.private_data_dir, Path(args.private_split_dir) / name, seed=args.seed)
        suite = {"natural": evaluate(model, loader(dataset, args, False), device, classes)}
    else:
        split = "dev" if args.stage == "tune" else "test"
        manifest = Path(args.nyuv2_dir) / "manifests" / f"{split}.jsonl"
        suite = {}
        conditions = [("clean", "medium")]
        conditions += [(name, severity) for name in CORRUPTIONS
                       if name not in {"clean", "mixture"} for severity in SEVERITIES]
        for corruption, severity in conditions:
            dataset = NYUv2PaperDataset(
                args.nyuv2_dir, manifest, args.calibration, corruption=corruption,
                severity=severity, split=split, image_size=(args.height, args.width), seed=args.seed,
            )
            metrics = evaluate(model, loader(dataset, args, False), device, classes)
            label = "clean" if corruption == "clean" else f"{corruption}:{severity}"
            suite[label] = metrics
            print(f"ROBUSTNESS_SUMMARY condition={label} miou={metrics['miou']:.6f}", flush=True)
        clean = suite["clean"]["miou"]
        for label, metrics in suite.items():
            metrics["miou_drop_from_clean"] = float(clean - metrics["miou"])
    output = Path(args.checkpoint).parent / f"robustness_{args.stage}.json"
    save_json(output, {"checkpoint": str(args.checkpoint), "suite": suite})
    print(f"ROBUSTNESS_COMPLETE output={output} conditions={len(suite)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Private/NYUv2 frozen paper benchmark runner")
    parser.add_argument("command", choices=("audit", "train", "evaluate"))
    parser.add_argument("--dataset", choices=("private", "nyuv2"), required=True)
    parser.add_argument("--stage", choices=("tune", "final"), default="tune")
    parser.add_argument("--confirm-final-protocol", action="store_true")
    parser.add_argument("--model", choices=MODELS, default="CMX_B2")
    parser.add_argument("--private-data-dir", default="data/new_data/merged")
    parser.add_argument("--private-split-dir", default="data/new_data/merged/paper_split_v1")
    parser.add_argument("--nyuv2-dir", default="data/public_semseg/nyuv2/processed")
    parser.add_argument("--calibration", default="data/new_data/merged/paper_split_v1/degradation_profile_v1/calibration.json")
    parser.add_argument("--corruption", default="clean")
    parser.add_argument("--severity", default="medium")
    parser.add_argument("--runs-dir", default="data/paper_benchmark_runs/v1")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0,
                        help="Safe default avoids the prior pin-memory worker crash; raise only after a smoke test.")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-power", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--t8-channels", type=int, default=38)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def audit(args) -> None:
    train_data, dev_data, test_data, classes = build_datasets(args)
    report = {
        "dataset": args.dataset, "stage": args.stage, "classes": classes,
        "train": len(train_data), "dev": None if dev_data is None else len(dev_data),
        "test": None if test_data is None else len(test_data),
        "sample_shape": list(train_data[0]["inputs"].shape),
        "sample_label_values": torch.unique(train_data[0]["labels"]).tolist(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "audit":
        audit(arguments)
    elif arguments.command == "train":
        train(arguments)
    else:
        evaluate_checkpoint(arguments)
