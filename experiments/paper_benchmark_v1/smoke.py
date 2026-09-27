from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.t8_transfer_statistics.model_transfer import T8TransferNet
from .data import PrivatePaperDataset
from .model import BACKBONES, CMXPaper, load_imagenet


def main() -> None:
    parser = argparse.ArgumentParser(description="One-batch forward/backward preflight; not a training run.")
    parser.add_argument("--data-dir", default="data/new_data/merged")
    parser.add_argument("--split-dir", default="data/new_data/merged/paper_split_v1")
    parser.add_argument("--pretrained-dir", default="experiments/cmx_initial_transfer/pretrained")
    parser.add_argument("--models", nargs="+", default=["T8", "CMX_B0", "CMX_B1", "CMX_B2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = PrivatePaperDataset(
        args.data_dir, Path(args.split_dir) / "train_indices.npy", training=True, seed=42
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0, pin_memory=False)))
    inputs = batch["inputs"].to(device)
    target = batch["labels"].to(device)
    reports = []
    for name in args.models:
        if name == "T8":
            model = T8TransferNet("T0_R1_REPRO", base_channels=38, num_classes=2)
            pretrained = None
        elif name in BACKBONES:
            model = CMXPaper(name, num_classes=2, image_size=128, decoder_dim=256)
            suffix = name.split("_")[1].lower()
            pretrained = Path(args.pretrained_dir) / f"mit_{suffix}.pth"
            load_imagenet(model, pretrained)
        else:
            raise ValueError(name)
        model.to(device).train()
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            logits = model(inputs)["logits"]
            loss = F.cross_entropy(logits, target)
        loss.backward()
        finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                     for parameter in model.parameters())
        report = {
            "model": name, "shape": list(logits.shape), "loss": float(loss.detach()),
            "finite_gradients": bool(finite), "pretrained": None if pretrained is None else str(pretrained),
        }
        reports.append(report)
        print("SMOKE_OK " + json.dumps(report, ensure_ascii=False), flush=True)
        del model, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({"status": "ok", "reports": reports}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
