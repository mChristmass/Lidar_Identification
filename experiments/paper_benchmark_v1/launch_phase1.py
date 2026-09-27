from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


MODELS = ("T8", "CMX_B0", "CMX_B1", "CMX_B2")
DEFAULT_SEEDS = (42, 777, 2025)


def result_path(runs: Path, dataset: str, model: str, seed: int) -> Path:
    return runs / dataset / "tune" / model / f"seed_{seed}" / "result.json"


def run_one(args, dataset: str, model: str, seed: int, corruption: str, severity: str) -> dict:
    destination = result_path(Path(args.runs_dir), dataset, model, seed)
    # Public conditions must not overwrite one another.
    if dataset == "nyuv2":
        condition_root = Path(args.runs_dir) / f"nyuv2_{corruption}_{severity}"
        destination = result_path(condition_root, "nyuv2", model, seed)
        runs_dir = condition_root
    else:
        runs_dir = Path(args.runs_dir)
    if destination.is_file() and args.skip_existing:
        return json.loads(destination.read_text(encoding="utf-8"))

    command = [
        sys.executable, "-u", "-m", "experiments.paper_benchmark_v1.run", "train",
        "--dataset", dataset, "--stage", "tune", "--model", model,
        "--seed", str(seed), "--epochs", str(args.epochs),
        "--runs-dir", str(runs_dir), "--device", args.device,
        "--workers", str(args.workers), "--corruption", corruption, "--severity", severity,
    ]
    if args.amp:
        command.append("--amp")
    if dataset == "private":
        command += ["--height", "128", "--width", "128", "--batch-size", str(args.private_batch)]
    else:
        command += [
            "--height", "480", "--width", "640", "--batch-size", str(args.public_batch),
            "--grad-accum", str(args.public_grad_accum), "--nyuv2-dir", args.nyuv2_dir,
        ]
    if model.startswith("CMX_"):
        suffix = model.split("_")[1].lower()
        command += ["--pretrained", str(Path(args.pretrained_dir) / f"mit_{suffix}.pth")]
    try:
        subprocess.run(command, check=True)
    except Exception as error:
        destination.parent.mkdir(parents=True, exist_ok=True)
        (destination.parent / "failure.json").write_text(
            json.dumps({
                "dataset": dataset, "model": model, "seed": seed,
                "corruption": corruption, "severity": severity,
                "command": command, "error": repr(error), "time": time.time(),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise
    return json.loads(destination.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen phase-1 framework diagnosis matrix.")
    parser.add_argument("--datasets", nargs="+", choices=("private", "nyuv2"), default=["private", "nyuv2"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--runs-dir", default="data/paper_benchmark_runs/phase1_v1")
    parser.add_argument("--nyuv2-dir", default="data/public_semseg/nyuv2/processed")
    parser.add_argument("--pretrained-dir", default="experiments/cmx_initial_transfer/pretrained")
    parser.add_argument("--private-batch", type=int, default=8)
    parser.add_argument("--public-batch", type=int, default=2)
    parser.add_argument("--public-grad-accum", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    conditions = {
        "private": [("natural", "clean", "medium")],
        "nyuv2": [("clean", "clean", "medium"), ("joint_medium", "joint", "medium")],
    }
    jobs = [
        (dataset, label, corruption, severity, model, seed)
        for dataset in args.datasets
        for label, corruption, severity in conditions[dataset]
        for model in args.models
        for seed in args.seeds
    ]
    completed = 0
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for dataset, label, corruption, severity, model, seed in jobs:
        result = run_one(args, dataset, model, seed, corruption, severity)
        completed += 1
        score = float(result["best_selection_score"])
        key = (dataset, label, model)
        grouped.setdefault(key, []).append(score)
        print(
            f"RUN_SUMMARY dataset={dataset} condition={label} model={model} seed={seed} "
            f"dev_miou={score:.6f} overall={completed}/{len(jobs)} ({100 * completed / len(jobs):.2f}%)",
            flush=True,
        )
        if len(grouped[key]) == len(args.seeds):
            values = np.asarray(grouped[key], dtype=np.float64)
            print(
                f"GROUP_3SEED_SUMMARY dataset={dataset} condition={label} model={model} "
                f"dev_miou={values.mean():.6f}±{values.std(ddof=1):.6f}", flush=True,
            )


if __name__ == "__main__":
    main()
