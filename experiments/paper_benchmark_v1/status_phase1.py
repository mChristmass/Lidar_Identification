from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


MODELS = ("T8", "CMX_B0", "CMX_B1", "CMX_B2")
SEEDS = (42, 777, 2025)
CONDITIONS = (
    ("private", "natural", ""),
    ("nyuv2", "clean", "nyuv2_clean_medium"),
    ("nyuv2", "joint_medium", "nyuv2_joint_medium"),
)


def run_dir(root: Path, dataset: str, condition_root: str, model: str, seed: int) -> Path:
    base = root / condition_root if condition_root else root
    return base / dataset / "tune" / model / f"seed_{seed}"


def snapshot(root: Path) -> str:
    total = len(CONDITIONS) * len(MODELS) * len(SEEDS)
    completed = 0
    lines = []
    active = []
    failures = []
    for dataset, condition, condition_root in CONDITIONS:
        for model in MODELS:
            scores = []
            for seed in SEEDS:
                directory = run_dir(root, dataset, condition_root, model, seed)
                result_path = directory / "result.json"
                if result_path.is_file():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    scores.append(float(result["best_selection_score"]))
                    completed += 1
                    continue
                history_path = directory / "history.json"
                if history_path.is_file():
                    history = json.loads(history_path.read_text(encoding="utf-8")).get("epochs", [])
                    if history:
                        last = history[-1]
                        active.append(
                            f"{dataset}/{condition}/{model}/seed_{seed} "
                            f"epoch={last['epoch']}/100 loss={last['loss']:.6f} "
                            f"dev_miou={last.get('dev_miou', float('nan')):.6f}"
                        )
                failure = directory / "failure.json"
                if failure.is_file():
                    failures.append(str(failure))
            if scores:
                values = np.asarray(scores, dtype=np.float64)
                spread = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
                lines.append(
                    f"GROUP dataset={dataset} condition={condition} model={model} "
                    f"completed={len(scores)}/3 dev_miou={values.mean():.6f}±{spread:.6f}"
                )
    header = f"OVERALL completed={completed}/{total} progress={100 * completed / total:.2f}%"
    if active:
        header += f" active={active[-1]}"
    if failures:
        header += f" failures={len(failures)} latest={failures[-1]}"
    return "\n".join([header, *lines])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the complete phase-1 experiment.")
    parser.add_argument("--runs-dir", type=Path, default=Path("data/paper_benchmark_runs/phase1_v1"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=1800)
    args = parser.parse_args()
    while True:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), snapshot(args.runs_dir), flush=True)
        if not args.watch:
            break
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
