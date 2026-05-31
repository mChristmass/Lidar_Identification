import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch


class TeeLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.stdout
        self.file = path.open("w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.file.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


@contextmanager
def tee_stdout(path: Path):
    logger = TeeLogger(path)
    previous = sys.stdout
    sys.stdout = logger
    try:
        yield
    finally:
        sys.stdout = previous
        logger.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def summarize_metrics(metrics: Iterable[Dict], metric_keys: List[str]) -> Dict:
    rows = list(metrics)
    summary = {"folds": rows, "mean": {}, "std": {}}
    for key in metric_keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        summary["mean"][key] = float(values.mean())
        summary["std"][key] = float(values.std(ddof=0))
    return summary


def print_summary(title: str, summary: Dict, metric_keys: List[str]) -> None:
    print(f"\n===== {title} =====")
    for row in summary["folds"]:
        values = "  ".join(f"{key}: {row[key]:.4f}" for key in metric_keys)
        print(f"Seed {row['seed']}: {values}")
    mean_values = "  ".join(f"{key}: {summary['mean'][key]:.4f}" for key in metric_keys)
    std_values = "  ".join(f"{key}: {summary['std'][key]:.4f}" for key in metric_keys)
    print(f"Mean: {mean_values}")
    print(f"Std : {std_values}")
