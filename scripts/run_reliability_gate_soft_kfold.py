import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.reliability_gate.config_reliability_gate as conf
from scripts.kfold_utils import save_json
from scripts.run_reliability_gate_kfold import (
    EXPERIMENT_SPECS,
    parse_args,
    run_experiment,
    summarize_paired_comparisons,
)


def main():
    args = parse_args()
    if args.experiments == ["all"]:
        args.experiments = ["R0", "R5", "R6", "R7"]
    if args.folds == list(range(conf.NUM_FOLDS)):
        args.folds = [0, 1]
    default_run4 = Path("data/new_data_run0612/run4")
    if Path(args.runs_dir).as_posix().endswith(default_run4.as_posix()):
        args.runs_dir = Path("data/new_data_run0612/run5")
    if any(fold < 0 or fold >= conf.NUM_FOLDS for fold in args.folds):
        raise ValueError(f"Folds must be in [0, {conf.NUM_FOLDS - 1}].")

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.runs_dir / "experiment_config.json",
        {
            "experiments": args.experiments,
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
            "experiment_specs": {
                name: EXPERIMENT_SPECS[name]
                for name in args.experiments
            },
        },
    )
    for experiment in args.experiments:
        run_experiment(experiment, args)
    summarize_paired_comparisons(args, args.experiments)


if __name__ == "__main__":
    main()
