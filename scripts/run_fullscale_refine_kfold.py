import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from configs import kfold_config as kfold
from scripts.train_stage1_only_strong import DEFAULT_THRESHOLDS as STAGE1_THRESHOLDS
from scripts.train_stage1_only_strong import train_all_folds as train_stage1_strong_all_folds
from scripts.predict_stage1_strong_priors import predict_one_fold as predict_stage1_priors_one_fold
from scripts.train_stage2_fullscale_refine import DEFAULT_THRESHOLDS as STAGE2_THRESHOLDS
from scripts.train_stage2_fullscale_refine import train_one_fold as train_stage2_refine_one_fold
from scripts.kfold_utils import print_summary, save_json, summarize_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full-scale Stage1-guided Stage2 refinement k-fold pipeline.")
    parser.add_argument(
        "--stage",
        choices=["stage1", "priors", "stage2", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument("--runs-dir", type=Path, default=kfold.RUNS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=kfold.KFOLD_SEEDS)

    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--stage1-batch-size", type=int, default=4)
    parser.add_argument(
        "--stage1-search",
        action="store_true",
        help="Search Stage1 loss hyperparameters. By default Stage1 uses one fixed setting.",
    )
    parser.add_argument("--stage1-target-weights", type=float, nargs="+", default=[5.0])
    parser.add_argument("--stage1-dice-weights", type=float, nargs="+", default=[1.0])
    parser.add_argument("--stage1-lrs", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--stage1-thresholds", type=float, nargs="+", default=STAGE1_THRESHOLDS)

    parser.add_argument("--prior-batch-size", type=int, default=4)
    parser.add_argument("--prior-threshold", type=float, default=None)
    parser.add_argument("--roi-margin", type=int, default=8)
    parser.add_argument("--roi-dilate-iter", type=int, default=0)

    parser.add_argument("--stage2-epochs", type=int, default=50)
    parser.add_argument("--stage2-batch-size", type=int, default=8)
    parser.add_argument("--stage2-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-thresholds", type=float, nargs="+", default=STAGE2_THRESHOLDS)
    parser.add_argument("--visualize-num", type=int, default=8)
    parser.add_argument("--stage2-output-name", default="stage2_fullscale_refine_error_focused")
    return parser.parse_args()


def run_stage1(args):
    target_weights = args.stage1_target_weights if args.stage1_search else [args.stage1_target_weights[0]]
    dice_weights = args.stage1_dice_weights if args.stage1_search else [args.stage1_dice_weights[0]]
    lrs = args.stage1_lrs if args.stage1_search else [args.stage1_lrs[0]]
    stage1_args = Namespace(
        seeds=args.seeds,
        epochs=args.stage1_epochs,
        batch_size=args.stage1_batch_size,
        target_weights=target_weights,
        dice_weights=dice_weights,
        lrs=lrs,
        thresholds=args.stage1_thresholds,
    )
    return train_stage1_strong_all_folds(stage1_args)


def run_priors(args):
    prior_args = Namespace(
        runs_dir=args.runs_dir,
        seeds=args.seeds,
        batch_size=args.prior_batch_size,
        threshold=args.prior_threshold,
        roi_margin=args.roi_margin,
        roi_dilate_iter=args.roi_dilate_iter,
    )
    for seed in args.seeds:
        predict_stage1_priors_one_fold(seed, prior_args)


def run_stage2(args):
    stage2_args = Namespace(
        runs_dir=args.runs_dir,
        seeds=args.seeds,
        epochs=args.stage2_epochs,
        batch_size=args.stage2_batch_size,
        lr=args.stage2_lr,
        thresholds=args.stage2_thresholds,
        visualize_num=args.visualize_num,
        output_name=args.stage2_output_name,
    )
    results = [train_stage2_refine_one_fold(seed, stage2_args) for seed in args.seeds]
    metric_keys = ["iou", "precision", "recall", "dice", "coverage", "pred_area"]
    summary = summarize_metrics(results, metric_keys)
    save_json(Path(args.runs_dir) / f"summary_{args.stage2_output_name}.json", summary)
    print_summary("Stage2 Full-Scale Refine K-Fold Summary", summary, metric_keys)
    return summary


def main():
    args = parse_args()
    kfold.RUNS_DIR = Path(args.runs_dir)
    kfold.ensure_data_layout()
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Runs dir: {args.runs_dir}")
    print(f"Seeds   : {args.seeds}")

    if args.stage in {"stage1", "all"}:
        run_stage1(args)

    if args.stage in {"priors", "all"}:
        run_priors(args)

    if args.stage in {"stage2", "all"}:
        run_stage2(args)


if __name__ == "__main__":
    main()
