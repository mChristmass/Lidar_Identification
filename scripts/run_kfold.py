import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from configs import kfold_config as kfold


def parse_args():
    parser = argparse.ArgumentParser(description="Run automated 5-fold stage1/stage2 experiments.")
    parser.add_argument(
        "--stage",
        choices=["stage1", "predict", "build_stage2", "stage2", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Do not save per-frame stage1 prediction PNG files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    kfold.ensure_data_layout()

    if args.stage in {"stage1", "all"}:
        from scripts.train_stage1 import train_all_folds as train_stage1_all_folds

        train_stage1_all_folds()

    if args.stage in {"predict", "all"}:
        from scripts.predict_stage1 import predict_all_folds

        predict_all_folds(save_png=not args.no_png)

    if args.stage in {"build_stage2", "all"}:
        from scripts.build_stage2_rois import build_all_folds

        build_all_folds()

    if args.stage in {"stage2", "all"}:
        from scripts.train_stage2 import train_all_folds as train_stage2_all_folds

        train_stage2_all_folds()


if __name__ == "__main__":
    main()
