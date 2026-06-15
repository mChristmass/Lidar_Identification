import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.new_data.config_new_data as conf
from scripts.new_data_test_reporting import evaluate_and_report_test_set
from scripts.run_new_data_experiments import (
    EXPERIMENTS,
    load_fold_indices,
    make_dataset,
    make_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate detailed reports and all-frame visualizations from saved checkpoints."
    )
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(conf.NUM_FOLDS)))
    parser.add_argument("--index-dir", type=Path, default=conf.INDEX_DIR)
    parser.add_argument("--batch-size", type=int, default=conf.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-test-visualizations", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_args = SimpleNamespace(
        intensity_base_channels=conf.INTENSITY_BASE_CHANNELS,
        edge_base_channels=conf.EDGE_BASE_CHANNELS,
        light_unet_base_channels=conf.LIGHT_UNET_BASE_CHANNELS,
    )

    for experiment in args.experiments:
        for fold in args.folds:
            out_dir = args.runs_dir / f"fold_{fold}" / experiment
            metrics_path = out_dir / "metrics.json"
            checkpoint_path = out_dir / "best_model.pth"
            summary_path = out_dir / "test_detailed_summary.json"
            if summary_path.exists() and not args.overwrite:
                print(f"Skip existing report: {summary_path}")
                continue
            if not metrics_path.exists() or not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint artifacts in {out_dir}")

            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            _, _, test_indices = load_fold_indices(args.index_dir, fold)
            test_set = make_dataset(test_indices, experiment)
            model = make_model(experiment, test_set.input_channels, model_args).to(conf.DEVICE)
            model.load_state_dict(torch.load(checkpoint_path, map_location=conf.DEVICE))

            print(
                f"Reporting {experiment} fold {fold}: "
                f"{len(test_set)} images, threshold={metrics['best_threshold']}"
            )
            detailed = evaluate_and_report_test_set(
                model=model,
                dataset=test_set,
                raw_items=conf.RAW_ITEMS,
                label_dir=conf.LABEL_DIR,
                device=conf.DEVICE,
                threshold=float(metrics["best_threshold"]),
                output_dir=out_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                save_visualizations=not args.skip_test_visualizations,
            )
            print(
                f"Mean image IoU={detailed['foreground_mean_image_iou']:.4f}, "
                f"background FP frame rate="
                f"{detailed['background_false_positive_frame_rate']:.4f}"
            )


if __name__ == "__main__":
    main()
