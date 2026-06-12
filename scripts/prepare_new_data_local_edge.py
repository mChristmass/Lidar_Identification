import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.new_data.config_new_data as conf
from datasets.dataset_stage1_input_ablation import build_local_depth_edge


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute local depth edge for new data.")
    parser.add_argument("--depth", type=Path, default=conf.RAW_ITEMS["depth"])
    parser.add_argument("--output", type=Path, default=conf.RAW_ITEMS["local_depth_edge"])
    parser.add_argument("--erode-iterations", type=int, default=conf.EDGE_ERODE_ITERATIONS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        print(f"Already exists: {args.output}")
        return

    depth = np.load(args.depth, mmap_mode="r")
    output = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.float32,
        shape=depth.shape,
    )
    for index in range(len(depth)):
        output[index] = build_local_depth_edge(
            np.asarray(depth[index], dtype=np.float32),
            erode_iterations=args.erode_iterations,
        )
        if (index + 1) % 100 == 0 or index + 1 == len(depth):
            print(f"Prepared {index + 1}/{len(depth)}")
    output.flush()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
