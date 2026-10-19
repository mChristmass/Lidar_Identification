#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-SEQUENTIAL}"
python -u -m experiments.paper_benchmark_v1.test_pseudo_intensity
python -u -m experiments.paper_benchmark_v1.run_pseudo_intensity --device cuda
