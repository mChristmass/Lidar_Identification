#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-SEQUENTIAL}"
python -u -m experiments.paper_benchmark_v1.test_consistency_scale
python -u -m experiments.paper_benchmark_v1.run_consistency_scale --device cuda --budget-hours 10
