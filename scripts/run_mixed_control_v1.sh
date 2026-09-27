#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${1:-0}"
python -u -m experiments.paper_benchmark_v1.train_mixed_control --device cuda
