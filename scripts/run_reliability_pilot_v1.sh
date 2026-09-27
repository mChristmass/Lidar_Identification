#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
echo 'RELIABILITY_PREFLIGHT synthetic_cpu_tests'
python -u -m experiments.paper_benchmark_v1.test_reliability
echo 'RELIABILITY_START new_groups=3 epochs_each=20 evaluation_conditions_each=22 R0=reused'
python -u -m experiments.paper_benchmark_v1.train_reliability --device cuda
