#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
echo 'NOISE_PREFLIGHT synthetic_cpu_tests'
python -u -m experiments.paper_benchmark_v1.test_noise_window
echo 'NOISE_START budget_hours=5 mandatory=C1+D1+D2 conditional=paired_seed777'
python -u -m experiments.paper_benchmark_v1.run_noise_window --device cuda --budget-hours 5
