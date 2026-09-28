#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
echo 'FUSION_PREFLIGHT synthetic_cpu_tests'
python -u -m experiments.paper_benchmark_v1.test_fusion_window
echo 'FUSION_START planned_window_hours=4.5 mandatory=diagnostic+F1+F2 conditional=paired_seed777'
python -u -m experiments.paper_benchmark_v1.run_fusion_window --device cuda --budget-hours 4.5
