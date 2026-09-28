#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
echo 'REPAIR_PREFLIGHT synthetic_cpu_tests'
python -u -m experiments.paper_benchmark_v1.test_repair_window
echo 'REPAIR_START budget_hours=8 mandatory=D1seed777+S1seed42+S2seed42 conditional=three_seed_confirmation'
python -u -m experiments.paper_benchmark_v1.run_repair_window --device cuda --budget-hours 8
