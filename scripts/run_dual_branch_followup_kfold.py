import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from configs import kfold_config as kfold
from configs.dual_branch_followup.config_dual_branch_followup import (
    EXPERIMENT_SPECS,
    RUNS_DIR_NAME,
)
from scripts.run_dual_branch_kfold import parse_args, run_from_args


def main():
    args = parse_args(
        experiment_specs=EXPERIMENT_SPECS,
        default_experiment="all",
        default_runs_dir=kfold.DATA_ROOT / f"runs/{RUNS_DIR_NAME}",
        description="Run run14 shallow-gate and boundary-supervision experiments.",
    )
    run_from_args(args)


if __name__ == "__main__":
    main()
