# Paper benchmark v1

This package freezes the data and robustness protocol before model-innovation
experiments. It supports the private binary SPL task and NYUv2-40 semantic
segmentation with a one-channel appearance input plus raw depth.

## Scientific scope

- Private corruption calibration reads `paper_split_v1/train_indices.npy` only.
- NYUv2 uses RGB converted to grayscale as an **appearance proxy**, not as real
  single-photon intensity.
- NYUv2 uses `rawDepths`, not the inpainted `depths` variable.
- The Poisson/background-noise condition is a reproducible stress-test proxy;
  it is not described as a physically exact SPL simulator.
- Modified NYUv2 inputs require retraining every baseline. Published RGB-D SOTA
  numbers are context only and cannot be used as direct competitors.

## Data preparation

Private data audit (already safe to rerun):

```bash
python scripts/audit_private_train_degradation.py
```

Download and extract the 2.8 GB NYUv2 labeled file on the server:

```bash
bash scripts/prepare_nyuv2_paper.sh
```

The extraction creates the standard 795/654 split, plus a fixed 715/80 internal
train/dev split used only for tuning. After the recipe is frozen, `stage=final`
re-trains on all 795 official training images and evaluates the 654 test images.

## Pre-flight checks

```bash
python -m experiments.paper_benchmark_v1.test_paper_benchmark

python -u -m experiments.paper_benchmark_v1.smoke --device cuda --amp

python -m experiments.paper_benchmark_v1.run audit \
  --dataset private --stage tune

python -m experiments.paper_benchmark_v1.run audit \
  --dataset nyuv2 --stage tune
```

Keep `--workers 0` for the first server run. It intentionally avoids the
pin-memory worker failure seen in the previous CMX experiment. Increase workers
and enable pin memory only after a short server smoke test passes.

## Tuning-stage examples

Private (final_test is not read):

```bash
python -u -m experiments.paper_benchmark_v1.run train \
  --dataset private --stage tune --model CMX_B2 \
  --height 128 --width 128 --batch-size 8 --epochs 100 \
  --pretrained experiments/cmx_initial_transfer/pretrained/mit_b2.pth \
  --device cuda --amp
```

NYUv2 clean grayscale/raw-depth control:

```bash
python -u -m experiments.paper_benchmark_v1.run train \
  --dataset nyuv2 --stage tune --model CMX_B2 \
  --corruption clean --height 480 --width 640 \
  --batch-size 2 --grad-accum 4 --epochs 100 \
  --pretrained experiments/cmx_initial_transfer/pretrained/mit_b2.pth \
  --device cuda --amp
```

Replace `clean` with `random_missing`, `block_missing`, `edge_missing`,
`low_signal_missing`, `depth_noise`, `photon_proxy`, or `joint`, and select
`--severity light|medium|heavy`. For degradation-aware training, `mixture`
deterministically balances corruption families and severities across samples.

Evaluate one fixed tuning checkpoint over the complete dev corruption suite:

```bash
python -u -m experiments.paper_benchmark_v1.run evaluate \
  --dataset nyuv2 --stage tune --model CMX_B2 \
  --checkpoint data/paper_benchmark_runs/v1/nyuv2/tune/CMX_B2/seed_42/best.pth \
  --height 480 --width 640 --batch-size 2 --device cuda --amp
```

This evaluates 22 conditions without retraining: clean plus seven corruption
families at three severities. The output records each mIoU and its drop from the
same checkpoint's clean score.

Do not run `--stage final` until architecture, corruption suite, hyperparameters,
and epoch count have been frozen in the experiment registry.

## Complete phase-1 matrix

After the pre-flight commands pass, the frozen first experiment can be launched
with:

```bash
python -u -m experiments.paper_benchmark_v1.launch_phase1 \
  --datasets private nyuv2 --amp --skip-existing \
  > data/paper_benchmark_runs/phase1_v1_launcher.log 2>&1
```

Every seed prints one `RUN_SUMMARY` with the overall completed/total count. After
three seeds of one model/condition, it prints `GROUP_3SEED_SUMMARY` with mean and
standard deviation. The launcher never reads either final test split.

The recommended all-in-one server entry performs data preparation, every
pre-flight check, and all 36 phase-1 runs:

```bash
mkdir -p data/paper_benchmark_runs/phase1_v1

nohup bash scripts/run_paper_phase1.sh 0 \
  data/paper_benchmark_runs/phase1_v1 \
  data/public_semseg/nyuv2/processed \
  data/public_semseg/nyuv2/raw \
  > data/paper_benchmark_runs/phase1_v1/launcher.log 2>&1 &

echo $!
```

Read a snapshot or watch it every 30 minutes:

```bash
python -u -m experiments.paper_benchmark_v1.status_phase1 \
  --runs-dir data/paper_benchmark_runs/phase1_v1

python -u -m experiments.paper_benchmark_v1.status_phase1 \
  --runs-dir data/paper_benchmark_runs/phase1_v1 \
  --watch --interval 1800
```
