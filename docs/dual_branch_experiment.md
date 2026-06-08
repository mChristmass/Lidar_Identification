# Lightweight dual-branch experiment

This experiment extends the C1 baseline without using Stage1 priors or an ROI
pipeline. It keeps the same full-scale intensity and local-depth-edge inputs,
data splits, loss, and threshold selection used by C1.

## Experiments

- `D1`: separate intensity/edge encoders, additive fusion without learned gates.
- `D2`: learned spatial gate at the bottleneck only.
- `D3`: learned spatial gates at all five encoder scales.

The intensity encoder is the main stream. The edge encoder uses half as many
base channels. At each selected scale:

```text
edge feature -> 1x1 projection -> spatial gate -> add to intensity feature
```

The default command runs the main `D3` experiment and writes results to
`data/runs/run13`:

```powershell
python scripts/run_dual_branch_kfold.py
```

Run the complete structural ablation:

```powershell
python scripts/run_dual_branch_kfold.py --experiment all
```

The boundary head is implemented but disabled by default so the initial
comparison changes only the feature-fusion structure. Enable it after D1-D3:

```powershell
python scripts/run_dual_branch_kfold.py --experiment D3 --boundary-loss-weight 0.2
```
