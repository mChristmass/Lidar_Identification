# New merged-data experiment plan

Dataset root:

`D:\myComputer\pointsCloud\data\identification\data\new_data\merged`

The merged dataset contains 1,390 samples. Ten group partitions are built from
contiguous 50-frame windows within each source batch. Each of the five outer
folds uses two partitions for test, one for validation, and seven for training,
giving approximately 70%/10%/20% splits. Every sample is used as test data
exactly once across the five folds.

## Experiments

- `I0`: intensity-only UNet baseline.
- `ID`: intensity + normalized raw depth UNet.
- `C1`: intensity + precomputed local depth edge UNet.
- `D1`: separate intensity/edge encoders with direct multiscale fusion.
- `D3`: separate intensity/edge encoders with multiscale spatial gates.

The run14 boundary-loss variants are intentionally excluded because they did
not improve the original dataset consistently.

## Preparation

```powershell
python scripts/prepare_new_data_local_edge.py
python scripts/build_new_data_group_folds.py
```

## Recommended execution

First screen every candidate on fold 0:

```powershell
python scripts/run_new_data_experiments.py --experiments all --folds 0
```

Then run the core comparison on all five folds:

```powershell
python scripts/run_new_data_experiments.py --experiments core
```

Results are written to `data/new_data_run0612/run1`.

Future experiments based on this merged dataset should use sequential folders
under `data/new_data_run0612`, such as `run2` and `run3`.
