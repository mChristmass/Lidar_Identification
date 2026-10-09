# Group-meeting visualization export

This is inference and figure export only. It uses the frozen private and NYUv2
development sets, never either final test set, and never modifies checkpoints.

Outputs include data-processing panels, seven heavy degradation families,
three joint severities, and automatically selected qualitative predictions for
T8 versus CMX, clean versus mixed training, fusion/reliability, restoration,
and observation-structure experiments.
`selection_manifest.json` records sample IDs and per-image scores. Samples are
selected as diagnostic cases (largest improvement, degradation, and model
disagreement), so slides must not describe them as random examples.

Run from the server repository root:

```bash
mkdir -p data/paper_benchmark_runs/group_meeting_visuals_v1
nohup bash scripts/export_group_meeting_visuals_v1.sh 0 \
  data/paper_benchmark_runs/group_meeting_visuals_v1 \
  > data/paper_benchmark_runs/group_meeting_visuals_v1/launcher.log 2>&1 &
echo $!
```

View brief progress with:

```bash
tail -f data/paper_benchmark_runs/group_meeting_visuals_v1/launcher.log
```

Successful completion writes `complete.json`. Sync the entire
`data/paper_benchmark_runs/group_meeting_visuals_v1/` directory locally.

Required server inputs are the complete `data/public_semseg/nyuv2/processed/`
directory, the frozen private data under `data/new_data/merged/`, and existing
checkpoints under `data/paper_benchmark_runs/` for mixed control, reliability,
fusion, noise, repair, and observation windows.
