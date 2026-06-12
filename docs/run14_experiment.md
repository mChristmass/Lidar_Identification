# Run14 dual-branch follow-up

Run14 follows directly from the run13 gate analysis.

- `E1`: use spatial gates at encoder scales 1-3 and disable edge injection at
  scales 4-5.
- `E2`: keep the D3 five-scale gates and add boundary loss with weight `0.1`.
- `E3`: keep the D3 five-scale gates and add boundary loss with weight `0.2`.

Run all three five-fold experiments:

```powershell
python scripts/run_dual_branch_followup_kfold.py
```

Run one experiment:

```powershell
python scripts/run_dual_branch_followup_kfold.py --experiment E1
```

Outputs are written to `data/runs/run14`.
