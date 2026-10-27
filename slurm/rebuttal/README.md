# Rebuttal-period jobs (historical record)

The one-off jobs that trained the balls-and-urns vocabulary models and ran the
diagnostics now in
[`paper_data/rebuttal_diagnostics/`](../../paper_data/rebuttal_diagnostics/).
They are kept **unchanged, as a record**: they hard-code another author's
cluster paths and conda environment, and they run module files by path from the
pre-cleanup layout (`balls_and_urns/rollout_diagnostics.py`, ...), which no
longer exists. They will not run as they are. In this public copy the
cluster username in their paths reads `<user>`.

| Job | What it did |
| --- | --- |
| `train_nope_v{6,12,24,48}.slurm` | trained one M=32 balls-and-urns model per vocabulary V, no positional encoding |
| `diag_nope_v{6,12,24,48}.slurm` | rollout diagnostics on those models, context extended to 1024 |
| `diag_paper_bau_m32.slurm` | rollout diagnostics on the submission's balls-and-urns M=32 checkpoint |
| `diag_markov_m32.slurm` | rollout diagnostics on the pre-retrain 1-Markov M=32 model |

To run the same diagnostics today, use the camera-ready jobs one directory up,
which go through `eval.py`:

```bash
sbatch slurm/bau_rollout_diagnostics.slurm --checkpoint-path <checkpoint> --output-dir <dir>
sbatch slurm/markov_rollout_diagnostics.slurm --checkpoint-path <checkpoint> --config-path <resolved_config.yaml> --output-dir <dir>
```

or submit the whole camera-ready batch with `bash slurm/submit_pmc_diagnostics.sh`.
