# Scripts

Shell glue around `eval.py` and `train.py`. Run them from anywhere; they `cd`
to the repository root themselves.

| Script | What it does | Needs |
| --- | --- | --- |
| `reproduce_paper.sh` | redraws every paper figure the committed data supports into `outputs/paper_reproduction/` (override with `OUTPUT_ROOT`) | `paper_data/` only: no checkpoints, GPU or network |
| `rebuild_sweep_figures.sh` | redraws the LR, balls-and-urns and Markov sweep figures, prior and posterior, for all three positional encodings | each sweep's `metrics.csv` under `outputs/` |
| `rebuild_marginals.sh` | redraws every marginal grid for all three positional encodings, with the options each needs | the sweeps' sample bundles under `outputs/` |
| `rebuild_prior_sidecars.sh` | recomputes the prior-panel sidecar CSVs (`eval.py lr-prior-delta-mse`, `bau-prior-kl`) | the sweep checkpoints; a GPU helps |

`reproduce_paper.sh` uses `uv run`; the `rebuild_*` scripts activate `.venv`
and call `python` directly, as the cluster jobs do. Each `rebuild_*` script
prints one `PASS` or `FAIL` line per step and carries on past failures.

## Cluster sync scripts

`m3/` holds the authors' helpers for moving code and results between a
workstation and the Monash M3 cluster. They read the remote from the
environment or from `m3/remote.env`, which git ignores:

```bash
cp scripts/m3/remote.env.example scripts/m3/remote.env   # once; set REMOTE and REMOTE_ROOT
```

| Script | What it does |
| --- | --- |
| `m3/sync_code.sh [--dry-run]` | sends every added, modified or untracked source file to the cluster checkout. Renames and deletions are not propagated: after a commit that moves files, `git pull` on the cluster instead |
| `m3/pull_figures.sh [--dry-run]` | fetches every `.png` and `.pdf` under the cluster's `outputs/`, leaving the data behind |
| `m3/pull_pmc_diagnostics.sh [--dry-run] [--with-checkpoints]` | fetches `outputs/pmc_diagnostics/` in full (enough to redraw and re-aggregate every diagnostic); `--with-checkpoints` also fetches every input file the run records name |
