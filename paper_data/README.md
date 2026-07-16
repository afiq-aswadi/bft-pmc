# Frozen paper data

This directory contains the small, immutable inputs used to regenerate the
paper figures. Generated figures are ignored; run `scripts/reproduce_paper.sh`
from the repository root to write them under `outputs/paper_reproduction/`.

## Contents

```text
paper_data/
  lr/
    sweep/            aggregate and per-prompt task-diversity metrics
    dynamics/         aggregate and per-prompt M=32 training dynamics
    eval_datasets/    fixed generalising and per-run memorising prompts
  bau/
    sweep/            aggregate task-diversity metrics
    dynamics/         aggregate and per-prompt M=32 training dynamics
    eval_datasets/    fixed generalising and per-run memorising prompts
  markov/
    sweep/            aggregate ED/SW metrics and per-run KL histories
    dynamics_m32/     M=32 aggregate, per-prompt, and KL histories
    dynamics_m8/      M=8 aggregate, per-prompt, and KL histories
```

Where present, `per_prompt_metrics.csv` contains the raw prompt-level
measurements behind the corresponding aggregate `metrics.csv` rows. The BAU
sweep snapshot contains only aggregate metrics; its dynamics snapshot retains
per-prompt measurements.

The Markov `wandb_kl_history.csv` files are committed KL-history snapshots that
were originally exported from W&B. Figure reproduction reads them locally and
does not contact W&B or require credentials.

## Individual figure commands

Linear-regression posterior sweep and dynamics:

```bash
uv run eval.py plot-lr-sweep \
  --metrics-csv paper_data/lr/sweep/metrics.csv \
  --prompt-length 8 \
  --output-dir outputs/paper_reproduction/lr/sweep

uv run eval.py plot-lr-dynamics \
  --metrics-csv paper_data/lr/dynamics/metrics.csv \
  --output-dir outputs/paper_reproduction/lr/dynamics
```

Balls-and-urns sweep and dynamics:

```bash
uv run eval.py plot-bau-sweep \
  --metrics-csv paper_data/bau/sweep/metrics.csv \
  --output-dir outputs/paper_reproduction/bau/sweep

uv run eval.py plot-bau-dynamics \
  --metrics-csv paper_data/bau/dynamics/metrics.csv \
  --output-dir outputs/paper_reproduction/bau/dynamics
```

Markov sweep and dynamics:

```bash
uv run eval.py plot-markov-sweep \
  --runs-dir paper_data/markov/sweep/runs \
  --metrics-csv paper_data/markov/sweep/metrics.csv \
  --out-path outputs/paper_reproduction/markov/sweep/sweep_combined.png

uv run eval.py plot-markov-dynamics \
  --runs-dir paper_data/markov/dynamics_m32/runs \
  --out-dir outputs/paper_reproduction/markov/dynamics_m32

uv run eval.py plot-markov-dynamics \
  --runs-dir paper_data/markov/dynamics_m8/runs \
  --out-dir outputs/paper_reproduction/markov/dynamics_m8
```

The prior-only and stitched LR commands are included in
`scripts/reproduce_paper.sh`.

## Generate sample bundles for marginal plots

The repository includes the code that generates every marginal-plot input.
Only the already-generated Predictive Monte Carlo (PMC) bundles and model
checkpoints are omitted because of their size. Unlike the aggregate sweep and
dynamics plots, a distribution marginal cannot be reconstructed from
`metrics.csv`: it needs the underlying PMC draws.

First train the corresponding sweep, or restore checkpoints using the layout in
[`MODEL_ZOO.md`](../MODEL_ZOO.md). Then run the task-family sweep evaluator as
described below. Distribution analysis is enabled by default; do not pass the
`--no-compute-distribution-metrics` option. The evaluators write both scalar
metrics and the sample bundles consumed by the marginal plotters.

`scripts/reproduce_paper.sh` intentionally remains a checkpoint-free,
CPU-friendly reproduction path and therefore skips the marginal figures. Once
a bundle has been generated, its plotter is fully offline: it does not load a
checkpoint, contact W&B, or rerun PMC. The committed W&B KL-history CSVs contain
scalar metrics, not samples, so they cannot replace these bundles.

### Linear regression

Run the sweep evaluator over the LR checkpoint tree:

```bash
uv run eval.py lr-sweep \
  --checkpoint-root checkpoints/lr/task_diversity \
  --output-dir outputs/lr/sweep_analysis
```

This creates a timestamped directory such as
`outputs/lr/sweep_analysis/sweep_YYYYMMDD_HHMMSS/`, with PMC bundles under its
`samples/` subdirectory. By default, evaluation prompts are generated in
memory. When evaluating the original paper checkpoints, append
`--eval-dataset-dir paper_data/lr/eval_datasets` to reuse the frozen
paper prompts.

Generate the marginal figures from that completed sweep directory:

```bash
uv run python -m linear_regression.plot_marginals \
  --sweep-dir outputs/lr/sweep_analysis/sweep_YYYYMMDD_HHMMSS
uv run python -m linear_regression.plot_stitched_marginals \
  --sweep-dir outputs/lr/sweep_analysis/sweep_YYYYMMDD_HHMMSS
uv run python -m linear_regression.plot_single_marginal \
  --npz-path outputs/lr/sweep_analysis/sweep_YYYYMMDD_HHMMSS/samples/T8_prior.npz \
  --out-path outputs/lr/marginals/prior_M8_dim0.pdf \
  --dim 0
```

### Balls and urns

BAU memorising prompts depend on each run's sampled task pool. For newly
trained checkpoints, generate a matching evaluation dataset first:

```bash
uv run python scripts/generate_bau_eval_dataset.py \
  --checkpoint-root checkpoints/bau/task_diversity \
  --output-dir outputs/bau/eval_datasets
```

Then run the sweep evaluator using that dataset:

```bash
uv run eval.py bau-sweep \
  --checkpoint-root checkpoints/bau/task_diversity \
  --eval-dataset-dir outputs/bau/eval_datasets \
  --output-dir outputs/bau/sweep_analysis
```

When evaluating the original paper checkpoints, use
`paper_data/bau/eval_datasets` instead of `outputs/bau/eval_datasets`. The
evaluator creates a timestamped directory such as
`outputs/bau/sweep_analysis/YYYYMMDD_HHMMSS/`, with PMC bundles under
`samples/`.

Generate the marginal figures from that completed sweep directory:

```bash
uv run python -m balls_and_urns.plot_marginals \
  --sweep-dir outputs/bau/sweep_analysis/YYYYMMDD_HHMMSS
uv run python -m balls_and_urns.plot_stitched_marginals \
  --sweep-dir outputs/bau/sweep_analysis/YYYYMMDD_HHMMSS
```

### Markov chains

Markov evaluation needs the trained transition-matrix pool as well as the model
checkpoints. Keep each run's `resolved_config.yaml`, `transition_matrices.npy`,
and `stationary_distributions.npy` under `outputs/markov/training/<run_name>/`, as
written by the sweep training command, then run:

```bash
uv run eval.py markov-sweep \
  --checkpoint-root checkpoints/markov/task_diversity \
  --training-output-root outputs/markov/training \
  --output-dir outputs/markov/sweep_analysis
```

The final-checkpoint bundles are written directly under
`outputs/markov/sweep_analysis/samples/`; per-checkpoint dynamics
bundles are written under `runs/<run_name>/samples/`. Generate the matrix
marginals with:

```bash
uv run python -m markov.plot_matrix_marginals \
  --samples-dir outputs/markov/sweep_analysis/samples \
  --out-dir outputs/markov/matrix_marginals
```

`markov.run_pmc` produces standalone artifacts for one checkpoint. Use
`markov-sweep` for the multi-M file layout expected by
`markov.plot_matrix_marginals`.

### Generated bundle layout

| Family | Expected sample files | Minimum fields used by the plotters |
| --- | --- | --- |
| Linear regression | `<sweep-dir>/samples/T{M}_prior.npz` and `T{M}_{discrete,gaussian}_L{L}.npz` | `pt`, `theta_pool`, and `dmmse_weights`; posterior files also need `baseline_generalising_posterior_means` and `baseline_generalising_posterior_covs` |
| Balls and urns | `<sweep-dir>/samples/{run_id}__source_{prior,data_generalising,data_memorising}.npz` | `model_samples`, `theta_pool`, `posterior_pool_weights`, `prior_dirichlet_alpha`, `posterior_dirichlet_alpha`, and `prompt_source` |
| Markov | `<samples-dir>/M{M}_prior.npz`, `M{M}_in_distribution_L8.npz`, and `M{M}_out_of_distribution_L8.npz` | `model_samples` and `training_transition_matrices`; posterior files also need `prompt_tokens` |

Use `uv run eval.py <task>-sweep --help` to inspect sampling counts, prompt
lengths, device selection, and other evaluation arguments.

## Data not included

The following artifacts are intentionally external:

- Predictive Monte Carlo sample bundles (`samples/*.npz`), including the Markov
  transition-matrix bundles;
- model checkpoints.

The sample bundles are required for the marginal figures and let metrics be
re-scored without rerunning PMC. Checkpoints are required to regenerate the
bundles, rerun the evaluation, or retrain models. Neither is needed for the
aggregate figures generated from the frozen metric tables. See
[`MODEL_ZOO.md`](../MODEL_ZOO.md) for the expected checkpoint layout.

## Metric convention

Energy distance uses the unbiased U-statistic estimator implemented in
`metrics.py`. A finite-sample estimate can therefore be slightly negative when
the two distributions are close; the BAU prior plots use a symmetric-log scale
when those values occur. The frozen tables and the released implementation use
the same convention.
