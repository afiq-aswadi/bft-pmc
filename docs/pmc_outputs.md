# What the evaluation commands write

Every evaluation command writes under `outputs/` (ignored by git). This page
describes those files: how to produce them, what they hold, and which plotters
read them. None of them is committed except the small frozen subsets in
[`paper_data/`](../paper_data/README.md); sample bundles and rollouts run to
tens of gigabytes for a full sweep.

## Sweep and dynamics evaluations

`eval.py {lr,bau,markov}-sweep` evaluates the final checkpoint of every run in
a task-diversity sweep, and `eval.py {lr,bau}-dynamics` every checkpoint of one
run. Each writes:

- `metrics.csv`: one row per run (or checkpoint) and prompt source, with the
  prediction-space metrics (`delta_vs_baseline_*`) and the latent-space ones
  (`dist/{ed,sw}_vs_baseline_{memorising,generalising}`);
- `per_prompt_metrics.csv` where available: the prompt-level measurements
  behind those rows;
- `config.json`: the settings it ran with;
- `samples/`: one PMC sample bundle per run and prompt source, the input of
  the marginal plotters, each with a sibling `<bundle>_rollouts.npz`.

The sweep commands create a timestamped directory under their `--output-dir`
(`outputs/lr/sweep_analysis/sweep_YYYYMMDD_HHMMSS/`,
`outputs/bau/sweep_analysis/YYYYMMDD_HHMMSS/`); the Markov sweep writes
directly to its `--output-dir`.

### Producing the bundles

Distribution analysis is on by default; do not pass
`--no-compute-distribution-metrics`.

```bash
# Linear regression. Add --eval-dataset-dir paper_data/lr/eval_datasets to reuse
# the frozen prompts of the original (pre-camera-ready) paper checkpoints.
uv run eval.py lr-sweep --checkpoint-root checkpoints/lr/task_diversity_none \
    --output-dir outputs/lr/sweep_analysis_none

# Balls and urns. The memorizing prompts depend on each run's urn pool, so
# generate a matching evaluation set first (or use paper_data/bau/eval_datasets
# with the original paper checkpoints).
uv run eval.py bau-eval-dataset --checkpoint-root checkpoints/bau/task_diversity_none \
    --output-dir outputs/bau/eval_datasets
uv run eval.py bau-sweep --checkpoint-root checkpoints/bau/task_diversity_none \
    --eval-dataset-dir outputs/bau/eval_datasets --output-dir outputs/bau/sweep_analysis_none

# 1-Markov. Needs each run's resolved_config.yaml and chain pool as well as its
# checkpoints, under outputs/markov/training/<run>/ as train.py markov writes them.
uv run eval.py markov-sweep --checkpoint-root checkpoints/markov/task_diversity \
    --training-output-root outputs/markov/training --output-dir outputs/markov/sweep_analysis
```

`eval.py markov-pmc` writes the same kind of bundle for a single Markov
checkpoint (`pmc_samples.npz`), outside the sweep layout.

### Bundle names and fields

| Family | Sample bundles | Fields the plotters read |
| --- | --- | --- |
| Linear regression | sweep: `samples/T{M}_prior.npz`, `samples/T{M}_{gaussian,discrete,random}_L{L}.npz`; dynamics: `<run_id>/samples/step_{step}_*.npz` | `pt`, `theta_pool`, `dmmse_weights`; posterior bundles also `baseline_generalising_posterior_means` and `baseline_generalising_posterior_covs` |
| Balls and urns | sweep: `samples/{run_id}__source_{prior,data_generalising,data_memorising}.npz`; dynamics: `<run_id>/samples/step{step:08d}__source_*.npz` | `model_samples`, `theta_pool`, `posterior_pool_weights`, `prior_dirichlet_alpha`, `posterior_dirichlet_alpha`, `prompt_source` |
| 1-Markov | sweep: `samples/M{M}_prior.npz`, `samples/M{M}_{in_distribution,out_of_distribution}_L{L}.npz`; per checkpoint: `runs/<run>/samples/step{step:08d}__source_*.npz` | `model_samples`, `training_transition_matrices`; posterior bundles also `prompt_tokens` |

`L` is the prompt length: 32 for the camera-ready runs, 8 (16 for
balls-and-urns) before them. Linear-regression prompt sources are `gaussian`
(a fresh task from the population prior: out-of-distribution), `discrete` (a
task from the training pool: in-distribution) and `random` (responses
independent of x).

### Rollouts

Every PMC estimator reduces a batch of generated sequences, the rollouts, to a
task estimate. Each sample bundle is written with a sibling
`<name>_rollouts.npz` holding them, so an estimate can be audited or
re-derived (different smoothing, a truncated rollout, sequence-level
diagnostics) without rerunning generation:

| Producer | Arrays |
| --- | --- |
| Linear-regression sweep and dynamics | `rollout_x` `(..., samples, total_len, dim)`, `rollout_y` `(..., samples, total_len)` |
| Balls-and-urns sweep and dynamics | `rollout_tokens` `(prompts, samples, total_len)` |
| Markov sweep | `rollout_states` `(prompts, samples, total_len)` |
| `train.py markov`, `eval.py markov-pmc` (`pmc_samples_rollouts.npz`) | `rollout_prior_states`, `rollout_posterior_states` |
| `train.py beta-bernoulli` (`beta_bernoulli_pmc_samples_rollouts.npz`) | `rollout_prior_tokens`, `rollout_posterior_tokens` |

Posterior rollouts keep the conditioning prompt as their prefix; prior
rollouts hold only generated states. Discrete rollouts are stored in the
smallest lossless integer type, and every archive is compressed. Load one with
`bft_pmc.analysis.rollouts.load_rollout_bundle`. They are much larger than the
estimates they explain, so each producer can skip them: `--no-save-rollouts`
for the sweep, dynamics and `markov-pmc` commands and `train.py beta-bernoulli`,
`--no-pmc-save-rollouts` for `train.py markov`.

## Reliability diagnostics

`eval.py {lr,bau,markov}-rollout-diagnostics` writes one directory per run
under `outputs/pmc_diagnostics/`. For each prompt source (`prior`,
`posterior`) it holds the rollouts and baseline samples
(`{source}_trajectories.npz`, `{source}_baselines.npz`), the curves
(`{source}_recovery_vs_length.csv`, `{source}_drift_vs_length.csv`,
`{source}_recovery_vs_count.csv` and every draw behind the last,
`{source}_recovery_vs_count_resamples.csv`), the per-path drift
(`{source}_drift_paths.npz`) and the figures; plus `run_record.json`
(settings, input checksums, code and environment), `figures.json` (each
figure's kind, title and data files) and `<run>_rollouts.npz`, the input of
the path-stability plot. [`slurm/PMC_DIAGNOSTICS.md`](../slurm/PMC_DIAGNOSTICS.md#what-each-run-saves)
describes every file and the settings of every camera-ready run.

From these directories, without a model:

| Command | Writes |
| --- | --- |
| `eval.py replot-pmc-diagnostics --roots outputs/pmc_diagnostics` | redraws every figure listed in each `figures.json`, in place |
| `eval.py recovery-vs-dimension --inputs 6=<dir> 12=<dir> ...` | `recovery_vs_dimension.csv` (or `recovery_vs_task_diversity.csv` with `--x-axis task-diversity`) and its figure |
| `eval.py path-stability --inputs label=<dir> ...` | the path-stability (jellyfish) curves below |
| `eval.py plot-reliability` | every Appendix D and supplement figure, and `reliability_figure_manifest.json` |

## Path stability

`eval.py path-stability` re-evaluates each family's PMC estimator on every
prefix of the saved rollouts and reports
`E[(1/p) || theta(F_n) - theta(F_N) ||_1]` against `N - n`, the scaled-L1
diagnostic of Ng et al. (TabMGP). Inputs are rollout bundles, directories of
them, or `label=path` pairs; prior bundles are skipped.

| File | Contents |
| --- | --- |
| `path_stability.csv` | one row per setup and prefix: `T`, `scaled_l1`, `p`, `prompt_len`, and the bundle's metadata |
| `path_stability_per_prompt.csv` | the same per prompt, that is per realisation of the observed data |
| `theta_reference.npz` | `theta(F_n)`, `theta(F_N)` and the prefix grid per setup |
| `path_stability*.png`, `.pdf` | all setups, by family, and per positional encoding |
| `run_record.json` | settings, the input bundles and their checksums |

`--stride` evaluates a coarser prefix grid; `--realisations k` draws one curve
for each of the first k prompts instead of averaging over all of them
(`--realisations 1` is TabMGP's single-dataset estimand).

## Metric convention

Energy distance uses the unbiased U-statistic estimator of
`bft_pmc/metrics.py`, so a finite-sample estimate can be slightly negative
when the two distributions are close; the balls-and-urns prior plots switch to
a symmetric-log scale when that happens. The frozen tables and this code use
the same convention.
