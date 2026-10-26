# Frozen paper data

Small, immutable inputs committed with the code so that figures can be redrawn
without checkpoints, a GPU or network access. Generated figures are not
tracked here; [`scripts/reproduce_paper.sh`](../scripts/reproduce_paper.sh)
redraws them under `outputs/paper_reproduction/`.

**Vintage.** The `lr/`, `bau/` and `markov/` snapshots are the *original*
(submission-era) runs: linear regression with an 8-pair prompt and 56 rollout
steps, balls-and-urns with a 16-token prompt, Markov with a 1024-state context.
Several camera-ready figures were redrawn from the retrained runs instead (no
positional encoding for LR and balls-and-urns, 256 positions everywhere, a
32-observation prompt). Their inputs are frozen separately in
[`camera_ready/`](camera_ready/README.md): the LR and balls-and-urns sweep
metrics and every input of the 41 reliability figures.
[`docs/figure_provenance.md`](../docs/figure_provenance.md) says which input
each paper figure uses. Each snapshot keeps the `config.json` it was produced
with, written by the code of that time, so some keys (the Laplace options of
`lr/sweep/config.json`, for instance) have no counterpart in today's CLI.

## Contents

```text
paper_data/
  lr/
    sweep/                 aggregate and per-prompt task-diversity metrics, prior Delta-MSE sidecar
    dynamics/              aggregate and per-prompt M=32 training dynamics
    eval_datasets/         fixed generalizing prompts (shared.npz) and per-run memorizing prompts
  bau/
    sweep/                 aggregate task-diversity metrics, prior-only (R=1024) metrics, prior KL sidecar
    dynamics/              aggregate and per-prompt M=32 training dynamics
    eval_datasets/         fixed generalizing and per-run memorizing prompts
  markov/
    sweep/                 aggregate ED/SW metrics, prior-only metrics, prior KL, per-run KL histories
    dynamics_m32/          M=32 aggregate and per-prompt metrics and KL history
    dynamics_m8/           M=8 aggregate and per-prompt metrics and KL history
  beta_bernoulli/          the 100k-step Beta-Bernoulli model, its PMC samples and summary (Figure 1)
  camera_ready/            inputs of the figures redrawn from the retrained runs; see its README
  rebuttal_diagnostics/    the rebuttal-period reliability diagnostics; see its README
```

`per_prompt_metrics.csv` holds the prompt-level measurements behind the
corresponding aggregate `metrics.csv` rows (the balls-and-urns sweep has only
the aggregate). The Markov `wandb_kl_history.csv` files are KL histories
originally exported from Weights & Biases; the plotters read them locally and
need no credentials. The `eval_datasets/` directories let the original paper
checkpoints be re-evaluated on exactly the prompts they were scored on (pass
`--eval-dataset-dir`).

## Redrawing the figures

All at once:

```bash
bash scripts/reproduce_paper.sh                                   # -> outputs/paper_reproduction/
OUTPUT_ROOT=/tmp/bft-pmc-figures bash scripts/reproduce_paper.sh  # elsewhere
```

Or one at a time:

```bash
uv run eval.py plot-lr-sweep --metrics-csv paper_data/lr/sweep/metrics.csv --prompt-length 8 \
    --output-dir outputs/paper_reproduction/lr/sweep
uv run eval.py plot-lr-dynamics --metrics-csv paper_data/lr/dynamics/metrics.csv \
    --output-dir outputs/paper_reproduction/lr/dynamics
uv run eval.py plot-lr-prior --metrics-csv paper_data/lr/sweep/metrics.csv \
    --delta-csv paper_data/lr/sweep/prior_delta_mse.csv \
    --out-path outputs/paper_reproduction/lr/sweep/sweep_prior_delta_ed_sw.png

uv run eval.py plot-bau-sweep --metrics-csv paper_data/bau/sweep/metrics.csv \
    --output-dir outputs/paper_reproduction/bau/sweep
uv run eval.py plot-bau-dynamics --metrics-csv paper_data/bau/dynamics/metrics.csv \
    --output-dir outputs/paper_reproduction/bau/dynamics
uv run eval.py plot-bau-prior --metrics-csv paper_data/bau/sweep/metrics_prior_n1024.csv \
    --kl-csv paper_data/bau/sweep/prior_predictive_kl.csv \
    --out-path outputs/paper_reproduction/bau/sweep/sweep_prior_kl_ed_sw.png

uv run eval.py plot-markov-sweep --runs-dir paper_data/markov/sweep/runs \
    --metrics-csv paper_data/markov/sweep/metrics.csv \
    --out-path outputs/paper_reproduction/markov/sweep/sweep_combined.png
uv run eval.py plot-markov-dynamics --runs-dir paper_data/markov/dynamics_m8/runs \
    --out-dir outputs/paper_reproduction/markov/dynamics_m8

uv run train.py beta-bernoulli --replot-from paper_data/beta_bernoulli/pmc_samples.npz
```

## Not included

PMC sample bundles, their rollouts, and model checkpoints are too large to
commit. The marginal figures need the bundles; re-scoring metrics, rerunning
PMC or retraining needs the checkpoints. [`docs/pmc_outputs.md`](../docs/pmc_outputs.md)
documents how to generate the bundles and what they contain, and
[`docs/model_zoo.md`](../docs/model_zoo.md) the checkpoint layout.

## Git

`paper_data/.gitignore` re-includes CSV and NPZ files, so the data survives a
global ignore rule for data files, and excludes generated PNG and PDF figures,
except the rebuttal-period figures, whose inputs were not kept.
