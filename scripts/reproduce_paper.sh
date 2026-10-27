#!/bin/bash
# Redraw every paper figure the committed data supports, with no checkpoints, GPU
# or network access:
#   - the Beta-Bernoulli figures (Figure 1) and the identifiability example (Figure 2);
#   - from paper_data/camera_ready/: the camera-ready LR and balls-and-urns sweep
#     figures and all 41 reliability figures of Appendix D and the supplement;
#   - from paper_data/{lr,bau,markov}/: the aggregate task-diversity, prior and
#     training-dynamics figures of the original runs (see paper_data/README.md).
#
#   bash scripts/reproduce_paper.sh                          # -> outputs/paper_reproduction/
#   OUTPUT_ROOT=/tmp/figures bash scripts/reproduce_paper.sh
#
# The marginal figures need PMC sample bundles and the reliability figures the
# diagnostics outputs; docs/figure_provenance.md lists every figure's inputs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# A relative OUTPUT_ROOT is taken relative to where the script is run from.
if [[ -n "${OUTPUT_ROOT:-}" && "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="$(pwd)/${OUTPUT_ROOT}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/paper_reproduction}"
cd "${REPO_ROOT}"

uv run train.py beta-bernoulli \
    --replot-from paper_data/beta_bernoulli/pmc_samples.npz \
    --output-dir "${OUTPUT_ROOT}/beta_bernoulli"

uv run eval.py plot-identifiability \
    --out-path "${OUTPUT_ROOT}/misc/intro_identifiability_spike_slab.png"

# ---- Camera-ready figures, from the retrained runs ---------------------------

uv run eval.py plot-lr-sweep \
    --metrics-csv paper_data/camera_ready/lr/sweep/metrics.csv \
    --prompt-length 32 \
    --output-dir "${OUTPUT_ROOT}/camera_ready/lr/sweep"

uv run eval.py plot-bau-sweep \
    --metrics-csv paper_data/camera_ready/bau/sweep/metrics.csv \
    --output-dir "${OUTPUT_ROOT}/camera_ready/bau/sweep"

uv run eval.py plot-reliability \
    --diagnostics-root paper_data/camera_ready/pmc_diagnostics \
    --paper-dir "${OUTPUT_ROOT}/camera_ready/reliability"

# ---- Original-run figures ----------------------------------------------------

uv run eval.py plot-lr-sweep \
    --metrics-csv paper_data/lr/sweep/metrics.csv \
    --prompt-length 8 \
    --output-dir "${OUTPUT_ROOT}/lr/sweep"

uv run eval.py plot-lr-dynamics \
    --metrics-csv paper_data/lr/dynamics/metrics.csv \
    --output-dir "${OUTPUT_ROOT}/lr/dynamics"

uv run eval.py plot-bau-sweep \
    --metrics-csv paper_data/bau/sweep/metrics.csv \
    --output-dir "${OUTPUT_ROOT}/bau/sweep"

uv run eval.py plot-bau-dynamics \
    --metrics-csv paper_data/bau/dynamics/metrics.csv \
    --output-dir "${OUTPUT_ROOT}/bau/dynamics"

uv run eval.py plot-lr-prior \
    --metrics-csv paper_data/lr/sweep/metrics.csv \
    --delta-csv paper_data/lr/sweep/prior_delta_mse.csv \
    --out-path "${OUTPUT_ROOT}/lr/sweep/sweep_prior_delta_ed_sw.png"

uv run eval.py plot-lr-sweep-and-dynamics \
    --sweep-csv paper_data/lr/sweep/metrics.csv \
    --dynamics-csv paper_data/lr/dynamics/metrics.csv \
    --out-path "${OUTPUT_ROOT}/lr/sweep_and_dynamics.png"

uv run eval.py plot-bau-prior \
    --metrics-csv paper_data/bau/sweep/metrics_prior_n1024.csv \
    --kl-csv paper_data/bau/sweep/prior_predictive_kl.csv \
    --out-path "${OUTPUT_ROOT}/bau/sweep/sweep_prior_kl_ed_sw.png"

uv run eval.py plot-markov-sweep \
    --runs-dir paper_data/markov/sweep/runs \
    --metrics-csv paper_data/markov/sweep/metrics.csv \
    --out-path "${OUTPUT_ROOT}/markov/sweep/sweep_combined.png"

uv run eval.py plot-markov-prior \
    --metrics-csv paper_data/markov/sweep/metrics_prior_n1024.csv \
    --kl-csv paper_data/markov/sweep/prior_predictive_kl.csv \
    --out-path "${OUTPUT_ROOT}/markov/sweep/sweep_prior_kl_ed_sw.png"

uv run eval.py plot-markov-dynamics \
    --runs-dir paper_data/markov/dynamics_m32/runs \
    --out-dir "${OUTPUT_ROOT}/markov/dynamics_m32"

uv run eval.py plot-markov-dynamics \
    --runs-dir paper_data/markov/dynamics_m8/runs \
    --out-dir "${OUTPUT_ROOT}/markov/dynamics_m8"
