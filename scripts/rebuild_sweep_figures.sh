#!/bin/bash
# Rebuild the LR and BAU sweep figures from the 2026-08 sweeps, for all three
# positional encodings, in both prior and posterior modes.
#
#   bash scripts/rebuild_sweep_figures.sh
#
# Plotting only, from each sweep's metrics.csv. Seconds, no GPU.
#
# Note the two LR invocations. Its plotter branches on --prompt-length: 0 takes
# the prior branch and writes sweep_combined.png, 32 takes the posterior branch
# and writes sweep_combined{,_2x2}.png. Both land on the same filename, so they
# need separate output directories or the second silently overwrites the first.
#
# `python -m` throughout: by-path invocation lets balls_and_urns/analysis.py
# shadow the top-level analysis/ package.

set -uo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=.

# The 2026-08 sweeps condition on 32 tokens; earlier runs used 8.
PROMPT_LENGTH="${PROMPT_LENGTH:-32}"

PASS=0
FAIL=0

run() {
  local name="$1"
  shift
  local out
  if out=$("$@" 2>&1); then
    echo "PASS  ${name}"
    PASS=$((PASS + 1))
  else
    echo "FAIL  ${name} :: $(echo "${out}" | tail -2 | tr '\n' ' ' | cut -c1-160)"
    FAIL=$((FAIL + 1))
  fi
}

for ENC in learned rope none; do
  LR_SWEEP=$(ls -d outputs/lr/sweep_analysis_"${ENC}"/sweep_*/ 2>/dev/null | tail -1)
  BAU_SWEEP=$(ls -d outputs/bau/sweep_analysis_"${ENC}"/*/ 2>/dev/null | tail -1)

  if [[ -n "${LR_SWEEP}" ]]; then
    run "lr/${ENC} sweep posterior" python -m linear_regression.plot_sweep_combined \
      --metrics-csv "${LR_SWEEP}metrics.csv" \
      --output-dir "${LR_SWEEP}figures_sweep_posterior" \
      --prompt-length "${PROMPT_LENGTH}"
    run "lr/${ENC} sweep prior" python -m linear_regression.plot_sweep_combined \
      --metrics-csv "${LR_SWEEP}metrics.csv" \
      --output-dir "${LR_SWEEP}figures_sweep_prior" \
      --prompt-length 0
  else
    echo "SKIP  lr/${ENC}: no sweep directory found"
  fi

  if [[ -n "${LR_SWEEP}" ]]; then
    # Prior-space panels come from a different plotter than the sweep grids.
    # The Delta MSE panel needs a sidecar CSV from
    # scripts/compute_lr_prior_delta_mse.py, which requires the checkpoints; the
    # step is reported as FAIL rather than skipped when that CSV is absent.
    run "lr/${ENC} prior panels" python -m linear_regression.plot_sweep_prior \
      --metrics-csv "${LR_SWEEP}metrics.csv" \
      --delta-csv "${LR_SWEEP}prior_delta_mse.csv" \
      --out-path "${LR_SWEEP}figures_sweep_prior/lr_sweep_prior_delta_ed_sw.pdf"
  fi

  if [[ -n "${BAU_SWEEP}" ]]; then
    # BAU's plotter emits sweep_posterior{,_2x2} and sweep_prior in one call.
    run "bau/${ENC} sweep" python -m balls_and_urns.plot_sweep_combined \
      --metrics-csv "${BAU_SWEEP}metrics.csv" \
      --output-dir "${BAU_SWEEP}figures_sweep"
    # --kl-csv defaults to a path the 2026-08 sweeps never wrote; point it at
    # the sidecar from scripts/rebuild_prior_sidecars.sh.
    run "bau/${ENC} prior panels" python -m balls_and_urns.plot_sweep_prior \
      --metrics-csv "${BAU_SWEEP}metrics.csv" \
      --kl-csv "${BAU_SWEEP}prior_predictive_kl.csv" \
      --out-path "${BAU_SWEEP}figures_sweep/bau_sweep_prior_kl_ed_sw.pdf"
  else
    echo "SKIP  bau/${ENC}: no sweep directory found"
  fi
done

# ---- Markov (no positional-encoding variants) ------------------------------
MARKOV=outputs/markov/sweep_analysis
if [[ -f "${MARKOV}/metrics.csv" ]]; then
  run "markov sweep" python -m markov.plot_sweep_combined \
    --runs-dir "${MARKOV}/runs" \
    --metrics-csv "${MARKOV}/metrics.csv" \
    --out-path outputs/markov/figures_sweep/markov_sweep_combined.pdf
  run "markov sweep prior" python -m markov.plot_sweep_prior \
    --metrics-csv "${MARKOV}/metrics.csv" \
    --out-path outputs/markov/figures_sweep/markov_sweep_prior_kl_ed_sw.pdf
  # KL comes from outputs/markov/training/<run>/training_log.csv, which this
  # sweep did write -- the W&B export the plotters used before does not exist.
  run "markov dynamics" python -m markov.plot_dynamics_combined \
    --runs-dir "${MARKOV}/runs" \
    --out-dir outputs/markov/figures_dynamics
else
  echo "SKIP  markov: no metrics.csv under ${MARKOV}"
fi

echo "----"
echo "${PASS} passed, ${FAIL} failed"
