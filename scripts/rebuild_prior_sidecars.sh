#!/bin/bash
# Compute the prior-space sidecar CSVs the 1x3 prior plotters read.
#
#   srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=0:30:00 \
#        bash scripts/rebuild_prior_sidecars.sh
#
# Cheap: one forward pass per model at zero context -- a single x_0 for linear
# regression, a bare [BOS] for balls-and-urns -- compared against two analytic
# baselines. Seconds per run, minutes for the sweep. Nothing like the PMC
# evaluations.
#
# These are not produced by the sweep analysis, which is why the 2026-08 runs
# have no prior_delta_mse.csv or prior_predictive_kl.csv and the prior panels
# could not be drawn from them.

set -uo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=.

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
    run "lr/${ENC} prior delta mse" python scripts/compute_lr_prior_delta_mse.py \
      --checkpoint-root "checkpoints/lr/task_diversity_${ENC}" \
      --noise-std 0.5 --seed 42 \
      --out-csv "${LR_SWEEP}prior_delta_mse.csv"
  fi

  if [[ -n "${BAU_SWEEP}" ]]; then
    run "bau/${ENC} prior predictive kl" python scripts/compute_bau_prior_predictive_kl.py \
      --checkpoint-root "checkpoints/bau/task_diversity_${ENC}" \
      --alpha-value 1.0 \
      --out-csv "${BAU_SWEEP}prior_predictive_kl.csv"
  fi
done

echo "----"
echo "${PASS} passed, ${FAIL} failed"
