#!/bin/bash
# Rebuild every marginal-grid figure in the paper from the 2026-08 sweeps.
#
#   bash scripts/rebuild_marginals.sh
#
# Plotting only: no GPU, no checkpoints, no regeneration. Minutes, not hours.
#
# Everything is invoked with `python -m` on purpose. Running these scripts by
# path puts their own directory first on sys.path, where `markov/plotting.py`
# shadows the top-level `plotting/` package and `balls_and_urns/analysis.py`
# shadows `analysis/` -- both fail with confusing import errors.

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

# ---- LR and BAU: one full set per positional encoding -----------------------
for ENC in learned rope none; do
  LR_SWEEP=$(ls -d outputs/lr/sweep_analysis_"${ENC}"/sweep_*/ 2>/dev/null | tail -1)
  BAU_SWEEP=$(ls -d outputs/bau/sweep_analysis_"${ENC}"/*/ 2>/dev/null | tail -1)

  if [[ -n "${LR_SWEEP}" ]]; then
    run "lr/${ENC} grid-4dim" python -m linear_regression.plot_marginals \
      --sweep-dir "${LR_SWEEP}" --num-dims 4 --output-subdir figures_marginals_4dim
    run "lr/${ENC} stitched" python -m linear_regression.plot_stitched_marginals \
      --sweep-dir "${LR_SWEEP}" --output-subdir figures_stitched

    # lr_1x1_M8_dim0: the manifest does not say which prompt source, so emit all
    # three and let the paper pick.
    mkdir -p "${LR_SWEEP}figures_single"
    for SRC in discrete gaussian random; do
      NPZ="${LR_SWEEP}samples/T8_${SRC}_L32.npz"
      if [[ -f "${NPZ}" ]]; then
        run "lr/${ENC} 1x1 M8 ${SRC}" python -m linear_regression.plot_single_marginal \
          --npz-path "${NPZ}" --dim 0 \
          --out-path "${LR_SWEEP}figures_single/lr_1x1_M8_dim0_${SRC}.pdf"
      fi
    done
  else
    echo "SKIP  lr/${ENC}: no sweep directory found"
  fi

  if [[ -n "${BAU_SWEEP}" ]]; then
    run "bau/${ENC} grid-4dim" python -m balls_and_urns.plot_marginals \
      --sweep-dir "${BAU_SWEEP}" --num-dims 4 --output-subdir figures_marginals_4dim
    run "bau/${ENC} stitched" python -m balls_and_urns.plot_stitched_marginals \
      --sweep-dir "${BAU_SWEEP}" --output-subdir figures_stitched
  else
    echo "SKIP  bau/${ENC}: no sweep directory found"
  fi
done

# ---- Markov matrix marginals (no positional-encoding variants) -------------
# --prompt-length 32 because this sweep raised the prompt length from 8.
MARKOV_SAMPLES=outputs/markov/sweep_analysis/samples
run "markov full" python -m markov.plot_matrix_marginals \
  --samples-dir "${MARKOV_SAMPLES}" --prompt-length 32 \
  --out-dir outputs/markov/matrix_marginals
run "markov K4 (M=4)" python -m markov.plot_matrix_marginals \
  --samples-dir "${MARKOV_SAMPLES}" --prompt-length 32 \
  --max-classes 4 --n-chains 4 --print-frac 0.45 \
  --out-dir outputs/markov/matrix_marginals_K4

echo "----"
echo "${PASS} passed, ${FAIL} failed"
