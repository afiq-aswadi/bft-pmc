#!/bin/bash
# High-rollout-count prior PMC for the Gaussian-vs-Laplace prior-shape test.
#
#   bash slurm/submit_prior_shape.sh --dry-run   # print, submit nothing
#   bash slurm/submit_prior_shape.sh             # submit one job per task diversity
#
# Why this exists. The rebuttal to Reviewer 9Er4 claims PMC separates a
# zero-mean Gaussian prior from a variance-matched zero-mean Laplace prior,
# where the prediction-space comparison provably cannot (both have mean zero,
# so both give the same prior point prediction). At the camera-ready budget of
# R = 1024 that test has almost no power: bft_pmc/linear_regression/prior_shape_discrimination.py
# shows a *true* Gaussian sample of that size separates the two references by
# only ~2 sigma, because they differ solely in kurtosis and tails. These runs
# raise R so the test can actually resolve the question.
#
# R = 8192 is 8x the camera-ready count, so the margin-to-noise ratio grows by
# about sqrt(8) ~ 2.8, taking the oracle separation from ~2 sigma to ~5-6
# sigma. Going further is cheap in principle but the energy distance is O(R^2)
# in memory: 8192 samples need ~0.5 GB per pairwise matrix, 16384 need ~2 GB.
#
# Everything here stays INSIDE the trained context: an empty prompt and 223
# generated pairs is 446 of the 512 tokens the LR models were trained with, so
# no n_ctx_override and no length extrapolation. The prior needs no prompt,
# which is what makes a large R affordable.
#
# Only the prior is generated (--prompt-sources prior). The in-job recovery and
# count panels are trimmed to the minimum the entrypoint will accept, since the
# analysis is done afterwards by bft_pmc/linear_regression/prior_shape_discrimination.py from the
# saved rollouts:
#   --length-grid 16 223   anchor_length must be a grid entry, hence the 16
#   --count-grid 1024      the count panel is not used here
#   --n-resamples 1        likewise
#
# After the jobs land, pull and analyse on the Mac:
#   bash scripts/m3/pull_pmc_diagnostics.sh
#   python eval.py prior-shape \
#     --run-template 'lr_m{m}_none_prior_r8192' --length 223

set -euo pipefail

DRY_RUN=false
for argument in "$@"; do
  case "${argument}" in
    --dry-run) DRY_RUN=true ;;
    *) echo "usage: bash slurm/submit_prior_shape.sh [--dry-run]" >&2; exit 1 ;;
  esac
done

TASK_DIVERSITIES=${TASK_DIVERSITIES:-"32 128 256"}
ROLLOUTS=${ROLLOUTS:-8192}
FINAL_STEP=150000
OUT=${OUTPUT_ROOT:-outputs/pmc_diagnostics}

if ! $DRY_RUN && ! command -v sbatch >/dev/null; then
  echo "sbatch not found: run this on a login node, or use --dry-run" >&2
  exit 1
fi

for m in ${TASK_DIVERSITIES}; do
  checkpoint="checkpoints/lr/task_diversity_none/M${m}/checkpoint_step_${FINAL_STEP}.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing ${checkpoint}" >&2
    $DRY_RUN || exit 1
  fi
done

for m in ${TASK_DIVERSITIES}; do
  arguments=(
    --checkpoint-path "checkpoints/lr/task_diversity_none/M${m}/checkpoint_step_${FINAL_STEP}.pt"
    --noise-std 0.5
    --prompt-sources prior
    --steps 223
    --length-grid 16 223
    --anchor-length 16
    --num-rollouts "${ROLLOUTS}"
    --count-grid 1024
    --n-resamples 1
    --jellyfish-prompts 1
    --jellyfish-rollouts 1
    --n-projections 100
    --chunk-size 64
    --temperature 1.0
    --seed 42
    --output-dir "${OUT}/lr_m${m}_none_prior_r${ROLLOUTS}"
  )
  if $DRY_RUN; then
    echo "  sbatch --job-name=pmc-lr-m${m}-prior-r${ROLLOUTS} --time=08:00:00 \\"
    echo "    slurm/lr_rollout_diagnostics.slurm ${arguments[*]}"
  else
    sbatch --job-name="pmc-lr-m${m}-prior-r${ROLLOUTS}" --time=08:00:00 \
      slurm/lr_rollout_diagnostics.slurm "${arguments[@]}"
  fi
done
