#!/bin/bash
# Re-run the three M=32 diagnostics with an 8192-rollout pool, so that
# Appendix D.3's error bars mean what a reader assumes they mean.
#
#   bash slurm/submit_count_r8192.sh --dry-run
#   bash slurm/submit_count_r8192.sh
#
# WHY. D.3 draws 50 subsets of R rollouts from a saved pool and reports their
# spread. With a pool of 1024 and a count grid ending at 1024, the endpoint
# has exactly one possible subset -- the pool itself -- so the figure carries
# no error bar at the budget every other figure in the paper uses. Below the
# endpoint the subsets overlap increasingly, so the reported standard
# deviation is understated by roughly sqrt(1 - R/1024): about 71% of its
# honest size at R=512.
#
# Independent replicates would fix this exactly, but need
# 50 x (16+32+...+1024) = 101,600 rollouts per source, about 99x what is
# generated now. Enlarging the pool to 8192 costs 8x and achieves most of it:
# the correction at R=1024 becomes sqrt(1 - 1024/8192) = 0.94, and at R=512
# 0.96. The count grid is unchanged, so the same points are plotted.
#
# SCOPE. Only the three runs D.1-D.3 actually plot. The dimension and
# task-diversity sweeps keep the 1024 pool: D.3 is the only subsection whose
# claim rests on the spread.
#
# RIPPLE. These runs also back the rollout-length figures, so D.2's curves and
# every number quoted in D.2 and D.3 will change -- 8x the samples gives a
# different, better estimate at each length. D.1 is unaffected: the
# path-stability bundle is built from its own 20 x 128 rollouts, independent
# of this pool. Outputs land under new names so the current results survive
# for comparison; nothing is switched over until the numbers are checked.

set -euo pipefail

DRY_RUN=false
for argument in "$@"; do
  case "${argument}" in
    --dry-run) DRY_RUN=true ;;
    *) echo "usage: bash slurm/submit_count_r8192.sh [--dry-run]" >&2; exit 1 ;;
  esac
done

OUT=${OUTPUT_ROOT:-outputs/pmc_diagnostics}
POOL=${POOL:-8192}
LR_FINAL_STEP=150000
FINAL_STEP=100000

# Unchanged from the current runs, so the figures plot the same points.
GEOMETRY=(--prompt-length 32 --steps 500
          --length-grid 8 16 32 64 128 223 256 384 500)
COUNTS=(--count-grid 16 32 64 128 256 512 1024 --n-resamples 50)
COMMON=(--prompt-sources prior posterior --anchor-length 16
        --n-projections 100 --temperature 1.0 --seed 42
        --jellyfish-prompts 20 --jellyfish-rollouts 128)

if ! $DRY_RUN && ! command -v sbatch >/dev/null; then
  echo "sbatch not found: run this on a login node, or use --dry-run" >&2
  exit 1
fi

submit() {
  if $DRY_RUN; then echo "  sbatch $*" >&2; else sbatch "$@"; fi
}

echo "=== linear regression, M=32, no positional encoding ==="
submit --job-name="pmc-lr-m32-none-r${POOL}" --time=12:00:00 \
  slurm/lr_rollout_diagnostics.slurm \
  --checkpoint-path "checkpoints/lr/task_diversity_none/M32/checkpoint_step_${LR_FINAL_STEP}.pt" \
  --noise-std 0.5 --n-ctx-override 1064 --chunk-size 64 \
  --num-rollouts "${POOL}" "${GEOMETRY[@]}" "${COUNTS[@]}" "${COMMON[@]}" \
  --output-dir "${OUT}/lr_m32_none_r${POOL}"

echo "=== balls-and-urns, M=32, no positional encoding ==="
submit --job-name="pmc-bau-m32-none-r${POOL}" --time=12:00:00 \
  slurm/bau_rollout_diagnostics.slurm \
  --checkpoint-path "checkpoints/bau/task_diversity_none/M32/checkpoint_step_${FINAL_STEP}.pt" \
  --alpha-value 1.0 --chunk-size 256 --n-ctx-override 1024 \
  --num-rollouts "${POOL}" "${GEOMETRY[@]}" "${COUNTS[@]}" "${COMMON[@]}" \
  --output-dir "${OUT}/bau_m32_none_r${POOL}"

echo "=== 1-Markov, M=32, rotary position interpolation 256/544 ==="
submit --job-name="pmc-markov-m32-r${POOL}" --time=12:00:00 \
  slurm/markov_rollout_diagnostics.slurm \
  --checkpoint-path "checkpoints/markov/task_diversity/markov_M32/checkpoint_step_${FINAL_STEP}.pt" \
  --config-path outputs/markov/training/markov_M32/resolved_config.yaml \
  --seq-len-override 544 --rope-scaling interpolate --smoothing 1.0 --chunk-size 128 \
  --num-rollouts "${POOL}" "${GEOMETRY[@]}" "${COUNTS[@]}" "${COMMON[@]}" \
  --output-dir "${OUT}/markov_m32_r${POOL}"

echo
echo "outputs: ${OUT}/{lr_m32_none,bau_m32_none,markov_m32}_r${POOL}"
echo "the existing 1024-pool runs are untouched; compare before switching the figures."
