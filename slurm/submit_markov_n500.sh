#!/bin/bash
# Bring the 1-Markov runs behind Appendix D.4 and D.5 up to N=500, so the
# whole appendix shares one rollout length.
#
#   bash slurm/submit_markov_n500.sh --dry-run   # print, submit nothing
#   bash slurm/submit_markov_n500.sh
#
# D.1-D.3 already use N=500 for all three families. D.4 (latent dimension) and
# D.5 (task diversity) still stop at N=223 for Markov, because no extended
# Markov run exists at those state spaces and task diversities. This submits
# them, with the same rescaling the rest of the appendix uses:
# --seq-len-override 544 --rope-scaling interpolate, i.e. scale 256/544.
#
# Nine jobs: four latent dimensions (k = 3, 4, 6, 8), three task diversities
# (M = 16, 64, 128), and the two aggregates that the appendix figures read.
# M = 32 is not resubmitted: markov_m32_n500_pi544 already has it, and the
# aggregates read that directory.
#
# Each diagnostics job took about 33 minutes on CPU at N=500; they run in
# parallel and the comp partition has no per-user GPU cap to wait on.
#
# WHAT THIS CHANGES IN THE RESULTS. Rescaling positions alters the model at
# every rollout length, not only past 256, so these runs are not the N=223
# runs continued -- they are a different model setting, and their N<=223
# points will not reproduce the existing markov_k*/markov_m*_ctx256 numbers.
# Expect D.5's Markov statements to change: in context, sliced Wasserstein
# reaches the memorizing reference at N=223 for M=16 and 32, and the
# interpolated model's posterior sits far from the memorizing reference at
# every length, so that crossing may disappear. Check before rewriting the
# text, and keep the in-context runs: they are what the main text's Markov
# results rest on.

set -euo pipefail

DRY_RUN=false
for argument in "$@"; do
  case "${argument}" in
    --dry-run) DRY_RUN=true ;;
    *) echo "usage: bash slurm/submit_markov_n500.sh [--dry-run]" >&2; exit 1 ;;
  esac
done

OUT=${OUTPUT_ROOT:-outputs/pmc_diagnostics}
FINAL_STEP=100000
STATE_SPACES=${MARKOV_STATE_SPACES:-"3 4 6 8"}
TASK_DIVERSITIES=${MARKOV_TASK_DIVERSITIES:-"16 64 128"}

# The same rescaling as Appendices D.1-D.3, and the same grids as the other
# N=500 runs in the appendix.
BEYOND=(--seq-len-override 544 --rope-scaling interpolate)
GEOMETRY=(--prompt-length 32 --steps 500
          --length-grid 8 16 32 64 128 223 256 384 500)
MARKOV_ARGS=(--smoothing 1.0 --chunk-size 128)
COMMON=(--num-rollouts 1024 --prompt-sources prior posterior --anchor-length 16
        --count-grid 16 32 64 128 256 512 1024 --n-resamples 50
        --n-projections 100 --temperature 1.0 --seed 42
        --jellyfish-prompts 20 --jellyfish-rollouts 128)

if ! $DRY_RUN && ! command -v sbatch >/dev/null; then
  echo "sbatch not found: run this on a login node, or use --dry-run" >&2
  exit 1
fi

submit() {
  if $DRY_RUN; then
    # stderr: stdout is captured by the caller for the job id.
    echo "  sbatch $*" >&2
    echo "DRYRUN"
  else
    sbatch --parsable "$@"
  fi
}

DIM_IDS=""
DIM_INPUTS=()
TD_IDS=""
TD_INPUTS=()

echo "=== latent dimension, k = ${STATE_SPACES} ==="
for k in ${STATE_SPACES}; do
  id=$(submit --job-name="pmc-markov-k${k}-n500" \
    slurm/markov_rollout_diagnostics.slurm \
    --checkpoint-path "checkpoints/markov_dim_sweep/k${k}/checkpoint_step_${FINAL_STEP}.pt" \
    --config-path "outputs/markov_dim_sweep/k${k}/resolved_config.yaml" \
    "${BEYOND[@]}" "${GEOMETRY[@]}" "${MARKOV_ARGS[@]}" "${COMMON[@]}" \
    --output-dir "${OUT}/markov_k${k}_n500" | tail -1)
  DIM_IDS="${DIM_IDS}:${id}"
  DIM_INPUTS+=("$(( k * (k - 1) ))=${OUT}/markov_k${k}_n500")
done

echo "=== task diversity, M = ${TASK_DIVERSITIES} (M=32 reuses markov_m32_n500_pi544) ==="
for m in ${TASK_DIVERSITIES}; do
  id=$(submit --job-name="pmc-markov-m${m}-n500" \
    slurm/markov_rollout_diagnostics.slurm \
    --checkpoint-path "checkpoints/markov/task_diversity/markov_M${m}/checkpoint_step_${FINAL_STEP}.pt" \
    --config-path "outputs/markov/training/markov_M${m}/resolved_config.yaml" \
    "${BEYOND[@]}" "${GEOMETRY[@]}" "${MARKOV_ARGS[@]}" "${COMMON[@]}" \
    --output-dir "${OUT}/markov_m${m}_n500" | tail -1)
  TD_IDS="${TD_IDS}:${id}"
  TD_INPUTS+=("${m}=${OUT}/markov_m${m}_n500")
done
TD_INPUTS+=("32=${OUT}/markov_m32_n500_pi544")

echo "=== aggregates (each waits on the runs it reads) ==="
submit --job-name=pmc-aggregate-markov-dim-n500 \
  --dependency="afterok${DIM_IDS}" slurm/aggregate_dimension_sweep.slurm \
  --inputs "${DIM_INPUTS[@]}" \
  --output-dir "${OUT}/markov_dimension_sweep_n500" \
  --title "1-Markov (rotary position interpolation, N=500): recovery error vs latent dimension" >/dev/null

submit --job-name=pmc-aggregate-markov-taskdiv-n500 \
  --dependency="afterok${TD_IDS}" slurm/aggregate_dimension_sweep.slurm \
  --x-axis task-diversity --inputs "${TD_INPUTS[@]}" \
  --output-dir "${OUT}/markov_task_diversity_n500" \
  --title "1-Markov (rotary position interpolation, N=500): recovery error vs task diversity" >/dev/null

echo
echo "submitted. outputs land in:"
echo "  ${OUT}/markov_k{3,4,6,8}_n500"
echo "  ${OUT}/markov_m{16,64,128}_n500"
echo "  ${OUT}/markov_dimension_sweep_n500   (Appendix D.4 figure)"
echo "  ${OUT}/markov_task_diversity_n500    (Appendix D.5 figure)"
