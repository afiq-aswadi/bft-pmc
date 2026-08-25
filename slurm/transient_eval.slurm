#!/bin/bash
# Evaluate every saved checkpoint of the M=32 runs, producing the data behind
# Figures 9 (linear regression) and 12 (balls-and-urns), then draw them.
#
#   sbatch --array=0-5 slurm/transient_eval.slurm      # both families
#   sbatch --array=3-5 slurm/transient_eval.slurm      # balls-and-urns only
#
# Index maps the same way as slurm/retrain_transient.slurm: 0-2 are lr x
# {learned, rope, none}, 3-5 are bau x the same.
#
# This walks checkpoints, so it is far heavier than the task-diversity sweep,
# which evaluates one checkpoint per run: several hundred full PMC evaluations
# rather than one. Override the stride with SUBSAMPLE=<n> in --export if the
# job needs to be shorter or the curve denser.
#
# Checkpoint root: prefers the dense-early retrain from
# slurm/retrain_transient.slurm when it exists, and otherwise falls back to the
# task-diversity sweep. The fallback is fine for LR, whose transient crossover
# is at step ~9000 and well covered by every-500 checkpoints. It is NOT fine
# for BAU, whose transient is over by step ~91 -- that run must be retrained
# first or the figure shows nothing.
#
#SBATCH --job-name=bft-transient-eval
# Ask for the L40S explicitly. A bare --gres=gpu:1 can land on m3t100's T4,
# which measured ~8x slower here: 100 min per checkpoint against ~12 on the
# L40S, turning a 12-hour job into an 87-hour one that cannot fit the
# walltime. Override with --gres at submit time if L40S nodes are busy.
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu

set -euo pipefail

source .venv/bin/activate
export PYTHONPATH=.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
# The chunk-32 OOM reported 12.45 GiB reserved-but-unallocated: fragmentation,
# not live tensors. Expandable segments let the allocator reuse those blocks
# instead of holding them against differently-sized requests.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INDEX=${SLURM_ARRAY_TASK_ID:-0}
FAMILIES=(lr bau)
POS_ENCODINGS=(learned rope none)
FAMILY=${FAMILIES[$((INDEX / 3))]}
POS=${POS_ENCODINGS[$((INDEX % 3))]}

M=32
RUN_ID="M${M}"

# Evaluate every Nth checkpoint. Each one is a full PMC evaluation at the
# sweep settings (prompt length 32, rollouts to the context length), so this
# is what keeps the job inside its walltime. It costs x-axis resolution
# only -- roughly 60 points across five decades at 5.
SUBSAMPLE="${SUBSAMPLE:-5}"

# How many prompts go through the model at once. 4 was an OOM workaround for a
# 14.5 GiB T4 and serialises the 128 prompts into 32 sequential passes, which
# dominated wall time: ~20 min per checkpoint with the GPU pinned at 100%.
# 32 was measured to OOM on the L40S -- 42.9 GiB of its 44.4 GiB usable, with a
# further 4 GiB requested -- so an eighth of the batch is the working size.
# Still 4x fewer sequential passes than the original 4.
PROMPT_CHUNK="${PROMPT_CHUNK:-16}"

# The two sources nest differently. The sweep passed a shared --checkpoint-root
# and the trainer made one M<num_tasks> directory per run inside it; the
# retrain passes a per-run root and the trainer writes checkpoints straight
# into it. The dynamics modules join root and run-id, so both cases are
# expressed by choosing where the split falls.
RETRAIN_ROOT="checkpoints/${FAMILY}"
RETRAIN_RUN="transient_td${M}_${POS}"
if [[ -n "$(find "${RETRAIN_ROOT}/${RETRAIN_RUN}" -maxdepth 1 -name '*.pt' 2>/dev/null | head -1)" ]]; then
  CKPT_ROOT="${RETRAIN_ROOT}"
  RUN_ID="${RETRAIN_RUN}"
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] using dense-early retrain: ${CKPT_ROOT}/${RUN_ID}"
else
  CKPT_ROOT="checkpoints/${FAMILY}/task_diversity_${POS}"
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] retrain absent; falling back to ${CKPT_ROOT}/${RUN_ID}"
  if [[ "${FAMILY}" == "bau" ]]; then
    echo "WARNING: the BAU transient ends by step ~91 and this root's earliest" >&2
    echo "         checkpoint is step 500. Figure 12 will show nothing." >&2
  fi
fi

OUT_DIR="outputs/${FAMILY}/transient_${POS}"
echo "[$(date +%Y-%m-%dT%H:%M:%S)] ${FAMILY}/${POS}: $(find "${CKPT_ROOT}/${RUN_ID}" -name '*.pt' | wc -l) checkpoints -> ${OUT_DIR}"

case "${FAMILY}" in
  lr)
    python -m linear_regression.distribution_dynamics \
      --run-id "${RUN_ID}" \
      --checkpoint-root "${CKPT_ROOT}" \
      --prompt-length 32 --n-prompts 128 --n-samples 100 \
      --predictive-steps 223 --n-projections 100 \
      --chunk-size 100 --prompt-chunk-size "${PROMPT_CHUNK}" \
      --checkpoint-subsample "${SUBSAMPLE}" \
      --compute-distribution --compute-delta \
      --seed 42 --output-dir "${OUT_DIR}"
    ;;
  bau)
    python -m balls_and_urns.distribution_dynamics \
      --run-id "${RUN_ID}" \
      --checkpoint-root "${CKPT_ROOT}" \
      --eval-dataset-dir "outputs/bau/eval_datasets_${POS}" \
      --compute-distribution --compute-delta --compute-prior-mode \
      --n-samples 100 --predictive-steps 223 --n-projections 100 \
      --chunk-size 100 --alpha-value 1.0 \
      --checkpoint-subsample "${SUBSAMPLE}" \
      --seed 42 --output-dir "${OUT_DIR}"
    ;;
esac

# No plotting step here on purpose. Both distribution_dynamics modules draw the
# figure themselves once the loop finishes, and both nest their output under
# <output-dir>/<run-id>. An external plotter pointed at <output-dir>/metrics.csv
# fails after the evaluation has already succeeded -- two hours of work reported
# as a failed job.
echo "[$(date +%Y-%m-%dT%H:%M:%S)] done"
find "${OUT_DIR}" -maxdepth 2 -type f \( -name '*.png' -o -name 'metrics.csv' \) | sort
