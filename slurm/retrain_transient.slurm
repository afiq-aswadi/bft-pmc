#!/bin/bash
# Retrain the transient-generalization models with the checkpoint schedule the
# paper documents.
#
#   sbatch --array=0-5 slurm/retrain_transient.slurm
#
# Why this exists. Appendices F and G both state:
#
#   "We save checkpoints on the union of 200 logarithmically-spaced steps and a
#    linear schedule every 500 steps."
#
# but slurm/train.slurm passed --checkpoint-schedule linear, so the earliest
# checkpoint written was step 500. Figures 9 (LR) and 12 (BAU) sweep all saved
# checkpoints at M = 32, and the balls-and-urns transient is finished by step
# ~91 -- its memorising and generalising curves cross at steps 6, 38, 51, 68
# and 91, then separate permanently. No evaluation of the existing checkpoints
# can recover it.
#
# Scope is two models per encoding, not two sweeps: the transient experiment
# fixes a single task diversity (paper section: "we fix an intermediate task
# diversity M ... and sweep all saved training checkpoints"), M = 32 for both
# families.
#
# Every hyperparameter below is copied verbatim from slurm/train.slurm, which
# matches the appendices. Only --checkpoint-schedule changes.
#
#SBATCH --job-name=bft-transient
# Ask for the L40S explicitly. A bare --gres=gpu:1 can land on m3t100's T4,
# which measured ~8x slower here: 100 min per checkpoint against ~12 on the
# L40S, turning a 12-hour job into an 87-hour one that cannot fit the
# walltime. Override with --gres at submit time if L40S nodes are busy.
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu

set -euo pipefail

source .venv/bin/activate

export WANDB_MODE=offline
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

INDEX=${SLURM_ARRAY_TASK_ID:-0}
FAMILIES=(lr bau)
POS_ENCODINGS=(learned rope none)
FAMILY=${FAMILIES[$((INDEX / 3))]}
POS=${POS_ENCODINGS[$((INDEX % 3))]}

# The transient figures are both at intermediate task diversity M = 32.
M=32
RUN_ROOT="checkpoints/${FAMILY}/transient_td${M}_${POS}"
LOG_ROOT="${RUN_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

echo "[$(date +%Y-%m-%dT%H:%M:%S)] transient retrain: ${FAMILY} M=${M} pos=${POS}"
echo "[$(date +%Y-%m-%dT%H:%M:%S)] checkpoints -> ${RUN_ROOT}"

case "${FAMILY}" in
  lr)
    python train.py lr \
      --num-tasks "${M}" \
      --pos-encoding "${POS}" \
      --d-model 512 --n-layers 2 --n-heads 4 --d-mlp 512 --d-head 128 \
      --d-vocab 256 --n-ctx 512 --act-fn gelu --input-dim 8 \
      --mask-type autoregressive-pfn --prediction-type distribution \
      --bucket-type uniform --bucket-support unbounded --y-min -10.0 --y-max 10.0 \
      --prior-type discrete --function-type linear --noise-std 0.5 \
      --batch-size 256 --seq-len 256 --num-steps 150000 \
      --learning-rate 1e-4 --warmup-steps 1000 \
      --save-checkpoint --checkpoint-schedule logarithmic \
      --n-log-checkpoints 200 --linear-checkpoint-interval 500 \
      --checkpoint-root "${RUN_ROOT}" \
      --eval-every 1000 --eval-batches 10 --seed 42 --no-use-wandb \
      --log-file "${LOG_ROOT}/num_tasks_${M}.json"
    ;;
  bau)
    python train.py bau \
      --num-tasks "${M}" \
      --pos-encoding "${POS}" \
      --vocab-size 12 --alpha-value 1.0 \
      --d-model 64 --d-mlp 256 --n-layers 2 --n-heads 1 --d-head 64 \
      --n-ctx 256 --act-fn gelu \
      --batch-size 64 --seq-len 256 --num-steps 100000 \
      --learning-rate 5e-4 --warmup-steps 0 \
      --save-checkpoint --checkpoint-schedule logarithmic \
      --n-log-checkpoints 200 --linear-checkpoint-interval 500 \
      --checkpoint-root "${RUN_ROOT}" \
      --eval-every 1000 --eval-batches 10 --seed 42 --no-use-wandb \
      --log-file "${LOG_ROOT}/num_tasks_${M}.json"
    ;;
  *)
    echo "unknown family ${FAMILY}" >&2
    exit 1
    ;;
esac

echo "[$(date +%Y-%m-%dT%H:%M:%S)] done; checkpoints written:"
find "${RUN_ROOT}" -name "*.pt" | wc -l
