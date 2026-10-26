#!/bin/bash
# Train one balls-and-urns model for the latent-dimension sweep.
#
#   sbatch --job-name=pmc-train-bau-vocab6 --export=ALL,VOCAB=6 slurm/train_bau_vocab.slurm
#
# Normally submitted by slurm/submit_pmc_diagnostics.sh. The command is the
# camera-ready balls-and-urns training in slurm/train.slurm for the
# task_diversity_none/M32 run (M=32, no positional encoding), with only the
# vocabulary changed, so the latent dimension is VOCAB-1. Vocabulary 12 is that
# camera-ready run itself and is not trained again.
#
#SBATCH --job-name=pmc-train-bau-vocab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${VOCAB:?set VOCAB=6|24|48, e.g. --export=ALL,VOCAB=6}"

source .venv/bin/activate
export WANDB_MODE=offline
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

SWEEP_ROOT="checkpoints/bau/vocab_sweep_none"
RUN_ROOT="${SWEEP_ROOT}/vocab${VOCAB}"
mkdir -p "${RUN_ROOT}" "${SWEEP_ROOT}/logs"
echo "[$(date +%Y-%m-%dT%H:%M:%S)] host=$(hostname) vocab=${VOCAB} -> ${RUN_ROOT}"
nvidia-smi -L || true

python train.py bau \
  --num-tasks 32 \
  --pos-encoding none \
  --vocab-size "${VOCAB}" --alpha-value 1.0 \
  --d-model 64 --d-mlp 256 --n-layers 2 --n-heads 1 --d-head 64 \
  --n-ctx 256 --act-fn gelu \
  --batch-size 64 --seq-len 256 --num-steps 100000 \
  --learning-rate 5e-4 --warmup-steps 0 \
  --save-checkpoint --checkpoint-schedule linear --save-every 500 \
  --checkpoint-root "${RUN_ROOT}" \
  --eval-every 1000 --eval-batches 10 --seed 42 --no-use-wandb \
  --log-file "${SWEEP_ROOT}/logs/vocab_${VOCAB}.json"

echo "[$(date +%Y-%m-%dT%H:%M:%S)] finished vocab ${VOCAB}"
