#!/bin/bash
# Train one 1-Markov model for the latent-dimension sweep.
#
#   sbatch --job-name=pmc-train-markov-k8 --export=ALL,K=8 slurm/train_markov_dim.slurm
#
# Normally submitted by slurm/submit_pmc_diagnostics.sh. The command is the
# camera-ready Markov training in slurm/train.slurm for the markov_M32 run
# (markov/train.yaml, n_chains 32, seq_len 256), with only the state space k
# changed. The latent task is the transition matrix: k(k-1) free parameters.
#
# Training goes through the root train.py. Running markov/train.py by path puts
# markov/ first on sys.path, where markov/plotting.py shadows the plotting
# package, and the job dies at import.
#
#SBATCH --job-name=pmc-train-markov-dim
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${K:?set K=3|4|6|8, e.g. --export=ALL,K=8}"

source .venv/bin/activate
export WANDB_MODE=offline
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

mkdir -p checkpoints/markov_dim_sweep outputs/markov_dim_sweep
echo "[$(date +%Y-%m-%dT%H:%M:%S)] host=$(hostname) k=${K}"
nvidia-smi -L || true

python train.py markov \
  --config-path markov/train.yaml \
  --k "${K}" \
  --n-chains 32 \
  --run-name "k${K}" \
  --checkpoint-root checkpoints/markov_dim_sweep \
  --output-root outputs/markov_dim_sweep \
  --pmc-num-samples 100 --pmc-prompt-len 32 --pmc-generation-length 223 \
  --no-use-wandb

echo "[$(date +%Y-%m-%dT%H:%M:%S)] finished k=${K}"
