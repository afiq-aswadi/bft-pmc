#!/bin/bash
# PMC rollout diagnostics for one 1-Markov checkpoint: recovery error vs rollout
# length, recovery error vs rollout count, and drift vs length.
#
#   sbatch --job-name=pmc-markov-m32-ctx256 slurm/markov_rollout_diagnostics.slurm \
#     --checkpoint-path checkpoints/markov/task_diversity/markov_M32/checkpoint_step_100000.pt \
#     --config-path outputs/markov/training/markov_M32/resolved_config.yaml \
#     --steps 248 --length-grid 8 16 32 64 128 192 248 \
#     --output-dir outputs/pmc_diagnostics/markov_m32_ctx256
#
# Every argument after the script is passed to `eval.py markov-rollout-diagnostics`.
# slurm/submit_pmc_diagnostics.sh holds the arguments of every run, and the run
# writes them to <output-dir>/run_record.json. CPU only: rollouts of these small
# models do not need a GPU, and the CPU partition has no per-user GPU cap.
#
# --config-path must be the run's own resolved_config.yaml. markov/train.yaml
# pins k=10, and without a config path the loader looks for the training chain
# pool beside the checkpoint, where the trainer does not write it.
#
#SBATCH --job-name=pmc-markov-diagnostics
#SBATCH --partition=comp
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if (( $# == 0 )); then
  echo "pass the diagnostics arguments after the script; see slurm/submit_pmc_diagnostics.sh" >&2
  exit 1
fi

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

echo "[$(date +%Y-%m-%dT%H:%M:%S)] host=$(hostname)"
echo "python eval.py markov-rollout-diagnostics $*"
python eval.py markov-rollout-diagnostics "$@"
echo "[$(date +%Y-%m-%dT%H:%M:%S)] finished"
