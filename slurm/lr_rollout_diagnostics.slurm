#!/bin/bash
# PMC rollout diagnostics for one linear-regression checkpoint: recovery error vs
# rollout length, recovery error vs rollout count, and drift vs length, plus the
# posterior rollouts the jellyfish plot reads.
#
#   sbatch --job-name=pmc-lr-m32-learned slurm/lr_rollout_diagnostics.slurm \
#     --checkpoint-path checkpoints/lr/task_diversity_learned/M32/checkpoint_step_150000.pt \
#     --prompt-length 32 --steps 223 --length-grid 8 16 32 64 128 223 \
#     --output-dir outputs/pmc_diagnostics/lr_m32_learned
#
# Every argument after the script is passed to `eval.py lr-rollout-diagnostics`.
# slurm/submit_pmc_diagnostics.sh holds the arguments of every run, and the run
# writes them to <output-dir>/run_record.json.
#
#SBATCH --job-name=pmc-lr-diagnostics
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if (( $# == 0 )); then
  echo "pass the diagnostics arguments after the script; see slurm/submit_pmc_diagnostics.sh" >&2
  exit 1
fi

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

echo "[$(date +%Y-%m-%dT%H:%M:%S)] host=$(hostname)"
nvidia-smi -L || true
echo "python eval.py lr-rollout-diagnostics $*"
python eval.py lr-rollout-diagnostics "$@"
echo "[$(date +%Y-%m-%dT%H:%M:%S)] finished"
