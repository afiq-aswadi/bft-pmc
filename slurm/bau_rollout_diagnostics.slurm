#!/bin/bash
# PMC rollout diagnostics for one balls-and-urns checkpoint: recovery error vs
# rollout length, recovery error vs rollout count, and drift vs length.
#
#   sbatch --job-name=pmc-bau-m32-learned slurm/bau_rollout_diagnostics.slurm \
#     --checkpoint-path checkpoints/bau/task_diversity_learned/M32/checkpoint_step_100000.pt \
#     --n-ctx-override None --steps 239 --length-grid 8 16 32 64 128 239 \
#     --output-dir outputs/pmc_diagnostics/bau_m32_learned
#
# Every argument after the script is passed to `eval.py bau-rollout-diagnostics`.
# slurm/submit_pmc_diagnostics.sh holds the arguments of every run, and the run
# writes them to <output-dir>/run_record.json.
#
#SBATCH --job-name=pmc-bau-diagnostics
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
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
echo "python eval.py bau-rollout-diagnostics $*"
python eval.py bau-rollout-diagnostics "$@"
echo "[$(date +%Y-%m-%dT%H:%M:%S)] finished"
