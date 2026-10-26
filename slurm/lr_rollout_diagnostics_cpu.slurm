#!/bin/bash
# PMC rollout diagnostics for one linear-regression checkpoint, on CPU.
#
#   sbatch --job-name=pmc-lr-m32-none-n500-pools \
#     slurm/lr_rollout_diagnostics_cpu.slurm \
#     --checkpoint-path checkpoints/lr/task_diversity_none/M32/checkpoint_step_150000.pt \
#     --prompt-length 32 --steps 500 ... --output-dir outputs/pmc_diagnostics/lr_m32_none_n500_pools
#
# Same entry point and arguments as slurm/lr_rollout_diagnostics.slurm; only the
# device differs. Use this one when the GPU partitions are saturated: on
# 2026-10-21 every A100, H100 and T4 in `gpu`/`m3h` was fully allocated and the
# GPU copy of this job estimated a three-day wait, while `comp` placed an
# equivalent CPU job immediately. The models here are small enough that a T4 is
# no great advantage, and `comp` has no per-user GPU cap to queue behind --
# the same reasoning that makes markov_rollout_diagnostics.slurm CPU-only.
#
# `--device cpu` precedes "$@" so a caller can still override it.
#
# NOT BIT-COMPARABLE WITH THE GPU RUNS. CPU and T4 kernels differ in the last
# digits, so this run's recovery-error-vs-length curve will not byte-match a
# T4 baseline. Compare at tolerance instead; a real regression in the rollout
# path shows up far above float noise (the 2026-10 prompt-seeding bug moved
# the balls-and-urns posterior mean by 50%).
#
#SBATCH --job-name=pmc-lr-diagnostics-cpu
#SBATCH --partition=comp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
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
echo "python eval.py lr-rollout-diagnostics --device cpu $*"
python eval.py lr-rollout-diagnostics --device cpu "$@"
echo "[$(date +%Y-%m-%dT%H:%M:%S)] finished"
