#!/bin/bash
# Aggregate per-run rollout diagnostics into recovery error vs latent dimension,
# or, with --x-axis task-diversity, vs the task diversity M.
#
#   sbatch --job-name=pmc-aggregate-markov-dim slurm/aggregate_dimension_sweep.slurm \
#     --inputs 6=outputs/pmc_diagnostics/markov_k3 12=outputs/pmc_diagnostics/markov_k4 \
#     --output-dir outputs/pmc_diagnostics/markov_dimension_sweep
#   sbatch --job-name=pmc-aggregate-markov-task-diversity slurm/aggregate_dimension_sweep.slurm \
#     --x-axis task-diversity \
#     --inputs 32=outputs/pmc_diagnostics/markov_m32_ctx256 64=outputs/pmc_diagnostics/markov_m64_ctx256 \
#     --output-dir outputs/pmc_diagnostics/markov_task_diversity
#
# Every argument after the script is passed to `eval.py recovery-vs-dimension`;
# slurm/submit_pmc_diagnostics.sh submits one per sweep once its runs finish.
#
#SBATCH --job-name=pmc-aggregate-dimension
#SBATCH --partition=comp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if (( $# == 0 )); then
  echo "pass the aggregation arguments after the script; see slurm/submit_pmc_diagnostics.sh" >&2
  exit 1
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "python eval.py recovery-vs-dimension $*"
python eval.py recovery-vs-dimension "$@"
