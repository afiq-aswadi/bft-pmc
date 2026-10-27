#!/bin/bash
# Gaussian-vs-Laplace prior-shape analysis of the high-R prior rollouts.
#
#   sbatch slurm/prior_shape.slurm
#
# Reads only the saved rollouts from slurm/submit_prior_shape.sh, so it needs
# no checkpoint and no GPU. Runs the analysis twice:
#
#   raw       the PMC samples as recovered. Both references are unit-variance
#             by construction, so a sample whose variance differs is penalised
#             for scale as well as shape.
#   matched   the same samples rescaled to unit variance, which isolates the
#             shape question the rebuttal actually claims.
#
# Read the oracle line first in each: it is a true Gaussian sample of the same
# size, so it bounds what any method could resolve at this budget. Only if the
# oracle separates cleanly do the PMC margins below it mean anything.
#
# Energy distance is a U-statistic over all pairs, hence --max-ed-samples;
# sliced Wasserstein uses the full set. Both variants together take a few
# minutes and stay under ~2 GB.
#
#SBATCH --job-name=prior-shape
#SBATCH --partition=comp
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

COMMON=(
  --run-template 'lr_m{m}_none_prior_r8192'
  --length 223
  --task-diversities 32 128 256 512 1024 4096 16384 65536
  --max-ed-samples 4096
  --n-blocks 32
  --seed 42
)

echo "[$(date +%Y-%m-%dT%H:%M:%S)] host=$(hostname)"

echo
echo "=================== raw (as recovered) ==================="
python eval.py prior-shape "${COMMON[@]}" \
  --output-dir outputs/prior_shape/raw

echo
echo "============ variance-matched (shape only) ==============="
python eval.py prior-shape "${COMMON[@]}" \
  --match-variance --output-dir outputs/prior_shape/matched

echo
echo "[$(date +%Y-%m-%dT%H:%M:%S)] finished"
