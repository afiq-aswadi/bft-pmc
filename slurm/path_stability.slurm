#!/bin/bash
# Path-stability (jellyfish) diagnostic over every sweep produced above.
# Pure NumPy on saved rollouts: no GPU, no checkpoints, no regeneration.
#
#   sbatch --dependency=afterany:<lr_eval>:<bau_eval>:<markov_eval> slurm/path_stability.slurm
#
#SBATCH --job-name=bft-jellyfish
#SBATCH --cpus-per-task=8
# Sized from measurement, not estimate: job 59365675 peaked at 84.7G of MaxRSS
# over 131 LR bundles (64G had already died in curves_to_frames before the
# tables were made columnar). A full three-encoding sweep carries ~204 LR
# bundles, so this leaves headroom in case the peak tracks bundle count.
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=comp

set -euo pipefail

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

# Each entry is label=path; the label prefixes every setup name in the outputs.
INPUTS=()
for POS in learned rope none; do
  for SWEEP in outputs/lr/sweep_analysis_${POS}/sweep_*/samples; do
    if [[ -d "${SWEEP}" ]]; then INPUTS+=("lr-${POS}=${SWEEP}"); fi
  done
  for SWEEP in outputs/bau/sweep_analysis_${POS}/*/samples; do
    if [[ -d "${SWEEP}" ]]; then INPUTS+=("bau-${POS}=${SWEEP}"); fi
  done
done
if [[ -d outputs/markov/sweep_analysis/samples ]]; then
  INPUTS+=("markov=outputs/markov/sweep_analysis/samples")
fi

if (( ${#INPUTS[@]} == 0 )); then
  echo "no sample directories found; run slurm/analyse.slurm first" >&2
  exit 1
fi

for ENTRY in "${INPUTS[@]}"; do
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] input: ${ENTRY}"
done

# --stride 1 evaluates theta at every prefix, matching the paper. Raise it to
# 5 or 10 if the LR least-squares path becomes the bottleneck.
python eval.py path-stability \
  --inputs "${INPUTS[@]}" \
  --output-dir outputs/path_stability \
  --stride 1

echo "[$(date +%Y-%m-%dT%H:%M:%S)] wrote outputs/path_stability"
