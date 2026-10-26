#!/bin/bash
# Next-token fidelity of the camera-ready BAU and 1-Markov models, scored to
# position 500.
#
#   sbatch slurm/predictive_fidelity.slurm
#
# Asks whether the trained model's next-token distribution *is* the Bayes
# predictive, position by position, in prediction space. Independent of PMC:
# no rollouts, one forward pass per sequence, so the whole job is minutes.
#
# Both families are scored past their trained context, which is the point of
# running to 500 rather than 255:
#   BAU     no positional encoding, so --n-ctx-override 1024 reuses the
#           trained weights unchanged.
#   Markov  RoPE, so position 500 is reachable only by rescaling positions
#           into the trained range (--seq-len-override 544, scale 256/544).
#           Its curve therefore describes the rescaled model, exactly as in
#           Appendix D.1--D.3, and a rise past position 256 cannot be read as
#           the model ceasing to be Bayesian.
#
# Linear regression is absent by design: its head is categorical over
# discretised response buckets and no exact predictive exists in that space.
#
#SBATCH --job-name=predictive-fidelity
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

FINAL_STEP=100000
OUT=outputs/predictive_fidelity

echo "[$(date +%Y-%m-%dT%H:%M:%S)] host=$(hostname)"

echo
echo "=== alignment self-test (no checkpoint) ==="
python eval.py predictive-fidelity --self-test

echo
echo "=== balls-and-urns, M=32, no positional encoding ==="
python eval.py predictive-fidelity --family bau \
  --checkpoint-paths "checkpoints/bau/task_diversity_none/M32/checkpoint_step_${FINAL_STEP}.pt" \
  --seq-len 500 --n-ctx-override 1024 --n-sequences 256 --batch-size 64 \
  --seed 42 --output-dir "${OUT}"

echo
echo "=== 1-Markov, M=32, rotary position interpolation 256/544 ==="
python eval.py predictive-fidelity --family markov \
  --checkpoint-paths "checkpoints/markov/task_diversity/markov_M32/checkpoint_step_${FINAL_STEP}.pt" \
  --config-path outputs/markov/training/markov_M32/resolved_config.yaml \
  --seq-len 500 --seq-len-override 544 --rope-scaling interpolate \
  --n-sequences 256 --batch-size 64 --seed 42 --output-dir "${OUT}"

echo
echo "[$(date +%Y-%m-%dT%H:%M:%S)] finished"
