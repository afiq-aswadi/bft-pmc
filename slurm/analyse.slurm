#!/bin/bash
# Checkpoint evaluation for one task family: metrics plus the PMC sample and
# rollout bundles that the path-stability diagnostic reads.
#
#   sbatch --job-name=bft-lr-eval  --array=0-2 --time=12:00:00 \
#          --dependency=afterok:<lr_train_jobid>     --export=ALL,FAMILY=lr     slurm/analyse.slurm
#   sbatch --job-name=bft-bau-eval --array=0-2 --time=08:00:00 \
#          --dependency=afterok:<bau_train_jobid>    --export=ALL,FAMILY=bau    slurm/analyse.slurm
#   sbatch --job-name=bft-mk-eval  --array=0-0 --time=24:00:00 \
#          --dependency=afterok:<markov_train_jobid> --export=ALL,FAMILY=markov slurm/analyse.slurm
#
# LR and BAU run one array task per positional encoding; Markov is a single task.
#
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
##SBATCH --qos=

set -euo pipefail

: "${FAMILY:?set FAMILY=lr|bau|markov, e.g. --export=ALL,FAMILY=lr}"

# module load cuda/12.4
source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

POS_ENCODINGS=(learned rope none)
POS=${POS_ENCODINGS[${SLURM_ARRAY_TASK_ID:-0}]:-}

echo "[$(date +%Y-%m-%dT%H:%M:%S)] host=$(hostname) family=${FAMILY} pos=${POS:-n/a}"

case "${FAMILY}" in
  lr)
    # Prompts are drawn from --seed, so each variant sees the same z_1:n even
    # though it caches into its own directory (no cross-variant write races).
    python eval.py lr-sweep \
      --checkpoint-root "checkpoints/lr/task_diversity_${POS}" \
      --output-dir "outputs/lr/sweep_analysis_${POS}" \
      --eval-dataset-dir "outputs/lr/eval_datasets_${POS}" \
      --prompt-lengths 0 32 \
      --prompt-sources gaussian discrete random \
      --n-prompts 128 --n-samples 100 --n-samples-prior 1024 \
      --predictive-steps 223 --n-projections 100 --seed 42
    ;;
  bau)
    # BAU memorising prompts depend on each run's sampled task pool, so the
    # evaluation dataset is rebuilt from this variant's checkpoints first.
    python scripts/generate_bau_eval_dataset.py \
      --checkpoint-root "checkpoints/bau/task_diversity_${POS}" \
      --output-dir "outputs/bau/eval_datasets_${POS}" \
      --seq-len 32 --batch-size 128 --alpha-value 1.0 --seed 42 --overwrite

    python eval.py bau-sweep \
      --checkpoint-root "checkpoints/bau/task_diversity_${POS}" \
      --eval-dataset-dir "outputs/bau/eval_datasets_${POS}" \
      --output-dir "outputs/bau/sweep_analysis_${POS}" \
      --n-samples 100 --n-samples-prior 1024 \
      --predictive-steps 223 --n-projections 100 --chunk-size 100 --seed 42
    ;;
  markov)
    # Walks every saved checkpoint (one per 200 steps) and runs PMC at each,
    # so this is the longest job and the largest consumer of disk.
    python eval.py markov-sweep \
      --checkpoint-root checkpoints/markov/task_diversity \
      --training-output-root outputs/markov/training \
      --output-dir outputs/markov/sweep_analysis \
      --n-samples 100 --n-samples-prior 1024 --n-prompts 128 \
      --prompt-len 32 --generation-length 223 \
      --n-projections 100 --chunk-size 64 --seed 0
    ;;
  *)
    echo "unknown FAMILY=${FAMILY}; expected lr, bau, or markov" >&2
    exit 1
    ;;
esac

echo "[$(date +%Y-%m-%dT%H:%M:%S)] finished ${FAMILY} ${POS:-}"
