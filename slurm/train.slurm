#!/bin/bash
# Training sweep for one task family, as a SLURM job array.
#
# The array size and walltime differ per family, so both are supplied at submit
# time. Run these from the repository root, in this order or in parallel:
#
#   sbatch --job-name=bft-lr     --array=0-50%8 --time=24:00:00 \
#          --export=ALL,FAMILY=lr     slurm/train.slurm
#   sbatch --job-name=bft-bau    --array=0-38%8 --time=12:00:00 \
#          --export=ALL,FAMILY=bau    slurm/train.slurm
#   sbatch --job-name=bft-markov --array=0-9%5  --time=12:00:00 \
#          --export=ALL,FAMILY=markov slurm/train.slurm
#
# LR and BAU cross every task count with all three positional encodings
# (17x3 = 51 and 13x3 = 39 runs); Markov trains one run per chain-pool size.
#
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
# M3's `gpu` partition (A100/A40/T4) needs no QoS. Only uncomment --qos if you
# switch to a partition that requires one, e.g. --partition=m3h --qos=m3h for
# H100s, or --partition=fit --qos=fitq if `mon_qos` shows you have FIT access.
##SBATCH --qos=

set -euo pipefail

: "${FAMILY:?set FAMILY=lr|bau|markov, e.g. --export=ALL,FAMILY=lr}"

# module load cuda/12.4                  # only if your cluster uses modules
source .venv/bin/activate                # created once on the login node by `uv sync`

export WANDB_MODE=offline                # compute nodes usually have no outbound network
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

INDEX=${SLURM_ARRAY_TASK_ID}
POS_ENCODINGS=(learned rope none)

echo "[$(date +%Y-%m-%dT%H:%M:%S)] host=$(hostname) family=${FAMILY} array_index=${INDEX}"
nvidia-smi -L || true

case "${FAMILY}" in
  lr|bau)
    if [[ "${FAMILY}" == "lr" ]]; then
      NUM_TASKS=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536)
    else
      NUM_TASKS=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
    fi
    TOTAL=$(( ${#NUM_TASKS[@]} * ${#POS_ENCODINGS[@]} ))
    if (( INDEX >= TOTAL )); then
      echo "array index ${INDEX} exceeds ${TOTAL} ${FAMILY} runs" >&2
      exit 1
    fi
    M=${NUM_TASKS[$(( INDEX % ${#NUM_TASKS[@]} ))]}
    POS=${POS_ENCODINGS[$(( INDEX / ${#NUM_TASKS[@]} ))]}

    # One directory per run. Without W&B the trainer writes straight into
    # --checkpoint-root, and the sweep analysis treats each subdirectory of
    # task_diversity_<pos> as a separate run, so they must not share a root.
    SWEEP_ROOT="checkpoints/${FAMILY}/task_diversity_${POS}"
    RUN_ROOT="${SWEEP_ROOT}/M${M}"
    mkdir -p "${RUN_ROOT}" "${SWEEP_ROOT}/logs"
    echo "[$(date +%Y-%m-%dT%H:%M:%S)] M=${M} pos=${POS} -> ${RUN_ROOT}"
    ;;
  markov)
    N_CHAINS_VALUES=(4 8 16 32 64 128 256 512 1024 2048)
    if (( INDEX >= ${#N_CHAINS_VALUES[@]} )); then
      echo "array index ${INDEX} exceeds ${#N_CHAINS_VALUES[@]} markov runs" >&2
      exit 1
    fi
    N_CHAINS=${N_CHAINS_VALUES[${INDEX}]}
    mkdir -p checkpoints/markov/task_diversity outputs/markov/training
    echo "[$(date +%Y-%m-%dT%H:%M:%S)] n_chains=${N_CHAINS}"
    ;;
  *)
    echo "unknown FAMILY=${FAMILY}; expected lr, bau, or markov" >&2
    exit 1
    ;;
esac

case "${FAMILY}" in
  lr)
    python train.py lr \
      --num-tasks "${M}" \
      --pos-encoding "${POS}" \
      --d-model 512 --n-layers 2 --n-heads 4 --d-mlp 512 --d-head 128 \
      --d-vocab 256 --n-ctx 512 --act-fn gelu --input-dim 8 \
      --mask-type autoregressive-pfn --prediction-type distribution \
      --bucket-type uniform --bucket-support unbounded --y-min -10.0 --y-max 10.0 \
      --prior-type discrete --function-type linear --noise-std 0.5 \
      --batch-size 256 --seq-len 256 --num-steps 150000 \
      --learning-rate 1e-4 --warmup-steps 1000 \
      --save-checkpoint --checkpoint-schedule linear --save-every 500 \
      --checkpoint-root "${RUN_ROOT}" \
      --eval-every 1000 --eval-batches 10 --seed 42 --no-use-wandb \
      --log-file "${SWEEP_ROOT}/logs/num_tasks_${M}.json"
    ;;
  bau)
    python train.py bau \
      --num-tasks "${M}" \
      --pos-encoding "${POS}" \
      --vocab-size 12 --alpha-value 1.0 \
      --d-model 64 --d-mlp 256 --n-layers 2 --n-heads 1 --d-head 64 \
      --n-ctx 256 --act-fn gelu \
      --batch-size 64 --seq-len 256 --num-steps 100000 \
      --learning-rate 5e-4 --warmup-steps 0 \
      --save-checkpoint --checkpoint-schedule linear --save-every 500 \
      --checkpoint-root "${RUN_ROOT}" \
      --eval-every 1000 --eval-batches 10 --seed 42 --no-use-wandb \
      --log-file "${SWEEP_ROOT}/logs/num_tasks_${M}.json"
    ;;
  markov)
    python train.py markov \
      --config-path markov/train.yaml \
      --n-chains "${N_CHAINS}" \
      --run-name "markov_M${N_CHAINS}" \
      --checkpoint-root checkpoints/markov/task_diversity \
      --output-root outputs/markov/training \
      --pmc-num-samples 100 --pmc-prompt-len 32 --pmc-generation-length 223 \
      --no-use-wandb
    ;;
esac

echo "[$(date +%Y-%m-%dT%H:%M:%S)] finished ${FAMILY} array index ${INDEX}"
