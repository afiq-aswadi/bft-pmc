#!/bin/bash
#SBATCH --job-name=bau-nope-v12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:T4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=<email>
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
cd /projects/qh36/<user>/bft-pmc
export PYTHONPATH=.:pfn_transformers
/scratch2/qh36/<user>/conda/envs/pfnbb/bin/python balls_and_urns/train.py   --num-tasks 32 --vocab-size 12 --alpha-value 1.0   --d-model 64 --d-mlp 256 --n-layers 2 --n-heads 1 --d-head 64   --n-ctx 256 --act-fn gelu --pos-encoding none   --batch-size 64 --seq-len 256 --num-steps 100000   --learning-rate 5e-4 --warmup-steps 0 --seed 42   --checkpoint-schedule logarithmic --linear-checkpoint-interval 500   --n-log-checkpoints 200   --checkpoint-root checkpoints/bau_nope_dim_sweep/vocab12
