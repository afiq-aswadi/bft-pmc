#!/bin/bash
#SBATCH --job-name=bau-diag-v48
#SBATCH --partition=gpu
#SBATCH --gres=gpu:T4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /projects/qh36/<user>/bft-pmc
export PYTHONPATH=.:pfn_transformers
/scratch2/qh36/<user>/conda/envs/pfnbb/bin/python balls_and_urns/rollout_diagnostics.py   --checkpoint-path checkpoints/bau_nope_dim_sweep/vocab48/checkpoint_step_100000.pt   --n-ctx-override 1024   --output-dir outputs/pmc_diagnostics/bau_nope_vocab48
