#!/bin/bash
#SBATCH --job-name=bau-diag-paper-m32
#SBATCH --partition=comp
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /projects/qh36/<user>/bft-pmc
export PYTHONPATH=.:pfn_transformers
/scratch2/qh36/<user>/conda/envs/pfnbb/bin/python balls_and_urns/rollout_diagnostics.py   --checkpoint-path /scratch2/qh36/pfn-consolidated/balls-and-urns/checkpoints/bau_num_tasks_sweep/7gem57bz/checkpoint_step_100000.pt   --n-ctx-override None --steps 239   --length-grid 8 16 32 64 128 239   --output-dir outputs/pmc_diagnostics/bau_paper_m32
