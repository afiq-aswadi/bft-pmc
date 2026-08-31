#!/bin/bash
#SBATCH --job-name=markov-diag-m32
#SBATCH --partition=comp
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=<email>
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
cd /projects/qh36/<user>/bft-pmc
export PYTHONPATH=.:pfn_transformers
/scratch2/qh36/<user>/conda/envs/pfnbb/bin/python markov/rollout_diagnostics.py   --checkpoint-path /scratch2/qh36/<user>/markov_m32/checkpoint_step_100000.pt   --output-dir outputs/pmc_diagnostics/markov_m32
