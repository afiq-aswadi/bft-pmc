#!/bin/bash
# Jellyfish plot (Ng et al., TabMGP, Figure 3) from rollout-diagnostics runs.
#
#   sbatch --job-name=pmc-jellyfish-m32 slurm/pmc_jellyfish.slurm \
#     --inputs lr-learned=outputs/pmc_diagnostics/lr_m32_learned \
#              bau-learned=outputs/pmc_diagnostics/bau_m32_learned \
#     --output-dir outputs/pmc_diagnostics/jellyfish_m32 --stride 1
#
# Every argument after the script is passed to `eval.py path-stability`, which
# reads each run's <run>_rollouts.npz. For every generated step it averages the
# scaled L1 distance between the estimate on the prompt and the estimate on the
# prompt plus the rollout over all 1024 rollouts, one trajectory per run. It
# draws one panel per family and one figure per positional encoding. Labels are
# <family>-<encoding>, or markov. CPU only.
#
#SBATCH --job-name=pmc-jellyfish
#SBATCH --partition=comp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if (( $# == 0 )); then
  echo "pass the path-stability arguments after the script; see slurm/submit_pmc_diagnostics.sh" >&2
  exit 1
fi

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

echo "python eval.py path-stability $*"
python eval.py path-stability "$@"
