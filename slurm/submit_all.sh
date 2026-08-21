#!/bin/bash
# Submit the whole pipeline with dependencies wired: three training arrays,
# three evaluation arrays that wait on them, and the diagnostic that waits on
# all three evaluations.
#
#   bash slurm/submit_all.sh --dry-run    # print the sbatch commands, submit nothing
#   bash slurm/submit_all.sh              # submit
#
# Override walltimes or the concurrency cap without editing anything:
#   LR_TIME=36:00:00 THROTTLE=4 bash slurm/submit_all.sh
#
# Run from the repository root, on a login node.

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
elif [[ -n "${1:-}" ]]; then
  echo "usage: bash slurm/submit_all.sh [--dry-run]" >&2
  exit 1
fi

LR_TIME=${LR_TIME:-24:00:00}
BAU_TIME=${BAU_TIME:-12:00:00}
MARKOV_TIME=${MARKOV_TIME:-12:00:00}
LR_EVAL_TIME=${LR_EVAL_TIME:-12:00:00}
BAU_EVAL_TIME=${BAU_EVAL_TIME:-08:00:00}
MARKOV_EVAL_TIME=${MARKOV_EVAL_TIME:-24:00:00}
THROTTLE=${THROTTLE:-8}

if ! $DRY_RUN && ! command -v sbatch >/dev/null; then
  echo "sbatch not found: run this on a login node, or use --dry-run" >&2
  exit 1
fi

# submit() runs inside command substitution, i.e. a subshell, so the dry-run
# counter lives in a file rather than a variable.
FAKE_JOBID_FILE=$(mktemp)
trap 'rm -f "${FAKE_JOBID_FILE}"' EXIT
echo 1000 > "${FAKE_JOBID_FILE}"

submit() {
  if $DRY_RUN; then
    echo "  sbatch $*" >&2
    local next=$(( $(cat "${FAKE_JOBID_FILE}") + 1 ))
    echo "${next}" > "${FAKE_JOBID_FILE}"
    echo "${next}"
  else
    sbatch --parsable "$@"
  fi
}

mkdir -p logs

echo "=== training ==="
LR=$(submit --job-name=bft-lr --array="0-50%${THROTTLE}" --time="${LR_TIME}" \
  --export=ALL,FAMILY=lr slurm/train.slurm)
BAU=$(submit --job-name=bft-bau --array="0-38%${THROTTLE}" --time="${BAU_TIME}" \
  --export=ALL,FAMILY=bau slurm/train.slurm)
MARKOV=$(submit --job-name=bft-markov --array="0-9%${THROTTLE}" --time="${MARKOV_TIME}" \
  --export=ALL,FAMILY=markov slurm/train.slurm)

echo "=== evaluation (waits on training) ==="
LR_EVAL=$(submit --job-name=bft-lr-eval --array=0-2 --time="${LR_EVAL_TIME}" \
  --dependency="afterok:${LR}" --export=ALL,FAMILY=lr slurm/analyse.slurm)
BAU_EVAL=$(submit --job-name=bft-bau-eval --array=0-2 --time="${BAU_EVAL_TIME}" \
  --dependency="afterok:${BAU}" --export=ALL,FAMILY=bau slurm/analyse.slurm)
MARKOV_EVAL=$(submit --job-name=bft-mk-eval --array=0-0 --time="${MARKOV_EVAL_TIME}" \
  --dependency="afterok:${MARKOV}" --export=ALL,FAMILY=markov slurm/analyse.slurm)

echo "=== diagnostic (waits on all evaluations) ==="
# afterany, not afterok: the diagnostic runs on whatever sample bundles exist,
# so one failed evaluation should not sink the whole pipeline.
JELLYFISH=$(submit --dependency="afterany:${LR_EVAL}:${BAU_EVAL}:${MARKOV_EVAL}" \
  slurm/path_stability.slurm)

cat <<SUMMARY

job ids
  lr train        ${LR}       (51 tasks)
  bau train       ${BAU}       (39 tasks)
  markov train    ${MARKOV}       (10 tasks)
  lr eval         ${LR_EVAL}       (3 tasks, after ${LR})
  bau eval        ${BAU_EVAL}       (3 tasks, after ${BAU})
  markov eval     ${MARKOV_EVAL}       (1 task,  after ${MARKOV})
  jellyfish       ${JELLYFISH}       (after the three evals)

watch:   squeue -u \$USER
cancel:  scancel ${LR} ${BAU} ${MARKOV} ${LR_EVAL} ${BAU_EVAL} ${MARKOV_EVAL} ${JELLYFISH}
SUMMARY

if $DRY_RUN; then
  echo
  echo "dry run: nothing was submitted, job ids above are placeholders"
fi
