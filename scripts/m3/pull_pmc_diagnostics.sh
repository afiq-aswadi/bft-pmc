#!/bin/bash
# Pull the PMC rollout diagnostics from M3 to this machine, data included.
#
#   bash scripts/m3/pull_pmc_diagnostics.sh                     # outputs
#   bash scripts/m3/pull_pmc_diagnostics.sh --with-checkpoints  # and their inputs
#   bash scripts/m3/pull_pmc_diagnostics.sh --dry-run           # list without transferring
#
# Run this on your workstation, not on M3.
#
# Unlike pull_figures.sh, this fetches everything under
# outputs/pmc_diagnostics/: the trajectories, baseline samples, curves,
# resamples, run records, figure indexes, jellyfish curves and the submission
# records (code snapshot, environment, commands, job logs) alongside the
# figures. With it every figure can be redrawn locally:
#
#   uv run eval.py replot-pmc-diagnostics --roots outputs/pmc_diagnostics
#
# --with-checkpoints also fetches every input file a pulled run record names
# (checkpoints, and for Markov the config and chain pool, at the same paths),
# which with the submission's code snapshot is what a rerun needs away from M3.

set -euo pipefail

source "$(dirname "$0")/remote.sh"

DRY=""
WITH_CHECKPOINTS=false
for argument in "$@"; do
  case "${argument}" in
    --dry-run) DRY="--dry-run" ;;
    --with-checkpoints) WITH_CHECKPOINTS=true ;;
    *)
      echo "usage: bash scripts/m3/pull_pmc_diagnostics.sh [--dry-run] [--with-checkpoints]" >&2
      exit 1
      ;;
  esac
done
if [[ -n "${DRY}" ]]; then
  echo "(dry run)"
fi

cd "${REPO_ROOT}"
mkdir -p outputs/pmc_diagnostics

rsync -av ${DRY} "${REMOTE}:${REMOTE_ROOT}/outputs/pmc_diagnostics/" outputs/pmc_diagnostics/

if $WITH_CHECKPOINTS; then
  INPUT_LIST=$(mktemp)
  trap 'rm -f "${INPUT_LIST}"' EXIT
  python3 - > "${INPUT_LIST}" <<'PYTHON'
import json
from pathlib import Path

paths = set()
for record in Path("outputs/pmc_diagnostics").rglob("run_record.json"):
    for path in json.loads(record.read_text()).get("inputs", {}):
        if not path.startswith(("/", "outputs/pmc_diagnostics")):
            paths.add(path)
print("\n".join(sorted(paths)))
PYTHON
  echo
  echo "input files named by the run records: $(grep -c . "${INPUT_LIST}" || true)"
  if [[ -s "${INPUT_LIST}" ]]; then
    rsync -av ${DRY} --files-from="${INPUT_LIST}" "${REMOTE}:${REMOTE_ROOT}/" ./
  fi
fi

echo
echo "run directories now under outputs/pmc_diagnostics/:"
find outputs/pmc_diagnostics -name run_record.json | wc -l
