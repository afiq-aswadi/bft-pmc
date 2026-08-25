#!/bin/bash
# Pull every figure from M3 to this machine, leaving the data behind.
#
#   bash scripts/pull_figures_from_m3.sh            # fetch
#   bash scripts/pull_figures_from_m3.sh --dry-run  # list without transferring
#
# Run this on the Mac.
#
# Only .png and .pdf are fetched. The sample bundles, rollout traces and
# per-prompt CSVs that sit in the same directories run to tens of gigabytes and
# stay on M3; `outputs/path_stability/*.csv` is the exception worth pulling by
# hand, since the jellyfish figures replot from it in seconds.

set -euo pipefail

REMOTE="${REMOTE:-haotongm@m3.massive.org.au}"
REMOTE_ROOT="${REMOTE_ROOT:-/projects/qh36/bft-pmc}"

cd "$(dirname "$0")/.."
mkdir -p outputs

DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY="--dry-run"
  echo "(dry run)"
fi

rsync -av ${DRY} \
  --include='*/' \
  --include='*.png' \
  --include='*.pdf' \
  --exclude='*' \
  --prune-empty-dirs \
  "${REMOTE}:${REMOTE_ROOT}/outputs/" outputs/

echo
echo "figures now under outputs/:"
find outputs -type f \( -name '*.png' -o -name '*.pdf' \) | wc -l
