#!/bin/bash
# Push every uncommitted source change to M3.
#
#   bash scripts/m3/sync_code.sh            # send
#   bash scripts/m3/sync_code.sh --dry-run  # list what would be sent
#
# Run this on your workstation, not on M3.
#
# The file list comes from `git status`, so it cannot go stale the way a
# hand-written rsync argument list does. Partial syncs are how M3 ended up
# running a new plot_dynamics_combined.py against an old paper_style.py, which
# fails at import with a confusing error.
#
# Only source files are sent. Outputs and figures are gitignored and stay where
# they are generated.

set -euo pipefail

source "$(dirname "$0")/remote.sh"

cd "${REPO_ROOT}"

# Added, modified, renamed (at their new path) and untracked files, excluding
# deletions: a deleted file has nothing to send, and rsync would error on the
# missing path. Deletions and renames are not propagated, so after a commit
# that moves files, `git pull` on M3 rather than syncing.
# A read loop rather than mapfile: macOS ships bash 3.2, which lacks it.
FILES=()
while IFS= read -r path; do
  FILES+=("${path}")
done < <(
  {
    git diff --name-only --diff-filter=ACMR HEAD
    git ls-files --others --exclude-standard
  } |
    grep -E '\.(py|sh|slurm|md|yaml|yml|toml|lock)$' |
    sort -u
)

if (( ${#FILES[@]} == 0 )); then
  echo "nothing to sync: working tree is clean"
  exit 0
fi

echo "${#FILES[@]} file(s) to sync to ${REMOTE}:${REMOTE_ROOT}"
printf '  %s\n' "${FILES[@]}"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo
  echo "(dry run; nothing sent)"
  exit 0
fi

echo
rsync -avR "${FILES[@]}" "${REMOTE}:${REMOTE_ROOT}/"
