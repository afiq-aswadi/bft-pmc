# Shared settings for the M3 sync scripts. Sourced by them, not run directly.
#
# REMOTE (user@host) and REMOTE_ROOT (the repository checkout on the cluster)
# come from the environment, or else from scripts/m3/remote.env, which git
# ignores. Create it once from the example:
#
#   cp scripts/m3/remote.env.example scripts/m3/remote.env   # then edit it
#
# Defines REPO_ROOT, the local repository root, for the scripts to cd into.

M3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${M3_DIR}/../.." && pwd)"
if [[ -f "${M3_DIR}/remote.env" ]]; then
  # shellcheck source=/dev/null
  source "${M3_DIR}/remote.env"
fi
: "${REMOTE:?set REMOTE=user@host, or create scripts/m3/remote.env from remote.env.example}"
: "${REMOTE_ROOT:?set REMOTE_ROOT to the repository path on the cluster (see scripts/m3/remote.env.example)}"
