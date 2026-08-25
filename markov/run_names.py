"""Parsing of Markov sweep run-directory names.

Two naming conventions are in use and both must keep working: runs before
2026-08 were written as ``markov_<id>_chains1024_<...>``, and the 2026-08
sweeps write ``markov_M1024``. The figure generators read the task-pool size
out of the directory name, so they need to understand either.
"""

from __future__ import annotations


def parse_n_chains(run_name: str) -> int:
    """Task-pool size encoded in a run directory name.

    Accepts a ``chains<N>`` token (pre-2026-08 sweeps) or an ``M<N>`` token
    (2026-08 onwards). Raises if neither is present.
    """
    for token in run_name.split("_"):
        for prefix in ("chains", "M"):
            if token.startswith(prefix) and token[len(prefix) :].isdigit():
                return int(token[len(prefix) :])
    raise ValueError(f"cannot parse n_chains from run directory {run_name!r}")
