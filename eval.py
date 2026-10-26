"""Evaluation, diagnostics and figure entry point.

    uv run eval.py --help                 # every command, grouped
    uv run eval.py <command> --help       # one command's options

The command table lives in :mod:`bft_pmc.cli`.
"""

from bft_pmc.cli import eval_main

if __name__ == "__main__":
    eval_main()
