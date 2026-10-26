"""Training entry point for every task family and for YAML sweeps.

    uv run train.py --help                # every setting
    uv run train.py <setting> --help      # one setting's options

The command table lives in :mod:`bft_pmc.cli`.
"""

from bft_pmc.cli import train_main

if __name__ == "__main__":
    train_main()
