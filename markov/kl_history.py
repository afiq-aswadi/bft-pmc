"""Symmetrised-KL histories for the Markov figures.

Two sources exist, and both must keep working.

Runs before 2026-08 were plotted from ``wandb_kl_history.csv``, a manual
Weights & Biases export sitting beside each run's sweep metrics, with columns
``kl/{id,ood}/{spec}/{role}``.

The 2026-08 sweeps trained with ``--no-use-wandb``, so no such export exists.
But ``markov/train.py`` writes the same quantities unconditionally to
``training_log.csv`` under the training output root, as
``kl_{spec}_{role}_{split}``. Both files carry symmetrised KL from
``metrics.symmetrised_kl``; only the column naming and the location differ.

This module normalises either source to the ``kl/{id,ood}/wellspec/{role}``
column names the figures use.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TRAINING_ROOT = Path("outputs/markov/training")

# The figures plot only the wellspec (order-1 bigram) baselines, which match the
# Markov data-generating process. misspec columns are recorded but unused here.
_ROLES = ("memorising", "generalising")
_SPLITS = ("id", "ood")


def _from_training_log(path: Path) -> pd.DataFrame:
    """Rename training_log.csv KL columns to the figure's naming."""
    frame = pd.read_csv(path)
    renamed = {"step": frame["step"]}
    for split in _SPLITS:
        for role in _ROLES:
            renamed[f"kl/{split}/wellspec/{role}"] = frame[
                f"kl_wellspec_{role}_{split}"
            ]
    return pd.DataFrame(renamed).sort_values("step")


def load_kl_history(
    run_dir: Path,
    run_name: str | None = None,
    training_root: Path | None = None,
) -> pd.DataFrame | None:
    """Per-step wellspec KL for one run, or None when neither source exists.

    The W&B export is preferred when present so that previously published
    figures reproduce byte-for-byte; the training log is the fallback and is
    the only source for the 2026-08 sweeps.
    """
    wandb_path = run_dir / "wandb_kl_history.csv"
    if wandb_path.is_file():
        return pd.read_csv(wandb_path).sort_values("step")

    root = TRAINING_ROOT if training_root is None else training_root
    training_path = root / (run_name or run_dir.name) / "training_log.csv"
    if training_path.is_file():
        return _from_training_log(training_path)
    return None
