from __future__ import annotations

from pathlib import Path

import pandas as pd

from markov.kl_history import load_kl_history


def _write_training_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "step": [100, 1],
            "n_chains": [8, 8],
            "train_loss": [0.4, 0.9],
            "id_kl": [0.1, 0.2],
            "ood_kl": [0.3, 0.4],
            "kl_wellspec_generalising_id": [0.11, 0.21],
            "kl_wellspec_memorising_id": [0.12, 0.22],
            "kl_misspec_generalising_id": [0.13, 0.23],
            "kl_misspec_memorising_id": [0.14, 0.24],
            "kl_wellspec_generalising_ood": [0.15, 0.25],
            "kl_wellspec_memorising_ood": [0.16, 0.26],
            "kl_misspec_generalising_ood": [0.17, 0.27],
            "kl_misspec_memorising_ood": [0.18, 0.28],
        }
    )
    frame.to_csv(path, index=False)


def test_training_log_is_renamed_and_sorted(tmp_path: Path) -> None:
    """The 2026-08 sweeps have only training_log.csv, under a separate root."""
    _write_training_log(tmp_path / "training" / "markov_M8" / "training_log.csv")
    run_dir = tmp_path / "runs" / "markov_M8"
    run_dir.mkdir(parents=True)

    history = load_kl_history(run_dir, training_root=tmp_path / "training")
    assert history is not None
    assert list(history["step"]) == [1, 100]
    assert list(history.columns) == [
        "step",
        "kl/id/wellspec/memorising",
        "kl/id/wellspec/generalising",
        "kl/ood/wellspec/memorising",
        "kl/ood/wellspec/generalising",
    ]
    # renaming maps role and split correctly, not just by position
    assert history.loc[history["step"] == 100, "kl/ood/wellspec/memorising"].item() == (
        0.16
    )


def test_wandb_export_wins_so_published_figures_reproduce(tmp_path: Path) -> None:
    _write_training_log(tmp_path / "training" / "markov_M8" / "training_log.csv")
    run_dir = tmp_path / "runs" / "markov_M8"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "step": [1, 2],
            "kl/id/wellspec/memorising": [9.0, 8.0],
            "kl/id/wellspec/generalising": [7.0, 6.0],
            "kl/ood/wellspec/memorising": [5.0, 4.0],
            "kl/ood/wellspec/generalising": [3.0, 2.0],
        }
    ).to_csv(run_dir / "wandb_kl_history.csv", index=False)

    history = load_kl_history(run_dir, training_root=tmp_path / "training")
    assert history is not None
    assert history["kl/id/wellspec/memorising"].iloc[0] == 9.0


def test_missing_from_both_sources_returns_none(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "markov_M8"
    run_dir.mkdir(parents=True)
    assert load_kl_history(run_dir, training_root=tmp_path / "training") is None
    # the run name may differ from the directory name
    _write_training_log(tmp_path / "training" / "other" / "training_log.csv")
    assert load_kl_history(run_dir, "other", tmp_path / "training") is not None
