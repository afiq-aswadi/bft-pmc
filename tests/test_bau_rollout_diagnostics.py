from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bft_pmc.analysis.path_stability import analyse_bundle
from bft_pmc.analysis.pmc_diagnostics import FIGURE_INDEX, RUN_RECORD, file_sha256
from bft_pmc.analysis.replot_pmc_diagnostics import replot_directory
from bft_pmc.balls_and_urns.rollout_diagnostics import DiagnosticsConfig, main

SOURCE_FILES = (
    "trajectories.npz",
    "baselines.npz",
    "recovery_vs_length.csv",
    "drift_vs_length.csv",
    "drift_paths.npz",
    "recovery_vs_count.csv",
    "recovery_vs_count_resamples.csv",
)


def _config(checkpoint: Path, output_dir: Path, **overrides: Any) -> DiagnosticsConfig:
    settings: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "output_dir": str(output_dir),
        "steps": 28,
        "length_grid": [4, 8, 16, 28],
        "n_ctx_override": None,
        # 17 rollouts in chunks of 8 leave a final single-rollout chunk.
        "num_rollouts": 17,
        "prompt_length": 4,
        "anchor_length": 8,
        "count_grid": [4, 8, 16],
        "n_resamples": 3,
        "n_projections": 8,
        "chunk_size": 8,
        "jellyfish_prompts": 3,
        "jellyfish_rollouts": 5,
        "device": "cpu",
    }
    settings.update(overrides)
    return DiagnosticsConfig(**settings)


def test_diagnostics_save_everything_needed_to_redraw_them(
    nope_checkpoint: Path, tmp_path: Path, saved_figures: list[Path]
) -> None:
    output_dir = tmp_path / "bau_nope"
    main(_config(nope_checkpoint, output_dir))

    for source in ("prior", "posterior"):
        for name in SOURCE_FILES:
            assert (output_dir / f"{source}_{name}").exists(), f"{source}_{name}"
    with np.load(output_dir / "posterior_trajectories.npz") as archive:
        assert archive["trajectories"].shape == (17, 4 + 28)
        assert 0 <= int(archive["urn_index"]) < 4
    with np.load(output_dir / "prior_trajectories.npz") as archive:
        assert int(archive["urn_index"]) == -1
    with np.load(output_dir / "posterior_baselines.npz") as archive:
        assert archive["baseline_generalising"].shape == (17, 4)
        assert archive["baseline_memorising"].shape == (17, 4)
        assert archive["task_pool"].shape == (4, 4)
        assert archive["prompt_tokens"].shape == (4,)

    record = json.loads((output_dir / RUN_RECORD).read_text())
    assert record["config"]["steps"] == 28
    assert record["inputs"] == {str(nope_checkpoint): file_sha256(nope_checkpoint)}
    assert record["context"]["device"] == "cpu"
    assert record["context"]["model_context"] == 32
    assert record["context"]["pos_encoding"] == "none"
    assert record["context"]["drift_scale"] == 3

    index = json.loads((output_dir / FIGURE_INDEX).read_text())
    assert sorted(index) == sorted(
        f"{source}_{figure}"
        for source in ("prior", "posterior")
        for figure in ("recovery_vs_length", "drift_vs_length", "recovery_vs_count")
    )
    drawn = list(saved_figures)
    replot_directory(output_dir)
    assert sorted(saved_figures[len(drawn) :]) == sorted(drawn)

    # The jellyfish bundle holds one prompt realisation per curve, each with its
    # own rollouts and urn, and every curve starts at 0.
    with np.load(output_dir / "bau_nope_rollouts.npz") as archive:
        assert archive["rollout_tokens"].shape == (3, 5, 4 + 28)
        assert archive["urn_index"].shape == (3,)
        assert set(archive["urn_index"].tolist()) <= set(range(4))
    curves = analyse_bundle(output_dir / "bau_nope_rollouts.npz", setup="bau-none/run")
    assert curves.family == "bau"
    assert list(curves.grid) == list(range(4, 4 + 28 + 1))
    assert curves.expected[0] == 0.0
    assert curves.per_prompt.shape == (3, 29)
    assert np.allclose(curves.per_prompt[:, 0], 0.0)


def test_diagnostics_stretch_the_context_of_a_nope_model(
    nope_checkpoint: Path, tmp_path: Path, saved_figures: list[Path]
) -> None:
    output_dir = tmp_path / "stretched"
    main(
        _config(
            nope_checkpoint,
            output_dir,
            n_ctx_override=64,
            steps=40,
            length_grid=[4, 8, 16, 40],
        )
    )
    assert (
        json.loads((output_dir / RUN_RECORD).read_text())["context"]["model_context"]
        == 64
    )
    frame = pd.read_csv(output_dir / "posterior_recovery_vs_length.csv")
    assert list(frame["rollout_length"]) == [4, 8, 16, 40]


def test_diagnostics_refuse_to_overrun_the_context_before_any_rollout(
    nope_checkpoint: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "overrun"
    with pytest.raises(ValueError, match=r"needs 33 positions.*--steps 28"):
        main(_config(nope_checkpoint, output_dir, steps=29, length_grid=[4, 8, 29]))
    assert not (output_dir / "prior_trajectories.npz").exists()
    assert not (output_dir / RUN_RECORD).exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"steps": 0}, "must be positive"),
        ({"length_grid": [4, 8, 64]}, "cannot exceed steps"),
        ({"count_grid": [4, 64]}, "cannot exceed num_rollouts"),
        ({"anchor_length": 12}, "anchor_length"),
        ({"jellyfish_prompts": 0}, "jellyfish_prompts"),
        ({"prompt_sources": ["prior", "middle"]}, "Unknown prompt sources"),
    ],
)
def test_config_validation_rejects_inconsistent_settings(
    overrides: dict[str, Any], message: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path / "unused.pt", tmp_path, **overrides).validate()


def test_independent_count_pools_give_every_count_its_own_rollouts(
    nope_checkpoint: Path, tmp_path: Path, saved_figures: list[Path]
) -> None:
    """With independent pools even R = num_rollouts gets n_resamples draws."""
    output_dir = tmp_path / "pools"
    main(
        _config(
            nope_checkpoint,
            output_dir,
            prompt_sources=["posterior"],
            count_grid=[4, 17],
            independent_count_pools=True,
        )
    )
    resamples = pd.read_csv(output_dir / "posterior_recovery_vs_count_resamples.csv")
    model = resamples[resamples["kind"] == "model"]
    draws = model.groupby("num_rollouts")["resample"].nunique().to_dict()
    assert draws == {r: 3 for r in [4, 17]}
    counts = pd.read_csv(output_dir / "posterior_recovery_vs_count.csv")
    endpoint = counts.set_index("num_rollouts").loc[[4, 17][-1]]
    assert endpoint["dist/sw_vs_baseline_generalising/sd"] > 0
    record = json.loads((output_dir / RUN_RECORD).read_text())
    assert record["config"]["independent_count_pools"] is True
