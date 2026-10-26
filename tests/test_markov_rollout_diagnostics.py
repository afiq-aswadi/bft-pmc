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
from bft_pmc.markov.rollout_diagnostics import (
    MarkovDiagnosticsConfig,
    check_rollouts_fit_context,
    main,
)


def test_rollouts_may_fill_the_context_exactly() -> None:
    check_rollouts_fit_context(
        steps=248,
        prompt_length=8,
        prompt_sources=["prior", "posterior"],
        context_length=256,
    )


def test_posterior_overrun_is_caught_before_the_prior_runs() -> None:
    # 250 steps fit the empty-prompt prior but not the 8-state posterior.
    with pytest.raises(ValueError, match=r"needs 258 positions.*--steps 248"):
        check_rollouts_fit_context(
            steps=250,
            prompt_length=8,
            prompt_sources=["prior", "posterior"],
            context_length=256,
        )


def test_prior_only_runs_ignore_the_prompt_length() -> None:
    check_rollouts_fit_context(
        steps=256,
        prompt_length=8,
        prompt_sources=["prior"],
        context_length=256,
    )
    with pytest.raises(ValueError, match="--steps 256"):
        check_rollouts_fit_context(
            steps=500,
            prompt_length=8,
            prompt_sources=["prior"],
            context_length=256,
        )


def _config(
    run_dir: Path, output_dir: Path, **overrides: Any
) -> MarkovDiagnosticsConfig:
    settings: dict[str, Any] = {
        "checkpoint_path": str(run_dir / "model.pt"),
        "config_path": str(run_dir / "resolved_config.yaml"),
        "output_dir": str(output_dir),
        "steps": 6,
        "length_grid": [2, 4, 6],
        "num_rollouts": 9,
        "prompt_length": 2,
        "anchor_length": 4,
        "count_grid": [3, 9],
        "n_resamples": 2,
        "n_projections": 4,
        "chunk_size": 4,
        "jellyfish_prompts": 3,
        "jellyfish_rollouts": 5,
    }
    settings.update(overrides)
    return MarkovDiagnosticsConfig(**settings)


def test_diagnostics_save_everything_needed_to_redraw_them(
    tiny_run: Path, tmp_path: Path, saved_figures: list[Path]
) -> None:
    output_dir = tmp_path / "markov"
    main(_config(tiny_run, output_dir))

    for source in ("prior", "posterior"):
        for name in (
            "trajectories.npz",
            "baselines.npz",
            "recovery_vs_length.csv",
            "drift_vs_length.csv",
            "drift_paths.npz",
            "recovery_vs_count.csv",
            "recovery_vs_count_resamples.csv",
        ):
            assert (output_dir / f"{source}_{name}").exists(), f"{source}_{name}"
    with np.load(output_dir / "posterior_baselines.npz") as archive:
        assert archive["baseline_generalising"].shape == (9, 4)
        assert archive["task_pool"].shape == (2, 2, 2)
        assert archive["prompt_tokens"].shape == (2,)
    with np.load(output_dir / "posterior_drift_paths.npz") as archive:
        assert archive["estimates_at_grid"].shape == (9, 3, 4)

    record = json.loads((output_dir / RUN_RECORD).read_text())
    read = (
        "model.pt",
        "resolved_config.yaml",
        "transition_matrices.npy",
        "stationary_distributions.npy",
    )
    assert record["inputs"] == {
        str(tiny_run / name): file_sha256(tiny_run / name) for name in read
    }
    context = json.loads((output_dir / RUN_RECORD).read_text())["context"]
    assert context["k"] == 2
    assert context["drift_scale"] == 2
    assert context["model_context"] == 8
    assert context["task_distribution_source"].startswith("npy:")

    assert len(json.loads((output_dir / FIGURE_INDEX).read_text())) == 6
    drawn = list(saved_figures)
    replot_directory(output_dir)
    assert sorted(saved_figures[len(drawn) :]) == sorted(drawn)

    with np.load(output_dir / "markov_rollouts.npz") as archive:
        assert archive["rollout_states"].shape == (3, 5, 2 + 6)
    curves = analyse_bundle(output_dir / "markov_rollouts.npz", setup="markov/run")
    assert curves.family == "markov"
    assert list(curves.grid) == list(range(2, 2 + 6 + 1))
    assert curves.expected[0] == 0.0
    assert curves.per_prompt.shape == (3, 7)


def test_diagnostics_refuse_to_overrun_the_context_before_any_rollout(
    tiny_run: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "overrun"
    with pytest.raises(ValueError, match=r"needs 9 positions.*--steps 6"):
        main(_config(tiny_run, output_dir, steps=7, length_grid=[2, 4, 7]))
    assert not (output_dir / "prior_trajectories.npz").exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_rollouts": 0}, "must be positive"),
        ({"length_grid": [2, 4, 12]}, "cannot exceed steps"),
        ({"count_grid": [3, 12]}, "cannot exceed num_rollouts"),
        ({"anchor_length": 3}, "anchor_length"),
        ({"jellyfish_prompts": 0}, "jellyfish_prompts"),
        ({"prompt_sources": ["posterior", "middle"]}, "Unknown prompt sources"),
        ({"rope_scaling": "stretch"}, "Unknown rope_scaling"),
    ],
)
def test_config_validation_rejects_inconsistent_settings(
    overrides: dict[str, Any], message: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, tmp_path, **overrides).validate()


def test_independent_count_pools_give_every_count_its_own_rollouts(
    tiny_run: Path, tmp_path: Path, saved_figures: list[Path]
) -> None:
    """With independent pools even R = num_rollouts gets n_resamples draws."""
    output_dir = tmp_path / "pools"
    main(
        _config(
            tiny_run,
            output_dir,
            prompt_sources=["posterior"],
            count_grid=[3, 9],
            independent_count_pools=True,
        )
    )
    resamples = pd.read_csv(output_dir / "posterior_recovery_vs_count_resamples.csv")
    model = resamples[resamples["kind"] == "model"]
    draws = model.groupby("num_rollouts")["resample"].nunique().to_dict()
    assert draws == {r: 2 for r in [3, 9]}
    counts = pd.read_csv(output_dir / "posterior_recovery_vs_count.csv")
    endpoint = counts.set_index("num_rollouts").loc[[3, 9][-1]]
    assert endpoint["dist/sw_vs_baseline_generalising/sd"] > 0
    record = json.loads((output_dir / RUN_RECORD).read_text())
    assert record["config"]["independent_count_pools"] is True
