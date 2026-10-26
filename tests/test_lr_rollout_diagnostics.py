from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from bft_pmc.analysis.path_stability import analyse_bundle
from bft_pmc.analysis.pmc_diagnostics import FIGURE_INDEX, RUN_RECORD, file_sha256
from bft_pmc.analysis.replot_pmc_diagnostics import replot_directory
from bft_pmc.linear_regression import train as lr_train
from bft_pmc.linear_regression.rollout_diagnostics import (
    LinearRegressionDiagnosticsConfig,
    main,
    running_least_squares,
)


@pytest.fixture(scope="module")
def lr_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A two-step model over 3-dimensional tasks with room for 32 (x, y) pairs."""
    root = tmp_path_factory.mktemp("lr")
    lr_train.main(
        lr_train.TrainConfig(
            num_tasks=4,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_mlp=32,
            d_head=8,
            d_vocab=16,
            n_ctx=64,
            input_dim=3,
            batch_size=4,
            seq_len=32,
            num_steps=2,
            warmup_steps=0,
            save_every=2,
            checkpoint_root=str(root),
            seed=42,
        )
    )
    return root / "checkpoint_step_2.pt"


def _config(
    checkpoint: Path, output_dir: Path, **overrides: Any
) -> LinearRegressionDiagnosticsConfig:
    settings: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "output_dir": str(output_dir),
        "steps": 24,
        "length_grid": [4, 8, 16, 24],
        "prompt_length": 8,
        "anchor_length": 8,
        "num_rollouts": 16,
        "count_grid": [4, 8, 16],
        "n_resamples": 3,
        "n_projections": 8,
        "chunk_size": 8,
        "jellyfish_prompts": 3,
        "jellyfish_rollouts": 5,
        "device": "cpu",
    }
    settings.update(overrides)
    return LinearRegressionDiagnosticsConfig(**settings)


def test_diagnostics_save_everything_needed_to_redraw_them(
    lr_checkpoint: Path, tmp_path: Path, saved_figures: list[Path]
) -> None:
    output_dir = tmp_path / "lr_m32_learned"
    main(_config(lr_checkpoint, output_dir))

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
    with np.load(output_dir / "posterior_trajectories.npz") as archive:
        assert archive["rollout_x"].shape == (16, 8 + 24, 3)
        assert archive["rollout_y"].shape == (16, 8 + 24)
        np.testing.assert_array_equal(
            archive["rollout_x"][:, :8],
            np.broadcast_to(archive["prompt_x"], (16, 8, 3)),
        )
        assert 0 <= int(archive["task_index"]) < 4
    with np.load(output_dir / "prior_trajectories.npz") as archive:
        assert archive["prompt_x"].shape == (0, 3)
        assert int(archive["task_index"]) == -1
    with np.load(output_dir / "posterior_baselines.npz") as archive:
        assert archive["baseline_generalising"].shape == (16, 3)
        assert archive["baseline_memorising"].shape == (16, 3)
        assert archive["generalising_covariance"].shape == (3, 3)
        assert archive["memorising_weights"].shape == (4,)
    with np.load(output_dir / "posterior_drift_paths.npz") as archive:
        assert archive["estimates_at_grid"].shape == (16, 4, 3)

    assert json.loads((output_dir / RUN_RECORD).read_text())["inputs"] == {
        str(lr_checkpoint): file_sha256(lr_checkpoint)
    }
    context = json.loads((output_dir / RUN_RECORD).read_text())["context"]
    assert context["family"] == "lr"
    assert context["device"] == "cpu"
    assert context["drift_scale"] == 3
    assert context["pair_capacity"] == 32
    assert context["pos_encoding"] == "learned"

    frame = pd.read_csv(output_dir / "posterior_recovery_vs_length.csv")
    assert list(frame["rollout_length"]) == [4, 8, 16, 24]
    assert len(json.loads((output_dir / FIGURE_INDEX).read_text())) == 6
    drawn = list(saved_figures)
    replot_directory(output_dir)
    assert sorted(saved_figures[len(drawn) :]) == sorted(drawn)

    with np.load(output_dir / "lr_m32_learned_rollouts.npz") as archive:
        assert archive["rollout_x"].shape == (3, 5, 8 + 24, 3)
        assert archive["rollout_y"].shape == (3, 5, 8 + 24)
        assert archive["task_index"].shape == (3,)
    curves = analyse_bundle(
        output_dir / "lr_m32_learned_rollouts.npz", setup="lr-learned/run"
    )
    assert curves.family == "lr"
    assert list(curves.grid) == list(range(8, 8 + 24 + 1))
    assert curves.expected[0] == 0.0
    assert curves.theta_reference.shape == (3, 3)
    assert curves.per_prompt.shape == (3, 25)


def test_diagnostics_roll_a_learned_model_past_its_trained_context(
    lr_checkpoint: Path, tmp_path: Path, saved_figures: list[Path]
) -> None:
    """The checkpoint holds 32 (x, y) pairs; a 128-token context fits 8 + 40."""
    output_dir = tmp_path / "lr_m32_learned_n500"
    main(
        _config(
            lr_checkpoint,
            output_dir,
            n_ctx_override=128,
            steps=40,
            length_grid=[4, 8, 16, 24, 40],
        )
    )
    context = json.loads((output_dir / RUN_RECORD).read_text())["context"]
    assert context["model_context"] == 128
    assert context["pair_capacity"] == 64
    curves = analyse_bundle(
        output_dir / "lr_m32_learned_n500_rollouts.npz", setup="lr-learned/run"
    )
    assert list(curves.grid) == list(range(8, 8 + 40 + 1))


def test_diagnostics_refuse_to_overrun_the_pair_capacity_before_any_rollout(
    lr_checkpoint: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "overrun"
    with pytest.raises(ValueError, match=r"needs 33 \(x, y\) pairs.*--steps 24"):
        main(_config(lr_checkpoint, output_dir, steps=25, length_grid=[4, 8, 25]))
    assert not (output_dir / "prior_trajectories.npz").exists()
    assert not (output_dir / RUN_RECORD).exists()


def test_least_squares_returns_the_minimum_norm_fit_for_short_prefixes() -> None:
    """With fewer pairs than weights the fit is underdetermined; lstsq picks the least-norm one."""
    xs = np.zeros((1, 2, 3), dtype=np.float32)
    xs[0, 0, 0] = 1.0
    xs[0, 1, 1] = 1.0
    ys = np.array([[2.0, -1.0]], dtype=np.float32)
    estimates = running_least_squares(xs, ys, prompt_length=0, grid=[2])
    np.testing.assert_allclose(estimates[0, 0], [2.0, -1.0, 0.0], atol=1e-6)


def test_least_squares_recovers_noiseless_weights() -> None:
    generator = torch.Generator().manual_seed(0)
    weights = torch.tensor([0.5, -1.0, 2.0])
    xs = torch.randn(2, 12, 3, generator=generator)
    ys = xs @ weights
    estimates = running_least_squares(
        xs.numpy(), ys.numpy(), prompt_length=4, grid=[4, 8]
    )
    np.testing.assert_allclose(
        estimates, np.broadcast_to(weights.numpy(), (2, 2, 3)), atol=1e-5
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"steps": 0}, "must be positive"),
        ({"length_grid": [4, 8, 64]}, "cannot exceed steps"),
        ({"count_grid": [4, 64]}, "cannot exceed num_rollouts"),
        ({"anchor_length": 12}, "anchor_length"),
        ({"noise_std": 0.0}, "noise_std"),
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
    lr_checkpoint: Path, tmp_path: Path, saved_figures: list[Path]
) -> None:
    """With independent pools even R = num_rollouts gets n_resamples draws."""
    output_dir = tmp_path / "pools"
    main(
        _config(
            lr_checkpoint,
            output_dir,
            prompt_sources=["posterior"],
            count_grid=[4, 16],
            independent_count_pools=True,
        )
    )
    resamples = pd.read_csv(output_dir / "posterior_recovery_vs_count_resamples.csv")
    model = resamples[resamples["kind"] == "model"]
    draws = model.groupby("num_rollouts")["resample"].nunique().to_dict()
    assert draws == {r: 3 for r in [4, 16]}
    counts = pd.read_csv(output_dir / "posterior_recovery_vs_count.csv")
    endpoint = counts.set_index("num_rollouts").loc[[4, 16][-1]]
    assert endpoint["dist/sw_vs_baseline_generalising/sd"] > 0
    record = json.loads((output_dir / RUN_RECORD).read_text())
    assert record["config"]["independent_count_pools"] is True
