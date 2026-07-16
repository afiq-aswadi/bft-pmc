from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import linear_regression.distribution_dynamics as dynamics


class DynamicsModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


@pytest.mark.parametrize(
    "config",
    [
        dynamics.DynamicsConfig(),
        dynamics.DynamicsConfig(run_id="run", prompt_length=-1),
        dynamics.DynamicsConfig(run_id="run", n_samples=0),
        dynamics.DynamicsConfig(run_id="run", n_prompts=0),
        dynamics.DynamicsConfig(run_id="run", predictive_steps=0),
        dynamics.DynamicsConfig(run_id="run", n_projections=0),
        dynamics.DynamicsConfig(run_id="run", eval_batch_size=0),
        dynamics.DynamicsConfig(run_id="run", eval_seq_len=0),
    ],
)
def test_dynamics_config_validation(config: dynamics.DynamicsConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_dynamics_config_mapping_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    config = dynamics.DynamicsConfig(
        run_id="run",
        prompt_length=2,
        n_samples=3,
        n_prompts=4,
        noise_std=0.25,
        device="cpu",
    )
    mapped = dynamics._to_sweep_config(config)
    assert mapped.prompt_lengths == (2,)
    assert mapped.n_samples == (3,)
    assert mapped.n_prompts == (4,)
    assert config.noise_variance == pytest.approx(0.0625)
    assert dynamics._resolve_device(config).type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert dynamics._resolve_device(replace(config, device=None)).type == "cpu"


def test_dynamics_run_analysis_posterior_prior_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = dynamics.DynamicsConfig(run_id="missing", checkpoint_root=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Run directory"):
        dynamics.run_analysis(missing)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    no_checkpoint = dynamics.DynamicsConfig(run_id="run", checkpoint_root=str(tmp_path))
    with pytest.raises(RuntimeError, match="No checkpoints"):
        dynamics.run_analysis(no_checkpoint)

    checkpoints = [run_dir / "checkpoint_step_1.pt", run_dir / "checkpoint_step_2.pt"]
    for checkpoint in checkpoints:
        checkpoint.touch()
    model = DynamicsModel()
    tasks = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    run_info = {
        "model": model,
        "task_size": 2,
        "num_tasks": 2,
        "tasks": tasks,
    }
    monkeypatch.setattr(dynamics, "SupervisedPFN", nn.Module)
    monkeypatch.setattr(dynamics, "load_run_info", lambda path, device: run_info)
    monkeypatch.setattr(
        dynamics,
        "load_or_create_shared_eval_context",
        lambda config, run_dirs: object(),
    )
    monkeypatch.setattr(
        dynamics,
        "build_run_eval_inputs",
        lambda *args: ({}, object()),
    )
    monkeypatch.setattr(
        dynamics,
        "compute_all_predictive_metrics",
        lambda *args, **kwargs: {
            "data_memorising/model_mse": 1.0,
            "data_memorising/delta_vs_baseline_memorising": 0.2,
            "data_generalising/delta_vs_baseline_generalising": 0.3,
            "data_random/delta_vs_baseline_memorising": 0.4,
        },
    )
    monkeypatch.setattr(
        dynamics,
        "prepare_model_for_long_rollout",
        lambda *args, **kwargs: 3,
    )

    def fake_distribution(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, float], dict[str, np.ndarray], list[dict[str, float]]]:
        del args, kwargs
        metrics = {
            "dist/ed_vs_baseline_memorising": 0.1,
            "dist/ed_vs_baseline_generalising": 0.2,
        }
        return metrics, {"pt": np.zeros((2, 2))}, [metrics]

    monkeypatch.setattr(
        dynamics, "compute_distribution_metrics_single", fake_distribution
    )
    posterior_config = dynamics.DynamicsConfig(
        run_id="run",
        checkpoint_root=str(tmp_path),
        prompt_length=2,
        n_samples=2,
        n_prompts=1,
        predictive_steps=3,
        n_projections=2,
        device="cpu",
    )
    metrics, num_tasks, per_prompt = dynamics.run_analysis(
        posterior_config,
        samples_dir=tmp_path / "samples",
    )
    assert num_tasks == 2
    assert len(metrics) == 2
    assert len(per_prompt) == 4
    assert "ed_vs_baseline_memorising_from_prompts_memorising" in metrics
    assert "delta_vs_baseline_memorising_on_data_memorising" in metrics
    assert "delta_vs_baseline_generalising_on_data_generalising" in metrics
    assert "delta_vs_baseline_memorising_on_random" in metrics
    assert "model_mse_on_data_memorising" not in metrics
    assert not any("/" in column for column in metrics.columns)
    assert len(list((tmp_path / "samples").glob("*.npz"))) == 4

    prior_metrics, _, prior_per_prompt = dynamics.run_analysis(
        replace(posterior_config, prompt_length=0, compute_delta=False),
    )
    assert "ed_vs_baseline_memorising" in prior_metrics
    assert set(prior_per_prompt["source"]) == {"N/A"}

    empty_metrics, _, empty_per_prompt = dynamics.run_analysis(
        replace(
            posterior_config,
            compute_delta=False,
            compute_distribution=False,
        )
    )
    assert list(empty_metrics.columns) == ["step"]
    assert empty_per_prompt.empty


def test_dynamics_main_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = pd.DataFrame({"step": [1], "metric": [0.1]})
    per_prompt = pd.DataFrame({"step": [1], "prompt_idx": [0]})
    monkeypatch.setattr(
        dynamics,
        "run_analysis",
        lambda config, samples_dir: (metrics, 2, per_prompt),
    )
    plotted: list[dynamics.PlotConfig] = []
    monkeypatch.setattr(dynamics, "plot_dynamics", plotted.append)
    config = dynamics.DynamicsConfig(run_id="run", output_dir=str(tmp_path))
    dynamics.main(config)
    output_dir = tmp_path / "run"
    assert (output_dir / "config.json").exists()
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "per_prompt_metrics.csv").exists()
    assert len(plotted) == 1

    monkeypatch.setattr(
        dynamics,
        "run_analysis",
        lambda config, samples_dir: (metrics, 2, pd.DataFrame()),
    )
    dynamics.main(replace(config, run_id="empty"))
    assert not (tmp_path / "empty/per_prompt_metrics.csv").exists()
    assert len(plotted) == 2

    dynamics.main(replace(config, run_id="delta-only", compute_distribution=False))
    dynamics.main(replace(config, run_id="distribution-only", compute_delta=False))
    assert len(plotted) == 2

    dynamics.main(
        replace(
            config,
            run_id="prior-only",
            prompt_length=0,
            compute_delta=False,
        )
    )
    assert len(plotted) == 3
