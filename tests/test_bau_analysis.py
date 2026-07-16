from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import balls_and_urns.analysis as bau_analysis
from balls_and_urns.dataset import save_generalising_dataset, save_memorising_dataset
import balls_and_urns.distribution_dynamics as bau_dynamics
import balls_and_urns.sweep_analysis as bau_sweep
from pfn_transformerlens.model.PFN import DistributionPrediction


class PredictiveBAUModel(nn.Module):
    def __init__(self, vocab_size: int = 2, tuple_result: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.tuple_result = tuple_result
        self.transformer = SimpleNamespace(cfg=SimpleNamespace(n_ctx=8))

    def predict_on_prompt(
        self,
        tokens: torch.Tensor,
    ) -> DistributionPrediction | tuple[DistributionPrediction, None]:
        probabilities = torch.full(
            (*tokens.shape, self.vocab_size),
            1.0 / self.vocab_size,
            device=tokens.device,
        )
        prediction = DistributionPrediction(
            probs=probabilities,
            y_grid=torch.arange(self.vocab_size, device=tokens.device).float(),
        )
        return (prediction, None) if self.tuple_result else prediction


def _fake_theta_pmc(
    *,
    vocab_size: int,
    num_rollouts: int,
    **kwargs: object,
) -> np.ndarray:
    del kwargs
    return np.full((num_rollouts, vocab_size), 1.0 / vocab_size)


def _write_eval_data(dataset_dir: Path, run_id: str = "run") -> None:
    generalising_tokens = np.array([[0, 1], [1, 0]])
    memorising_tokens = np.array([[0, 0], [1, 1]])
    save_generalising_dataset(
        dataset_dir / "shared.npz",
        generalising_tokens,
        np.ones(2),
        vocab_size=2,
        seq_len=2,
        batch_size=2,
    )
    save_memorising_dataset(
        dataset_dir / f"{run_id}.npz",
        memorising_tokens,
        np.array([[0.8, 0.2], [0.2, 0.8]]),
        num_tasks=2,
    )


def test_shared_bau_analysis_predictive_and_sampling_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = torch.tensor([[0, 1], [1, 0]])
    with_bos = bau_analysis.prepend_bos(tokens, 2)
    assert with_bos.shape == (2, 3)
    with pytest.raises(ValueError, match="2D"):
        bau_analysis.prepend_bos(torch.ones(2), 2)

    alpha = torch.ones(2)
    theta_pool = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    metrics = bau_analysis.compute_predictive_metrics(
        PredictiveBAUModel(tuple_result=True),
        {"data": with_bos},
        alpha,
        theta_pool,
    )
    assert metrics.keys() == {
        "data/delta_vs_baseline_generalising",
        "data/delta_vs_baseline_memorising",
    }

    with pytest.raises(ValueError, match="n_samples"):
        bau_analysis.precompute_baseline_samples(0, alpha, theta_pool, None)
    prior_baselines = bau_analysis.precompute_baseline_samples(
        3, alpha, theta_pool, None
    )
    posterior_baselines = bau_analysis.precompute_baseline_samples(
        3, alpha, theta_pool, tokens
    )
    assert isinstance(prior_baselines, dict)
    assert isinstance(posterior_baselines, list)
    assert len(posterior_baselines) == 2

    monkeypatch.setattr(
        bau_analysis,
        "predictive_monte_carlo_theta_chunked",
        _fake_theta_pmc,
    )
    prior_metrics, prior_per_prompt = bau_analysis.compute_distribution_metrics(
        PredictiveBAUModel(),
        vocab_size=2,
        bos_token=2,
        alpha=alpha,
        theta_pool=theta_pool,
        effective_steps=2,
        n_samples=3,
        n_projections=2,
        chunk_size=2,
        prompts=None,
        baseline_samples=prior_baselines,
        samples_save_path=tmp_path / "prior.npz",
        step=1,
        prompt_source="prior",
    )
    assert prior_metrics == prior_per_prompt[0]
    assert (tmp_path / "prior.npz").exists()

    posterior_metrics, posterior_per_prompt = bau_analysis.compute_distribution_metrics(
        PredictiveBAUModel(),
        vocab_size=2,
        bos_token=2,
        alpha=alpha,
        theta_pool=theta_pool,
        effective_steps=2,
        n_samples=3,
        n_projections=2,
        chunk_size=2,
        prompts=tokens,
        baseline_samples=posterior_baselines,
        samples_save_path=tmp_path / "posterior.npz",
        step=1,
        prompt_source="data_memorising",
    )
    assert len(posterior_per_prompt) == 2
    assert posterior_metrics.keys() == prior_metrics.keys()
    with pytest.raises(ValueError, match="effective_steps"):
        bau_analysis.compute_distribution_metrics(
            PredictiveBAUModel(),
            vocab_size=2,
            bos_token=2,
            alpha=alpha,
            theta_pool=theta_pool,
            effective_steps=0,
            n_samples=3,
            n_projections=2,
            chunk_size=2,
            prompts=None,
            baseline_samples=prior_baselines,
            samples_save_path=tmp_path / "bad.npz",
            step=1,
            prompt_source="prior",
        )
    with pytest.raises(ValueError, match="must match"):
        bau_analysis.compute_distribution_metrics(
            PredictiveBAUModel(),
            vocab_size=2,
            bos_token=2,
            alpha=alpha,
            theta_pool=theta_pool,
            effective_steps=2,
            n_samples=3,
            n_projections=2,
            chunk_size=2,
            prompts=tokens,
            baseline_samples=[],
            samples_save_path=tmp_path / "bad.npz",
            step=1,
            prompt_source="data_memorising",
        )


def test_bau_evaluation_dataset_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_eval_data(tmp_path)
    theta_pool = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    alpha = torch.ones(2)
    shared_base = {
        "generalising_tokens": np.array([[0, 1]]),
        "alpha": np.ones(2),
        "vocab_size": 2,
        "seq_len": 2,
        "batch_size": 1,
    }
    memorising_base = {
        "memorising_tokens": np.array([[1, 0]]),
        "theta_pool": theta_pool.numpy(),
        "num_tasks": 2,
    }
    shared = shared_base.copy()
    memorising = memorising_base.copy()
    monkeypatch.setattr(bau_analysis, "load_generalising_dataset", lambda path: shared)
    monkeypatch.setattr(
        bau_analysis, "load_memorising_dataset", lambda path: memorising
    )

    invalid_cases = [
        ("shared", "vocab_size", 3, "vocabulary"),
        ("shared", "alpha", np.zeros(2), "alpha"),
        ("memorising", "num_tasks", 3, "task count"),
        ("memorising", "theta_pool", np.full((2, 2), 0.5), "task pool"),
        ("shared", "generalising_tokens", np.ones((1, 1, 1)), "must be 2D"),
        ("memorising", "memorising_tokens", np.array([[2]]), "outside"),
    ]
    for target, key, value, message in invalid_cases:
        shared.clear()
        shared.update(shared_base)
        memorising.clear()
        memorising.update(memorising_base)
        (shared if target == "shared" else memorising)[key] = value
        with pytest.raises(ValueError, match=message):
            bau_analysis.load_evaluation_tokens(tmp_path, "run", theta_pool, alpha)


@pytest.mark.parametrize(
    "config",
    [
        bau_sweep.SweepConfig(),
        bau_sweep.SweepConfig(eval_dataset_dir="data", n_samples=0),
        bau_sweep.SweepConfig(eval_dataset_dir="data", n_samples_prior=0),
        bau_sweep.SweepConfig(eval_dataset_dir="data", n_projections=0),
        bau_sweep.SweepConfig(eval_dataset_dir="data", predictive_steps=0),
        bau_sweep.SweepConfig(eval_dataset_dir="data", chunk_size=0),
        bau_sweep.SweepConfig(eval_dataset_dir="data", alpha_value=0),
    ],
)
def test_bau_sweep_config_validation(config: bau_sweep.SweepConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_bau_sweep_analysis_modes_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "data"
    _write_eval_data(dataset_dir)
    checkpoint_root = tmp_path / "checkpoints"
    run_dir = checkpoint_root / "run"
    skipped_dir = checkpoint_root / "skip"
    run_dir.mkdir(parents=True)
    skipped_dir.mkdir()
    checkpoint = run_dir / "checkpoint_step_5.pt"
    checkpoint.touch()
    model = PredictiveBAUModel()
    run_info = {
        "model": model,
        "tasks": torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
        "num_tasks": 2,
    }
    monkeypatch.setattr(bau_sweep, "UnsupervisedPFN", nn.Module)
    monkeypatch.setattr(bau_sweep, "load_run_info", lambda path, device: run_info)
    monkeypatch.setattr(
        bau_analysis,
        "predictive_monte_carlo_theta_chunked",
        _fake_theta_pmc,
    )
    config = bau_sweep.SweepConfig(
        checkpoint_root=str(checkpoint_root),
        eval_dataset_dir=str(dataset_dir),
        n_samples=2,
        n_samples_prior=3,
        n_projections=2,
        predictive_steps=2,
        chunk_size=2,
        device="cpu",
    )
    metrics = bau_sweep.run_analysis(config, tmp_path / "output")
    assert set(metrics["prompt_source"]) == {
        "prior",
        "data_generalising",
        "data_memorising",
    }

    prior_only = bau_sweep.run_analysis(
        replace(config, prior_only_distribution=True),
        tmp_path / "prior_only",
    )
    assert set(prior_only["prompt_source"]) == {"prior"}
    no_distribution = bau_sweep.run_analysis(
        replace(config, compute_distribution_metrics=False),
        tmp_path / "no_distribution",
    )
    assert set(no_distribution["prompt_source"]) == {"none"}

    with pytest.raises(FileNotFoundError, match="Checkpoint root"):
        bau_sweep.run_analysis(
            replace(config, checkpoint_root=str(tmp_path / "missing")),
            tmp_path / "bad",
        )
    with pytest.raises(FileNotFoundError, match="Generalising BAU"):
        bau_sweep.run_analysis(
            replace(config, eval_dataset_dir=str(tmp_path / "missing_data")),
            tmp_path / "bad",
        )
    (dataset_dir / "run.npz").unlink()
    with pytest.raises(FileNotFoundError, match="Memorising"):
        bau_sweep.run_analysis(config, tmp_path / "bad")


def test_bau_device_resolution_empty_sweep_and_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep_config = bau_sweep.SweepConfig(eval_dataset_dir="data")
    dynamics_config = bau_dynamics.DynamicsConfig(run_id="run", eval_dataset_dir="data")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert bau_sweep._resolve_device(sweep_config) == "cuda"
    assert bau_dynamics._resolve_device(dynamics_config) == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert bau_sweep._resolve_device(sweep_config) == "mps"
    assert bau_dynamics._resolve_device(dynamics_config) == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert bau_sweep._resolve_device(sweep_config) == "cpu"
    assert bau_dynamics._resolve_device(dynamics_config) == "cpu"

    checkpoint_root = tmp_path / "empty_checkpoints"
    (checkpoint_root / "run").mkdir(parents=True)
    dataset_dir = tmp_path / "data"
    _write_eval_data(dataset_dir)
    config = replace(
        sweep_config,
        checkpoint_root=str(checkpoint_root),
        eval_dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "output"),
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="No BAU checkpoints"):
        bau_sweep.run_analysis(config, tmp_path / "analysis")

    metrics = pd.DataFrame({"num_tasks": [2], "step": [1]})
    monkeypatch.setattr(bau_sweep, "run_analysis", lambda received, output: metrics)
    plotted: list[bau_sweep.PlotConfig] = []
    monkeypatch.setattr(bau_sweep, "plot_sweep", plotted.append)
    bau_sweep.main(config)
    output_dirs = list((tmp_path / "output").iterdir())
    assert len(output_dirs) == 1
    assert (output_dirs[0] / "config.json").exists()
    assert (output_dirs[0] / "metrics.csv").exists()
    assert len(plotted) == 1

    predictive_only = replace(
        config,
        compute_distribution_metrics=False,
        output_dir=str(tmp_path / "predictive_only"),
    )
    bau_sweep.main(predictive_only)
    predictive_dirs = list((tmp_path / "predictive_only").iterdir())
    assert len(predictive_dirs) == 1
    assert (predictive_dirs[0] / "metrics.csv").exists()
    assert len(plotted) == 1


@pytest.mark.parametrize(
    "config",
    [
        bau_dynamics.DynamicsConfig(),
        bau_dynamics.DynamicsConfig(run_id="run"),
        bau_dynamics.DynamicsConfig(run_id="run", eval_dataset_dir="data", n_samples=0),
        bau_dynamics.DynamicsConfig(
            run_id="run", eval_dataset_dir="data", predictive_steps=0
        ),
        bau_dynamics.DynamicsConfig(
            run_id="run", eval_dataset_dir="data", n_projections=0
        ),
        bau_dynamics.DynamicsConfig(
            run_id="run", eval_dataset_dir="data", chunk_size=0
        ),
        bau_dynamics.DynamicsConfig(
            run_id="run", eval_dataset_dir="data", checkpoint_subsample=0
        ),
        bau_dynamics.DynamicsConfig(
            run_id="run", eval_dataset_dir="data", alpha_value=0
        ),
    ],
)
def test_bau_dynamics_config_validation(config: bau_dynamics.DynamicsConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_bau_dynamics_analysis_and_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "data"
    _write_eval_data(dataset_dir)
    checkpoint_root = tmp_path / "checkpoints"
    run_dir = checkpoint_root / "run"
    run_dir.mkdir(parents=True)
    for step in [1, 2, 3]:
        (run_dir / f"checkpoint_step_{step}.pt").touch()
    model = PredictiveBAUModel(tuple_result=True)
    run_info = {
        "model": model,
        "tasks": torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
        "num_tasks": 2,
    }
    monkeypatch.setattr(bau_dynamics, "UnsupervisedPFN", nn.Module)
    monkeypatch.setattr(bau_dynamics, "load_run_info", lambda path, device: run_info)
    monkeypatch.setattr(
        bau_analysis,
        "predictive_monte_carlo_theta_chunked",
        _fake_theta_pmc,
    )
    config = bau_dynamics.DynamicsConfig(
        run_id="run",
        checkpoint_root=str(checkpoint_root),
        eval_dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "output"),
        n_samples=2,
        n_projections=2,
        predictive_steps=2,
        chunk_size=2,
        checkpoint_subsample=2,
        device="cpu",
    )
    metrics, task_count, per_prompt = bau_dynamics.run_analysis(config)
    assert task_count == 2
    assert metrics["step"].tolist() == [1, 3]
    assert not per_prompt.empty
    assert "delta_vs_baseline_memorising_on_data_memorising" in metrics

    no_metrics, _, no_per_prompt = bau_dynamics.run_analysis(
        replace(config, compute_delta=False, compute_distribution=False)
    )
    assert list(no_metrics.columns) == ["step"]
    assert no_per_prompt.empty

    monkeypatch.setattr(
        bau_dynamics,
        "run_analysis",
        lambda config: (pd.DataFrame({"step": [1]}), 2, pd.DataFrame({"step": [1]})),
    )
    plotted: list[bau_dynamics.PlotConfig] = []
    monkeypatch.setattr(bau_dynamics, "plot_dynamics", plotted.append)
    bau_dynamics.main(config)
    output_dir = Path(config.output_dir) / "run"
    assert (output_dir / "config.json").exists()
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "per_prompt_metrics.csv").exists()
    assert len(plotted) == 1

    bau_dynamics.main(replace(config, run_id="distribution-only", compute_delta=False))
    bau_dynamics.main(replace(config, run_id="delta-only", compute_distribution=False))
    assert len(plotted) == 1


def test_bau_dynamics_missing_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bau_dynamics.DynamicsConfig(
        run_id="run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_dataset_dir=str(tmp_path / "data"),
        device="cpu",
    )
    with pytest.raises(FileNotFoundError, match="Run directory"):
        bau_dynamics.run_analysis(config)

    run_dir = tmp_path / "checkpoints/run"
    run_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="No checkpoints"):
        bau_dynamics.run_analysis(config)

    checkpoint = run_dir / "checkpoint_step_1.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        bau_dynamics,
        "load_run_info",
        lambda path, device: {
            "model": PredictiveBAUModel(),
            "tasks": torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
            "num_tasks": 2,
        },
    )
    with pytest.raises(FileNotFoundError, match="evaluation dataset"):
        bau_dynamics.run_analysis(config)
