from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import analysis.checkpoints as checkpoint_helpers
from linear_regression.analysis.config import SweepConfig
import linear_regression.analysis.data as analysis_data
import linear_regression.analysis.metrics as analysis_metrics
from linear_regression.analysis.data import PromptData
from linear_regression.priors import DiscretePrior
from pfn_transformerlens.model.PFN import DistributionPrediction


class PredictiveRegressionModel(nn.Module):
    def __init__(self, task_size: int = 2, tuple_result: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.task_size = task_size
        self.tuple_result = tuple_result

    def predict_on_prompt(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> DistributionPrediction | tuple[DistributionPrediction, None]:
        del ys
        y_grid = torch.tensor([-1.0, 1.0], device=xs.device)
        probs = torch.full((*xs.shape[:2], 2), 0.5, device=xs.device)
        prediction = DistributionPrediction(probs=probs, y_grid=y_grid)
        return (prediction, None) if self.tuple_result else prediction


def _fake_pmc(
    *,
    model: PredictiveRegressionModel,
    forward_recursion_samples: int,
    init_x: torch.Tensor | None,
    **kwargs: object,
) -> np.ndarray:
    del kwargs
    if init_x is None:
        return np.zeros((forward_recursion_samples, model.task_size))
    if init_x.ndim == 2:
        return np.zeros((forward_recursion_samples, model.task_size))
    return np.zeros((init_x.shape[0], forward_recursion_samples, model.task_size))


def test_checkpoint_discovery_and_metadata_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert checkpoint_helpers.find_latest_checkpoint(tmp_path) is None
    assert checkpoint_helpers.find_all_checkpoints(tmp_path) == []
    paths = [tmp_path / "checkpoint_step_10.pt", tmp_path / "checkpoint_step_2.pt"]
    for path in paths:
        path.touch()
    assert checkpoint_helpers.get_step(paths[0]) == 10
    assert checkpoint_helpers.find_latest_checkpoint(tmp_path) == paths[0]
    assert checkpoint_helpers.find_all_checkpoints(tmp_path) == [paths[1], paths[0]]

    model = PredictiveRegressionModel()
    model.task_distribution = {
        "tasks": torch.ones(2, 2),
        "num_tasks": 2,
    }
    monkeypatch.setattr(checkpoint_helpers, "BasePFN", nn.Module)
    monkeypatch.setattr(
        checkpoint_helpers.checkpointing,
        "load_checkpoint",
        lambda *args, **kwargs: (model, None, {"step": 10}),
    )
    info = checkpoint_helpers.load_run_info(paths[0], "cpu")
    assert info["num_tasks"] == 2

    del model.task_distribution
    with pytest.raises(AssertionError, match="no task_distribution"):
        checkpoint_helpers.load_run_info(paths[0], "cpu")


def test_lr_analysis_data_helpers_and_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert analysis_data.find_latest_checkpoint(run_dir) is None
    first = run_dir / "checkpoint_step_1.pt"
    latest = run_dir / "checkpoint_step_4.pt"
    first.touch()
    latest.touch()
    assert analysis_data.find_latest_checkpoint(run_dir) == latest

    prior = DiscretePrior(task_size=2, tasks=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    memorising, generalising = analysis_data.create_generators(prior, 2, 0.1)
    memorising_xs, memorising_ys = memorising.generate(3)
    generalising_xs, generalising_ys = generalising.generate(3)
    assert memorising_xs.shape == generalising_xs.shape == (3, 2)
    assert memorising_ys.shape == generalising_ys.shape == (3,)
    for source in ["gaussian", "discrete", "random"]:
        xs, ys = analysis_data.sample_prompt(prior, source, 3, 0.1)
        assert xs.shape == (3, 2)
        assert ys.shape == (3,)
    with pytest.raises(ValueError, match="unsupported prompt source"):
        analysis_data.sample_prompt(prior, "bad", 3, 0.1)

    prompt_data = analysis_data.generate_prompt_data(prior, 3, 4, 0.1)
    assert prompt_data.xs.shape == (4, 3, 2)
    for source in ["gaussian", "discrete", "random"]:
        xs, ys = analysis_data.get_prompts_for_config(prompt_data, source, 2, 3)
        assert xs.shape == (3, 2, 2)
        assert ys.shape == (3, 2)
    with pytest.raises(ValueError, match="unsupported prompt source"):
        analysis_data.get_prompts_for_config(prompt_data, "bad", 2, 3)

    shared_path = tmp_path / "datasets" / "shared.npz"
    analysis_data.save_shared_dataset(
        shared_path,
        prompt_data.xs,
        prompt_data.ys_gaussian,
        prompt_data.ys_random,
        prompt_data.xs,
        prompt_data.ys_gaussian,
        prompt_data.xs,
        prompt_data.ys_random,
        task_size=2,
        noise_std=0.1,
        eval_batch_size=4,
        seq_len=3,
        max_prompt_length=3,
        max_n_prompts=4,
    )
    assert int(analysis_data.load_shared_dataset(shared_path)["task_size"]) == 2

    prior_path = tmp_path / "datasets" / "run.npz"
    analysis_data.save_prior_dataset(
        prior_path,
        prompt_data.ys_discrete,
        prompt_data.xs,
        prompt_data.ys_discrete,
        num_tasks=2,
        run_id="run",
    )
    assert str(analysis_data.load_prior_dataset(prior_path)["run_id"]) == "run"

    model = PredictiveRegressionModel()
    model.task_distribution = {
        "num_tasks": 2,
        "task_size": 2,
        "tasks": prior.tasks,
    }
    monkeypatch.setattr(analysis_data, "SupervisedPFN", nn.Module)
    monkeypatch.setattr(
        analysis_data.checkpointing,
        "load_checkpoint",
        lambda *args, **kwargs: (model, None, {"step": 4}),
    )
    info = analysis_data.load_run_info(latest, device="cpu")
    assert info["task_size"] == 2
    del model.task_distribution
    with pytest.raises(ValueError, match="no task_distribution"):
        analysis_data.load_run_info(latest, device="cpu")


@pytest.mark.parametrize(
    "config",
    [
        SweepConfig(eval_batch_size=0),
        SweepConfig(noise_std=0),
        SweepConfig(seq_len=2, eval_position=2),
        SweepConfig(prompt_lengths=()),
        SweepConfig(prompt_sources=("bad",)),
        SweepConfig(n_prompts=(0,)),
        SweepConfig(n_projections=0),
        SweepConfig(eval_n_prompts=0),
    ],
)
def test_lr_analysis_config_rejects_invalid_settings(config: SweepConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_lr_predictive_and_posterior_metrics() -> None:
    model = PredictiveRegressionModel(tuple_result=True)
    prior = DiscretePrior(task_size=2, tasks=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    xs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    ys = torch.tensor([[1.0, 0.0]])
    config = SweepConfig(eval_position=1, eval_batch_size=1, seq_len=2)
    metrics = analysis_metrics.compute_predictive_metrics(
        model,
        xs,
        ys,
        prior,
        noise_variance=0.25,
        config=config,
    )
    assert metrics.keys() >= {
        "model_mse",
        "baseline_memorising_mse",
        "delta_vs_baseline_generalising",
    }

    weights = analysis_metrics.compute_dmmse_weights(xs[0], ys[0], prior, 0.25)
    assert weights.sum() == pytest.approx(1.0)
    assert analysis_metrics.sample_dmmse_posterior(
        xs[0], ys[0], prior, 0.25, 3
    ).shape == (3, 2)
    mean, covariance = analysis_metrics.gaussian_posterior_params(xs[0], ys[0], 0.25)
    assert mean.shape == (2,)
    assert covariance.shape == (2, 2)
    assert analysis_metrics.sample_ridge_posterior(xs[0], ys[0], 0.25, 3).shape == (
        3,
        2,
    )


def test_lr_distribution_metrics_cover_prior_and_posterior_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = PredictiveRegressionModel()
    prior = DiscretePrior(task_size=2, tasks=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    monkeypatch.setattr(
        analysis_metrics,
        "predictive_monte_carlo_beta_chunked",
        _fake_pmc,
    )
    monkeypatch.setattr(
        analysis_metrics,
        "prepare_model_for_long_rollout",
        lambda model, rollout_length, prompt_length: rollout_length,
    )

    prior_metrics, prior_samples, prior_per_prompt = (
        analysis_metrics.compute_distribution_metrics_single(
            model,
            prior,
            noise_std=0.1,
            noise_variance=0.01,
            n_projections=2,
            prompt_source="N/A",
            prompt_length=0,
            predictive_steps=4,
            n_samples=0,
            n_samples_prior=4,
            n_prompts=0,
        )
    )
    assert prior_metrics == prior_per_prompt[0]
    assert prior_samples["dmmse_weights"].sum() == pytest.approx(1.0)

    prompt_data = PromptData(
        xs=np.ones((2, 2, 2), dtype=np.float32),
        ys_gaussian=np.ones((2, 2), dtype=np.float32),
        ys_discrete=np.zeros((2, 2), dtype=np.float32),
        ys_random=np.full((2, 2), 0.5, dtype=np.float32),
    )
    posterior_metrics, posterior_samples, per_prompt = (
        analysis_metrics.compute_distribution_metrics_single(
            model,
            prior,
            noise_std=0.1,
            noise_variance=0.01,
            n_projections=2,
            prompt_source="gaussian",
            prompt_length=2,
            predictive_steps=4,
            n_samples=3,
            n_samples_prior=0,
            n_prompts=2,
            model_prepared=True,
            prompt_data=prompt_data,
        )
    )
    assert len(per_prompt) == 2
    assert posterior_metrics.keys() >= {"dist/ed_vs_baseline_memorising"}
    assert posterior_samples["pt"].shape == (2, 3, 2)

    sampled_metrics, sampled_data, sampled_per_prompt = (
        analysis_metrics.compute_distribution_metrics_single(
            model,
            prior,
            noise_std=0.1,
            noise_variance=0.01,
            n_projections=2,
            prompt_source="random",
            prompt_length=2,
            predictive_steps=4,
            n_samples=3,
            n_samples_prior=0,
            n_prompts=1,
            model_prepared=True,
        )
    )
    assert len(sampled_per_prompt) == 1
    assert sampled_metrics.keys() == posterior_metrics.keys()
    assert sampled_data["prompt_xs"].shape == (1, 2, 2)


def test_lr_all_predictive_metrics_and_result_merging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = PredictiveRegressionModel()
    prior = DiscretePrior(task_size=2, tasks=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    config = SweepConfig(eval_batch_size=1, seq_len=2, include_random_eval=True)
    batch = (torch.ones(1, 2, 2), torch.ones(1, 2))
    eval_data = {
        "data_memorising": batch,
        "data_generalising": batch,
        "random": batch,
    }
    monkeypatch.setattr(analysis_metrics, "SupervisedPFN", nn.Module)
    provided = analysis_metrics.compute_all_predictive_metrics(
        model,
        prior,
        config,
        eval_data=eval_data,
    )
    assert provided.keys() >= {"data_memorising/model_mse", "data_random/model_mse"}

    monkeypatch.setattr(
        analysis_metrics,
        "sample_batch",
        lambda generator, batch_size, seq_len: (
            torch.ones(batch_size, seq_len, 2),
            torch.ones(batch_size, seq_len),
        ),
    )
    generated = analysis_metrics.compute_all_predictive_metrics(
        model,
        prior,
        config,
    )
    assert generated.keys() == provided.keys()

    no_random = analysis_metrics.compute_all_predictive_metrics(
        model,
        prior,
        SweepConfig(eval_batch_size=1, seq_len=2, include_random_eval=False),
        eval_data=eval_data,
    )
    assert not any(key.startswith("data_random") for key in no_random)

    predictive = [{"run_id": "a", "num_tasks": 2, "checkpoint_step": 1, "mse": 1.0}]
    only_predictive = analysis_metrics.merge_results(predictive, [])
    assert only_predictive.iloc[0]["mse"] == 1.0
    distribution = [
        {
            "run_id": "a",
            "num_tasks": 2,
            "checkpoint_step": 1,
            "prompt_source": "N/A",
            "prompt_length": 0,
            "n_samples": 0,
            "n_samples_prior": 2,
            "n_prompts": 0,
        }
    ]
    merged = analysis_metrics.merge_results(predictive, distribution)
    assert merged.iloc[0]["mse"] == 1.0
    assert SweepConfig(noise_std=0.5).noise_variance == pytest.approx(0.25)
