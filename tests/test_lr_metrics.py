from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from linear_regression.baselines import (
    compute_generalising_posterior,
    compute_memorising_posterior,
    generalising_predictor,
    memorising_predictor,
)
from linear_regression.evals import ICLEvaluator, mse
from linear_regression.likelihoods import linear_regression
from linear_regression.predictive_monte_carlo import (
    predictive_monte_carlo_beta,
    predictive_monte_carlo_beta_chunked,
    prepare_model_for_long_rollout,
)
from linear_regression.priors import DiscretePrior
from metrics import (
    energy_distance_multidim,
    marginal_wasserstein,
    sliced_wasserstein,
    symmetrised_kl,
)
from pfn_transformerlens import DeterministicGenerator
from pfn_transformerlens.model.PFN import DistributionPrediction


class RegressionRolloutModel(nn.Module):
    def __init__(self, n_ctx: int = 16) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.transformer = SimpleNamespace(cfg=SimpleNamespace(n_ctx=n_ctx))

    def generate(
        self,
        *,
        num_generate: int,
        prompt_x: torch.Tensor | None,
        prompt_y: torch.Tensor | None,
        num_rollouts: int,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_prompts = 1 if prompt_x is None else prompt_x.shape[0]
        prompt_length = 0 if prompt_x is None else prompt_x.shape[1]
        input_dim = 2 if prompt_x is None else prompt_x.shape[2]
        total_length = prompt_length + num_generate
        base_x = torch.arange(1, total_length + 1, dtype=torch.float32)
        x = torch.stack((base_x, torch.ones_like(base_x)), dim=-1)
        x = x.expand(num_prompts, num_rollouts, total_length, input_dim).clone()
        y = 2.0 * x[..., 0] - x[..., 1]
        if prompt_x is not None:
            assert prompt_y is not None
            x[:, :, :prompt_length] = prompt_x[:, None]
            y[:, :, :prompt_length] = prompt_y[:, None]
        return x, y


class PredictiveModel(nn.Module):
    def __init__(self, tuple_result: bool) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.tuple_result = tuple_result

    def predict_on_prompt(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> DistributionPrediction | tuple[DistributionPrediction, None]:
        del ys
        grid = torch.tensor([-1.0, 1.0], device=xs.device)
        probs = torch.full((*xs.shape[:2], 2), 0.5, device=xs.device)
        prediction = DistributionPrediction(probs=probs, y_grid=grid)
        return (prediction, None) if self.tuple_result else prediction


def test_distribution_metrics_cover_singleton_and_multisample_cases() -> None:
    p = torch.tensor([[0.25, 0.75]])
    assert symmetrised_kl(p, p) == pytest.approx(0.0)
    assert symmetrised_kl(p, torch.tensor([[0.75, 0.25]])) > 0

    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[1.0, 0.0], [2.0, 1.0]])
    assert sliced_wasserstein(a, a, n_projections=8) == pytest.approx(0.0)
    assert sliced_wasserstein(a, b, n_projections=8) > 0
    assert energy_distance_multidim(a, b) != 0
    assert energy_distance_multidim(a[:1], b[:1]) == pytest.approx(2.0)
    assert energy_distance_multidim(a, a) == pytest.approx(-np.sqrt(2.0))
    distances = marginal_wasserstein(a, b)
    assert distances.shape == (2,)
    assert distances[0] > distances[1]


@pytest.mark.parametrize(
    ("a", "b", "message"),
    [
        (np.ones(2), np.ones((2, 1)), "2D"),
        (np.ones((2, 1)), np.ones((2, 2)), "dimensions"),
        (np.empty((0, 1)), np.ones((2, 1)), "non-empty"),
        (np.array([[np.nan]]), np.ones((2, 1)), "finite"),
    ],
)
def test_sample_distance_metrics_reject_invalid_matrices(
    a: np.ndarray,
    b: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        energy_distance_multidim(a, b)


def test_distribution_metrics_reject_invalid_probabilities_and_projection_count() -> (
    None
):
    valid = torch.tensor([[0.25, 0.75]])
    invalid_pairs = [
        (valid, torch.ones(2, 2), "shapes"),
        (valid, torch.tensor([[float("nan"), 0.0]]), "finite"),
        (valid, torch.tensor([[-0.1, 1.1]]), "non-negative"),
        (torch.tensor([[0.2, 0.2]]), valid, "p must sum"),
        (valid, torch.tensor([[0.2, 0.2]]), "q must sum"),
    ]
    for p, q, message in invalid_pairs:
        with pytest.raises(ValueError, match=message):
            symmetrised_kl(p, q)

    with pytest.raises(ValueError, match="n_projections"):
        sliced_wasserstein(np.ones((1, 1)), np.ones((1, 1)), n_projections=0)


def test_discrete_prior_construction_sampling_and_device() -> None:
    tasks = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    prior = DiscretePrior(task_size=2, tasks=tasks)
    assert prior.num_tasks == 2
    assert prior.sample().shape == (2,)
    assert prior.sample((3,)).shape == (3, 2)
    assert prior.sample([2, 2]).shape == (2, 2, 2)
    assert prior.to("cpu") is prior
    assert prior.device.type == "cpu"

    generated = DiscretePrior(task_size=2, num_tasks=3)
    assert generated.tasks.shape == (3, 2)
    with pytest.raises(ValueError, match="num_tasks"):
        DiscretePrior(task_size=2)
    with pytest.raises(ValueError, match="tasks must have shape"):
        DiscretePrior(task_size=2, tasks=torch.ones(2, 3))


def test_linear_regression_baselines_and_posteriors() -> None:
    task_pool = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prior = DiscretePrior(task_size=2, tasks=task_pool)
    xs = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    ys = torch.tensor([[[1.0], [0.0], [1.0]]])

    memorising = memorising_predictor(
        xs,
        ys,
        prior,
        noise_variance=0.1,
        chunk_size=1,
        batch_chunk_size=1,
        compute_device="cpu",
    )
    assert memorising.shape == ys.shape
    assert memorising[0, 0, 0] == pytest.approx(0.5)

    generalising = generalising_predictor(xs, ys, noise_variance=0.1)
    assert generalising.shape == ys.shape
    assert torch.isfinite(generalising).all()
    assert linear_regression(xs[0], task_pool[0]).shape == (3,)

    mean, covariance = compute_generalising_posterior(
        xs[0],
        ys[0],
        sigma2=0.1,
    )
    assert mean.shape == (2,)
    assert covariance.shape == (2, 2)
    mean_batched, covariance_batched = compute_generalising_posterior(
        xs,
        ys,
        sigma2=0.1,
        tau2=2.0,
        device=torch.device("cpu"),
    )
    assert mean_batched.shape == mean.shape
    assert covariance_batched.shape == covariance.shape

    values, weights, masses = compute_memorising_posterior(
        xs[0],
        ys[0],
        task_pool,
        sigma2=0.1,
    )
    assert values.shape == (2, 2)
    assert weights.sum() == pytest.approx(1.0)
    assert len(masses) == 2
    values_batched, weights_batched, _ = compute_memorising_posterior(
        xs,
        ys,
        task_pool,
        sigma2=0.1,
        device=torch.device("cpu"),
    )
    np.testing.assert_allclose(values_batched, values)
    np.testing.assert_allclose(weights_batched, weights)


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (generalising_predictor, (torch.ones(2, 2), torch.ones(2, 1), 1.0)),
        (
            compute_generalising_posterior,
            (torch.ones(1, 1, 1, 1), torch.ones(1, 1), 1.0),
        ),
        (
            compute_memorising_posterior,
            (torch.ones(1, 1), torch.ones(1, 1), torch.ones(1, 1, 1), 1.0),
        ),
    ],
)
def test_linear_regression_baselines_fail_fast(
    function: Callable[..., object],
    args: tuple[object, ...],
) -> None:
    with pytest.raises(AssertionError):
        function(*args)


def test_predictive_monte_carlo_shapes_validation_and_chunking() -> None:
    model = RegressionRolloutModel(n_ctx=12)
    normal = torch.distributions.Normal(0.0, 1.0)
    assert prepare_model_for_long_rollout(model, rollout_length=3, prompt_length=1) == 3
    assert prepare_model_for_long_rollout(model, rollout_length=8, prompt_length=1) == 5
    with pytest.raises(AssertionError, match="exceeds"):
        prepare_model_for_long_rollout(model, rollout_length=1, prompt_length=6)

    no_prompt = predictive_monte_carlo_beta(model, normal, 3, 2)
    assert no_prompt.shape == (2, 2)
    np.testing.assert_allclose(
        no_prompt, np.array([[2.0, -1.0], [2.0, -1.0]]), atol=1e-5
    )

    prompt_x = torch.tensor([[1.0, 1.0]])
    prompt_y = torch.tensor([1.0])
    single, ys = predictive_monte_carlo_beta(
        model,
        normal,
        2,
        3,
        init_x=prompt_x,
        init_y=prompt_y,
        save_y=True,
    )
    assert single.shape == (3, 2)
    assert ys.shape == (3, 3)

    batched = predictive_monte_carlo_beta(
        model,
        normal,
        2,
        2,
        init_x=prompt_x.repeat(2, 1, 1),
        init_y=prompt_y.repeat(2, 1),
    )
    assert batched.shape == (2, 2, 2)

    chunked, chunked_y = predictive_monte_carlo_beta_chunked(
        model,
        normal,
        2,
        5,
        chunk_size=2,
        init_x=prompt_x.repeat(2, 1, 1),
        init_y=prompt_y.repeat(2, 1),
        save_y=True,
    )
    assert chunked.shape == (2, 5, 2)
    assert chunked_y.shape == (2, 5, 3)
    assert predictive_monte_carlo_beta_chunked(
        model,
        normal,
        2,
        3,
        chunk_size=2,
    ).shape == (3, 2)

    with pytest.raises(ValueError, match="both"):
        predictive_monte_carlo_beta(model, normal, 2, 2, init_x=prompt_x)
    with pytest.raises(AssertionError):
        predictive_monte_carlo_beta(model, normal, 0, 2)
    with pytest.raises(AssertionError):
        predictive_monte_carlo_beta_chunked(model, normal, 2, 2, chunk_size=0)


def test_icl_evaluator_with_tensor_and_tuple_predictions() -> None:
    tasks = torch.tensor([[1.0], [-1.0]])
    prior = DiscretePrior(task_size=1, tasks=tasks)
    memorising_generator = DeterministicGenerator(
        prior=prior,
        function=linear_regression,
        input_dim=1,
        noise_std=0.0,
    )
    generalising_generator = DeterministicGenerator(
        prior=torch.distributions.Normal(torch.zeros(1), torch.ones(1)),
        function=linear_regression,
        input_dim=1,
        noise_std=0.0,
    )
    evaluator = ICLEvaluator(
        memorising_generator,
        generalising_generator,
        max_examples=3,
        eval_batch_size=2,
        noise_variance=0.1,
    )
    result = evaluator(PredictiveModel(tuple_result=True))
    assert result.keys() >= {
        "mse/data_memorising",
        "mse/data_generalising",
        "pertoken/data_memorising/0",
    }
    assert mse(torch.tensor([1.0]), torch.tensor([3.0])).item() == 4.0
