from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from balls_and_urns.baselines import (
    dirichlet_posterior_alpha,
    discrete_posterior_weights,
    generalising_predictive,
    memorising_predictive,
    sample_generalising_posterior,
    sample_memorising_posterior,
)
from balls_and_urns.data import (
    BOSGenerator,
    make_bau_generator,
    make_discrete_bau_generator,
    make_generator_from_pool,
)
from balls_and_urns.dataset import (
    load_generalising_dataset,
    load_memorising_dataset,
    load_predictive_samples,
    save_generalising_dataset,
    save_memorising_dataset,
    save_predictive_samples,
)
from balls_and_urns.evals import BAUEvaluator, _cross_entropy, _mse, _per_position_ce
from balls_and_urns.predictive_monte_carlo import (
    predictive_monte_carlo_theta,
    predictive_monte_carlo_theta_chunked,
)
from pfn_transformerlens.model.PFN import DistributionPrediction


class CategoricalRolloutModel(nn.Module):
    def __init__(self, squeeze: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.squeeze = squeeze

    def generate(
        self,
        *,
        num_generate: int,
        prompt: torch.Tensor | None,
        num_rollouts: int,
        **_: object,
    ) -> torch.Tensor:
        prefix = torch.empty(0) if prompt is None else prompt
        generated = torch.arange(num_generate) % 2
        rows = torch.cat((prefix.long(), generated.long())).repeat(num_rollouts, 1)
        return rows[0] if self.squeeze else rows


class UniformBAUModel(nn.Module):
    def __init__(self, vocab_size: int, tuple_result: bool) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.tuple_result = tuple_result

    def predict_on_prompt(
        self,
        tokens: torch.Tensor,
    ) -> DistributionPrediction | tuple[DistributionPrediction, None]:
        probs = torch.full(
            (*tokens.shape, self.vocab_size),
            1.0 / self.vocab_size,
            device=tokens.device,
        )
        prediction = DistributionPrediction(
            probs=probs,
            y_grid=torch.arange(self.vocab_size, device=tokens.device).float(),
        )
        return (prediction, None) if self.tuple_result else prediction


def test_bau_baselines_are_normalized_and_sample_correct_shapes() -> None:
    tokens = torch.tensor([[0, 1, 1], [1, 0, 1]])
    alpha = torch.tensor([1.0, 2.0])
    theta_pool = torch.tensor([[0.8, 0.2], [0.2, 0.8]])

    generalising = generalising_predictive(tokens, alpha)
    memorising = memorising_predictive(tokens, theta_pool)
    assert generalising.shape == memorising.shape == (2, 3, 2)
    assert torch.allclose(generalising.sum(dim=-1), torch.ones(2, 3))
    assert torch.allclose(memorising.sum(dim=-1), torch.ones(2, 3))
    assert torch.allclose(memorising[:, 0], theta_pool.mean(dim=0).expand(2, -1))

    posterior_alpha = dirichlet_posterior_alpha(tokens[0], alpha)
    assert torch.equal(posterior_alpha, torch.tensor([2.0, 4.0]))
    weights = discrete_posterior_weights(tokens[0], theta_pool)
    assert weights.sum() == pytest.approx(1.0)
    assert sample_generalising_posterior(tokens[0], alpha, 4).shape == (4, 2)
    memorising_samples = sample_memorising_posterior(tokens[0], theta_pool, 4)
    assert memorising_samples.shape == (4, 2)


def test_bau_generators_and_bos_wrapper() -> None:
    torch.manual_seed(1)
    alpha = torch.ones(3)
    population_generator = make_bau_generator(alpha)
    discrete_generator, theta_pool = make_discrete_bau_generator(alpha, num_tasks=4)
    fixed_generator = make_generator_from_pool(theta_pool)
    assert population_generator.generate(5).shape == (5,)
    assert discrete_generator.generate(5).shape == (5,)
    assert fixed_generator.generate(5).shape == (5,)

    wrapped = BOSGenerator(fixed_generator, bos_token=3)
    tokens = wrapped.generate(5)
    assert tokens.shape == (5,)
    assert tokens[0].item() == 3
    assert wrapped.prior is fixed_generator.prior


def test_bau_dataset_round_trip(tmp_path) -> None:
    generalising_path = tmp_path / "eval" / "generalising.npz"
    tokens = np.array([[0, 1, 0]])
    alpha = np.array([1.0, 1.0])
    save_generalising_dataset(generalising_path, tokens, alpha, 2, 3, 1)
    generalising = load_generalising_dataset(generalising_path)
    np.testing.assert_array_equal(generalising["generalising_tokens"], tokens)
    assert generalising["vocab_size"] == 2

    memorising_path = tmp_path / "eval" / "memorising.npz"
    theta_pool = np.array([[0.8, 0.2], [0.3, 0.7]])
    save_memorising_dataset(memorising_path, tokens, theta_pool, 2)
    memorising = load_memorising_dataset(memorising_path)
    np.testing.assert_array_equal(memorising["theta_pool"], theta_pool)
    assert memorising["num_tasks"] == 2

    samples_path = tmp_path / "samples" / "run.npz"
    model_samples = np.full((1, 3, 2), 0.5)
    posterior_alpha = np.array([[2.0, 2.0]])
    posterior_weights = np.array([[0.4, 0.6]])
    save_predictive_samples(
        samples_path,
        model_samples,
        model_samples.copy(),
        model_samples.copy(),
        posterior_alpha,
        posterior_weights,
        alpha,
        theta_pool,
        tokens,
        step=10,
        prompt_source="data_memorising",
    )
    loaded = load_predictive_samples(samples_path)
    assert loaded["step"] == 10
    assert loaded["prompt_source"] == "data_memorising"
    np.testing.assert_array_equal(loaded["model_samples"], model_samples)


def test_bau_predictive_monte_carlo_prompt_and_chunk_paths() -> None:
    model = CategoricalRolloutModel()
    prompt = torch.tensor([1, 0])
    samples = predictive_monte_carlo_theta(
        model,
        vocab_size=2,
        forward_recursion_steps=4,
        num_rollouts=3,
        prompt=prompt,
        bos_token=2,
    )
    assert samples.shape == (3, 2)
    np.testing.assert_allclose(samples.sum(axis=1), 1.0)

    squeezed = predictive_monte_carlo_theta(
        CategoricalRolloutModel(squeeze=True),
        vocab_size=2,
        forward_recursion_steps=4,
        num_rollouts=1,
    )
    assert squeezed.shape == (1, 2)

    chunked = predictive_monte_carlo_theta_chunked(
        model,
        vocab_size=2,
        forward_recursion_steps=4,
        num_rollouts=5,
        chunk_size=2,
    )
    assert chunked.shape == (5, 2)


def test_bau_evaluator_and_scalar_metrics() -> None:
    alpha = torch.ones(2)
    memorising_base, _ = make_discrete_bau_generator(alpha, num_tasks=2)
    generalising_base = make_bau_generator(alpha)
    evaluator = BAUEvaluator(
        BOSGenerator(memorising_base, bos_token=2),
        BOSGenerator(generalising_base, bos_token=2),
        alpha,
        eval_batch_size=2,
        seq_len=4,
    )
    result = evaluator(UniformBAUModel(vocab_size=2, tuple_result=True))
    assert result.keys() >= {
        "ce/data_memorising",
        "ce/data_generalising",
        "deltas/data_memorising/delta_vs_baseline_memorising",
    }

    uniform = torch.full((2, 3, 2), 0.5)
    assert _mse(uniform, uniform) == 0.0
    assert _cross_entropy(uniform, uniform) == pytest.approx(np.log(2.0))
    assert _per_position_ce(uniform, uniform).shape == (3,)
