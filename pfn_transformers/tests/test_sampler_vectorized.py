"""Tests for the vectorized Sampler behavior and batching semantics."""

from __future__ import annotations

import math

import pytest
import torch
from jaxtyping import Float
from torch.distributions import Normal
from torch.utils.data import DataLoader

from pfn_transformerlens.model.configs import SupervisedRegressionPFNConfig
from pfn_transformerlens.sampler.data_generator import SupervisedProbabilisticGenerator
from pfn_transformerlens.sampler.dataloader import sample_batch
from pfn_transformerlens.sampler.prior_likelihood import (
    LikelihoodDistribution,
    PriorDistribution,
)
from pfn_transformerlens.sampler.sampler import Sampler, SamplerConfig


def _make_linear_likelihood(
    noise_std: float, input_dim: int = 1
) -> LikelihoodDistribution:
    """Linear likelihood: y = theta^T x + Normal(0, noise_std)."""

    # PyTorch's Normal requires scale > 0, so use a small epsilon for zero noise
    eps = 1e-10
    effective_noise_std = max(noise_std, eps)
    base_dist = Normal(0.0, effective_noise_std)

    def parameterizer(
        prior_sample: Float[torch.Tensor, "..."],
        x: Float[torch.Tensor, "seq input_dim"],
    ) -> dict[str, Float[torch.Tensor, "..."]]:
        # Match the logic used in the repo's example (handle shapes robustly)
        if prior_sample.dim() == 0:
            prior_sample = prior_sample.unsqueeze(0).expand(x.shape[-1])
        if prior_sample.dim() == 1:
            prior_sample = prior_sample.unsqueeze(0)
        if prior_sample.shape[-1] != x.shape[-1]:
            raise ValueError(
                f"Prior dimension {prior_sample.shape[-1]} must match input dimension {x.shape[-1]}"
            )

        mean = torch.sum(prior_sample * x, dim=-1)
        std = torch.full_like(mean, max(noise_std, eps))
        return {"loc": mean, "scale": std}

    return LikelihoodDistribution(
        base_distribution=base_dist,
        parameterizer=parameterizer,
        input_dim=input_dim,
    )


def test_sampler_batch_shapes_and_collation() -> None:
    """Sampler yields batched (x, y) with correct shapes via DataLoader collation."""
    B = 16
    T = 12
    D = 1
    noise_std = 0.1

    prior = PriorDistribution(Normal(0.0, 1.0))
    likelihood = _make_linear_likelihood(noise_std, input_dim=D)
    cfg = SupervisedRegressionPFNConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        d_mlp=64,
        d_vocab=100,
        n_ctx=100,
        d_head=32,
        act_fn="gelu",
        normalization_type="LN",
        input_dim=D,
        bucket_type="uniform",
        y_min=-5.0,
        y_max=5.0,
    )
    sampler = Sampler(
        seq_len=T, config=SamplerConfig(prior, likelihood, cfg), internal_batch_size=B
    )

    loader = DataLoader(sampler, batch_size=B, shuffle=False, num_workers=0)
    x, y = next(iter(loader))

    assert x.shape == (B, T, D)
    assert y.shape == (B, T)


def test_sampler_batch_shapes_mismatched_internal_batch() -> None:
    """Sampler still produces correct batch shapes when internal_batch_size != loader batch_size."""
    internal_B = 7
    loader_B = 5
    T = 9
    D = 1

    prior = PriorDistribution(Normal(0.0, 1.0))
    likelihood = _make_linear_likelihood(0.2, input_dim=D)
    cfg = SupervisedRegressionPFNConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        d_mlp=64,
        d_vocab=100,
        n_ctx=100,
        d_head=32,
        act_fn="gelu",
        normalization_type="LN",
        input_dim=D,
        bucket_type="uniform",
        y_min=-5.0,
        y_max=5.0,
    )
    sampler = Sampler(
        seq_len=T,
        config=SamplerConfig(prior, likelihood, cfg),
        internal_batch_size=internal_B,
    )

    loader = DataLoader(sampler, batch_size=loader_B, shuffle=False, num_workers=0)
    x, y = next(iter(loader))

    assert x.shape == (loader_B, T, D)
    assert y.shape == (loader_B, T)


def test_sampler_conditioning_per_item_means_with_constant_x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    With x set to all-ones, the per-item mean of y across the sequence should
    match that item's theta (up to noise), verifying item-wise conditioning.
    """
    B = 8
    T = 64
    D = 1
    noise_std = 0.05

    # Create a custom distribution that returns ones
    class OnesDistribution(torch.distributions.Distribution):
        arg_constraints: dict[str, object] = {}

        def sample(
            self, sample_shape: torch.Size = torch.Size()
        ) -> Float[torch.Tensor, "..."]:
            return torch.ones(sample_shape)

    x_dist = OnesDistribution()

    # Build prior and monkeypatch its sample to return known thetas
    prior = PriorDistribution(Normal(0.0, 1.0))
    theta_batch = torch.linspace(-1.0, 1.0, steps=B)
    theta_iter = iter(theta_batch)
    monkeypatch.setattr(prior, "sample", lambda sample_shape: next(theta_iter))

    likelihood = _make_linear_likelihood(noise_std, input_dim=D)
    cfg = SupervisedRegressionPFNConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        d_mlp=64,
        d_vocab=100,
        n_ctx=100,
        d_head=32,
        act_fn="gelu",
        normalization_type="LN",
        input_dim=D,
        bucket_type="uniform",
        y_min=-5.0,
        y_max=5.0,
    )
    sampler = Sampler(
        seq_len=T,
        config=SamplerConfig(prior, likelihood, cfg, x_distribution=x_dist),
        internal_batch_size=B,
    )
    loader = DataLoader(sampler, batch_size=B, shuffle=False, num_workers=0)

    x, y = next(iter(loader))
    assert x.shape == (B, T, D)
    assert torch.allclose(x, torch.ones(B, T, D))

    # Per-item mean across time should approximate theta_i
    y_means = y.mean(dim=1)
    # Tolerance ~ 3 * noise_std / sqrt(T)
    tol = 3 * noise_std / math.sqrt(T)
    assert torch.allclose(y_means, theta_batch, atol=tol)


def test_sampler_empirical_y_stats_match_expectation() -> None:
    """
    For 1D theta ~ N(0,1), x ~ N(0,1), y = theta*x + Normal(0, s), across items/positions:
    E[y] ~ 0 and Var[y] ~ 1 + s^2.
    """
    torch.manual_seed(0)
    B = 64
    T = 128
    D = 1
    noise_std = 0.1

    prior = PriorDistribution(Normal(0.0, 1.0))
    likelihood = _make_linear_likelihood(noise_std, input_dim=D)
    cfg = SupervisedRegressionPFNConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        d_mlp=64,
        d_vocab=100,
        n_ctx=100,
        d_head=32,
        act_fn="gelu",
        normalization_type="LN",
        input_dim=D,
        bucket_type="uniform",
        y_min=-5.0,
        y_max=5.0,
    )
    sampler = Sampler(
        seq_len=T, config=SamplerConfig(prior, likelihood, cfg), internal_batch_size=B
    )
    loader = DataLoader(sampler, batch_size=B, shuffle=False, num_workers=0)

    _, y = next(iter(loader))
    y_flat = y.reshape(-1)

    mean = y_flat.mean().item()
    var = y_flat.var(unbiased=True).item()

    assert abs(mean) < 0.1  # close to 0
    expected_var = 1.0 + noise_std * noise_std
    assert abs(var - expected_var) < 0.2


def test_y_over_x_constant_when_noise_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    With noise_std = 0 and x set to ones, y = theta * x exactly, so y/x == theta
    for every timestep within each prompt. This verifies constant theta per item.
    """
    B = 8
    T = 32
    D = 1
    noise_std = 0.0

    # Create a custom distribution that returns ones
    class OnesDistribution(torch.distributions.Distribution):
        arg_constraints: dict[str, object] = {}

        def sample(
            self, sample_shape: torch.Size = torch.Size()
        ) -> Float[torch.Tensor, "..."]:
            return torch.ones(sample_shape)

    x_dist = OnesDistribution()

    # Fix per-item theta across the batch
    prior = PriorDistribution(Normal(0.0, 1.0))
    theta_batch = torch.linspace(-1.0, 1.0, steps=B)
    theta_iter = iter(theta_batch)
    monkeypatch.setattr(prior, "sample", lambda sample_shape: next(theta_iter))

    likelihood = _make_linear_likelihood(noise_std, input_dim=D)
    cfg = SupervisedRegressionPFNConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        d_mlp=64,
        d_vocab=100,
        n_ctx=100,
        d_head=32,
        act_fn="gelu",
        normalization_type="LN",
        input_dim=D,
        bucket_type="uniform",
        y_min=-5.0,
        y_max=5.0,
    )
    sampler = Sampler(
        seq_len=T,
        config=SamplerConfig(prior, likelihood, cfg, x_distribution=x_dist),
        internal_batch_size=B,
    )
    loader = DataLoader(sampler, batch_size=B, shuffle=False, num_workers=0)

    x, y = next(iter(loader))
    # y/x should equal theta for every timestep when noise == 0 and x == 1
    ratios = y / x.squeeze(-1)
    target = theta_batch.view(B, 1).expand(B, T)

    assert torch.allclose(ratios, target, atol=1e-7, rtol=0.0)


def test_y_over_x_near_constant_when_noise_tiny(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    With very small noise and x = 1, y/x should be close to theta per prompt;
    the per-item standard deviation over time should be tiny.
    """
    B = 8
    T = 64
    D = 1
    noise_std = 1e-4

    # Create a custom distribution that returns ones
    class OnesDistribution(torch.distributions.Distribution):
        arg_constraints: dict[str, object] = {}

        def sample(
            self, sample_shape: torch.Size = torch.Size()
        ) -> Float[torch.Tensor, "..."]:
            return torch.ones(sample_shape)

    x_dist = OnesDistribution()

    prior = PriorDistribution(Normal(0.0, 1.0))
    theta_batch = torch.linspace(-0.5, 0.5, steps=B)
    theta_iter = iter(theta_batch)
    monkeypatch.setattr(prior, "sample", lambda sample_shape: next(theta_iter))

    likelihood = _make_linear_likelihood(noise_std, input_dim=D)
    cfg = SupervisedRegressionPFNConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        d_mlp=64,
        d_vocab=100,
        n_ctx=100,
        d_head=32,
        act_fn="gelu",
        normalization_type="LN",
        input_dim=D,
        bucket_type="uniform",
        y_min=-5.0,
        y_max=5.0,
    )
    sampler = Sampler(
        seq_len=T,
        config=SamplerConfig(prior, likelihood, cfg, x_distribution=x_dist),
        internal_batch_size=B,
    )
    loader = DataLoader(sampler, batch_size=B, shuffle=False, num_workers=0)

    x, y = next(iter(loader))
    ratios = y / x.squeeze(-1)

    # Per-item std over time should scale with noise_std
    per_item_std = ratios.std(dim=1, unbiased=True)
    assert torch.all(per_item_std < 5e-4)


def test_sample_batch_returns_correct_shapes() -> None:
    batch_size = 4
    seq_len = 6
    input_dim = 1
    prior = PriorDistribution(Normal(0.0, 1.0))
    likelihood = _make_linear_likelihood(0.1, input_dim=input_dim)
    data_generator = SupervisedProbabilisticGenerator(prior, likelihood)

    x, y = sample_batch(
        data_generator,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    assert x is not None, "supervised generator should return x"
    assert x.shape == (batch_size, seq_len, input_dim)
    assert y.shape == (batch_size, seq_len)
