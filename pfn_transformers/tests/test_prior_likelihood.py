"""Tests for prior and likelihood distribution classes."""

import pytest
import torch
from jaxtyping import Float
from torch.distributions import Bernoulli, Gamma, Normal

from pfn_transformerlens.sampler.prior_likelihood import (
    DistributionParams,
    LikelihoodDistribution,
    PriorDistribution,
)


class TestPriorDistribution:
    """Tests for PriorDistribution wrapper."""

    def test_init(self) -> None:
        """Test initialization with a base distribution."""
        base_dist = Normal(0.0, 1.0)
        prior = PriorDistribution(base_dist)

        assert prior.base_distribution is base_dist
        assert prior.batch_shape == base_dist.batch_shape
        assert prior.event_shape == base_dist.event_shape

    def test_sample(self) -> None:
        """Test sampling from prior."""
        base_dist = Normal(0.0, 1.0)
        prior = PriorDistribution(base_dist)

        samples = prior.sample((100,))
        assert samples.shape == (100,)
        assert torch.abs(samples.mean()) < 0.2  # Should be close to 0
        assert torch.abs(samples.std() - 1.0) < 0.2  # Should be close to 1

    def test_log_prob(self) -> None:
        """Test log probability computation."""
        base_dist = Normal(0.0, 1.0)
        prior = PriorDistribution(base_dist)

        value = torch.tensor([0.0, 1.0, -1.0])
        log_prob = prior.log_prob(value)
        expected = base_dist.log_prob(value)

        assert torch.allclose(log_prob, expected)

    def test_mean_variance(self) -> None:
        """Test mean and variance properties."""
        base_dist = Normal(2.0, 3.0)
        prior = PriorDistribution(base_dist)

        assert torch.allclose(prior.mean, torch.tensor(2.0))
        assert torch.allclose(prior.variance, torch.tensor(9.0))


class TestLikelihoodDistribution:
    """Tests for LikelihoodDistribution with different distributions and parameter formats."""

    def test_init(self) -> None:
        """Test initialization."""
        base_dist = Normal(0.0, 1.0)

        def parameterizer(
            prior: Float[torch.Tensor, "..."],
            x: Float[torch.Tensor, "seq input_dim"],
        ) -> dict[str, Float[torch.Tensor, "..."]]:
            return {"loc": prior, "scale": torch.ones_like(prior)}

        likelihood = LikelihoodDistribution(
            base_distribution=base_dist,
            parameterizer=parameterizer,
            input_dim=1,
        )

        assert likelihood.base_distribution is base_dist
        assert likelihood.distribution_class == Normal
        assert likelihood.input_dim == 1
        assert likelihood._current_params is None

    def test_normal_with_dict_params(self) -> None:
        """Test Normal distribution with dict parameters (loc, scale)."""
        torch.manual_seed(42)  # For reproducible results
        base_dist = Normal(0.0, 1.0)

        # Parameterizer returns dict with 'loc' and 'scale'
        def parameterizer(
            prior: Float[torch.Tensor, "..."],
            x: Float[torch.Tensor, "seq input_dim"],
        ) -> dict[str, Float[torch.Tensor, "..."]]:
            seq_len = x.shape[0]
            mean = prior.expand(seq_len)
            std = torch.ones(seq_len)
            return {"loc": mean, "scale": std}

        likelihood = LikelihoodDistribution(
            base_distribution=base_dist,
            parameterizer=parameterizer,
            input_dim=1,
        )

        # Condition on prior and input
        prior_sample = torch.tensor(2.5)
        x = torch.randn(10, 1)
        conditioned = likelihood.condition_on_prior_and_input(prior_sample, x)

        # Test properties
        assert torch.allclose(conditioned.mean, torch.full((10,), 2.5))
        assert torch.allclose(conditioned.variance, torch.ones(10))

        # Test sampling with larger sample size for more reliable test
        samples = conditioned.sample((1000,))
        assert samples.shape == (1000, 10)
        # Check that each position has roughly the correct mean
        mean_per_position = samples.mean(dim=0)
        assert torch.allclose(mean_per_position, torch.full((10,), 2.5), atol=0.15)

        # Test log_prob
        test_values = torch.full((10,), 2.5)
        log_prob = conditioned.log_prob(test_values)
        assert log_prob.shape == (10,)

    def test_normal_with_dict_params_alternative(self) -> None:
        """Test Normal distribution with dict parameters using different std."""
        base_dist = Normal(0.0, 1.0)

        def parameterizer(
            prior: Float[torch.Tensor, "..."],
            x: Float[torch.Tensor, "seq input_dim"],
        ) -> dict[str, Float[torch.Tensor, "..."]]:
            seq_len = x.shape[0]
            mean = prior.expand(seq_len)
            std = torch.ones(seq_len) * 0.5
            return {"loc": mean, "scale": std}

        likelihood = LikelihoodDistribution(
            base_distribution=base_dist,
            parameterizer=parameterizer,
            input_dim=1,
        )

        prior_sample = torch.tensor(1.5)
        x = torch.randn(5, 1)
        conditioned = likelihood.condition_on_prior_and_input(prior_sample, x)

        assert torch.allclose(conditioned.mean, torch.full((5,), 1.5))
        samples = conditioned.sample()
        assert samples.shape == (5,)

    def test_normal_with_dict_params_specific(self) -> None:
        """Test Normal distribution with dictionary parameters (specific values)."""
        base_dist = Normal(0.0, 1.0)

        # Parameterizer returns dict with 'loc' and 'scale'
        def parameterizer(
            prior: Float[torch.Tensor, "..."],
            x: Float[torch.Tensor, "seq input_dim"],
        ) -> dict[str, Float[torch.Tensor, "..."]]:
            seq_len = x.shape[0]
            return {
                "loc": prior.expand(seq_len),
                "scale": torch.ones(seq_len) * 0.5,
            }

        likelihood = LikelihoodDistribution(
            base_distribution=base_dist,
            parameterizer=parameterizer,
            input_dim=1,
        )

        prior_sample = torch.tensor(-1.0)
        x = torch.randn(3, 1)
        conditioned = likelihood.condition_on_prior_and_input(prior_sample, x)

        # Test properties
        assert torch.allclose(conditioned.mean, torch.full((3,), -1.0))
        assert torch.allclose(conditioned.variance, torch.full((3,), 0.25))  # 0.5^2

    def test_bernoulli_distribution(self) -> None:
        """Test with Bernoulli distribution (different parameter structure)."""
        base_dist = Bernoulli(probs=torch.tensor(0.5))

        # Parameterizer returns probabilities
        def parameterizer(
            prior: Float[torch.Tensor, "..."],
            x: Float[torch.Tensor, "seq input_dim"],
        ) -> dict[str, Float[torch.Tensor, "..."]]:
            seq_len = x.shape[0]
            # Use sigmoid of prior to get valid probabilities
            probs = torch.sigmoid(prior).expand(seq_len)
            return {"probs": probs}

        likelihood = LikelihoodDistribution(
            base_distribution=base_dist,
            parameterizer=parameterizer,
            input_dim=1,
        )

        prior_sample = torch.tensor(0.0)  # sigmoid(0) = 0.5
        x = torch.randn(4, 1)
        conditioned = likelihood.condition_on_prior_and_input(prior_sample, x)

        # Test sampling
        samples = conditioned.sample()
        assert samples.shape == (4,)
        assert torch.all((samples == 0) | (samples == 1))  # Binary values

        # Test log_prob
        test_values = torch.tensor([0.0, 1.0, 0.0, 1.0])
        log_prob = conditioned.log_prob(test_values)
        assert log_prob.shape == (4,)

    def test_gamma_distribution(self) -> None:
        """Test with Gamma distribution (two parameters: concentration, rate)."""
        base_dist = Gamma(concentration=torch.tensor(1.0), rate=torch.tensor(1.0))

        # Parameterizer returns (concentration, rate) as tuple
        def parameterizer(
            prior: Float[torch.Tensor, "..."],
            x: Float[torch.Tensor, "seq input_dim"],
        ) -> dict[str, Float[torch.Tensor, "..."]]:
            seq_len = x.shape[0]
            concentration = torch.abs(prior).expand(seq_len) + 0.1  # Ensure positive
            rate = torch.ones(seq_len)
            return {"concentration": concentration, "rate": rate}

        likelihood = LikelihoodDistribution(
            base_distribution=base_dist,
            parameterizer=parameterizer,
            input_dim=1,
        )

        prior_sample = torch.tensor(2.0)
        x = torch.randn(6, 1)
        conditioned = likelihood.condition_on_prior_and_input(prior_sample, x)

        # Test sampling
        samples = conditioned.sample()
        assert samples.shape == (6,)
        assert torch.all(samples >= 0)  # Gamma samples are non-negative

        # Test properties
        mean = conditioned.mean
        variance = conditioned.variance
        assert mean.shape == (6,)
        assert variance.shape == (6,)

    def test_unconditioned_behavior_errors(self) -> None:
        """Unconditioned likelihood should error when used before conditioning."""
        base_dist = Normal(5.0, 2.0)

        def parameterizer(
            prior: Float[torch.Tensor, "..."],
            x: Float[torch.Tensor, "seq input_dim"],
        ) -> dict[str, Float[torch.Tensor, "..."]]:
            return {"loc": prior, "scale": torch.ones_like(prior)}

        likelihood = LikelihoodDistribution(
            base_distribution=base_dist,
            parameterizer=parameterizer,
            input_dim=1,
        )

        with pytest.raises(RuntimeError, match="unconditioned"):
            _ = likelihood.mean
        with pytest.raises(RuntimeError, match="unconditioned"):
            _ = likelihood.variance
        with pytest.raises(RuntimeError, match="unconditioned"):
            _ = likelihood.sample((100,))

    def test_invalid_param_type(self) -> None:
        """Test that invalid parameter types raise an error."""
        base_dist = Normal(0.0, 1.0)

        # Parameterizer returns invalid type (string)
        def bad_parameterizer(
            prior: Float[torch.Tensor, "..."], x: Float[torch.Tensor, "seq input_dim"]
        ) -> str:
            return "invalid"

        likelihood = LikelihoodDistribution(
            base_distribution=base_dist,
            parameterizer=bad_parameterizer,
            input_dim=1,
        )

        prior_sample = torch.tensor(0.0)
        x = torch.randn(2, 1)
        conditioned = likelihood.condition_on_prior_and_input(prior_sample, x)

        # Should raise ValueError when trying to use invalid params
        with pytest.raises(ValueError, match="Unsupported parameter type"):
            conditioned.sample()

    def test_sample_with_batch_shape(self) -> None:
        """Test sampling with explicit batch shape."""
        base_dist = Normal(0.0, 1.0)

        def parameterizer(
            prior: Float[torch.Tensor, "..."],
            x: Float[torch.Tensor, "seq input_dim"],
        ) -> dict[str, Float[torch.Tensor, "..."]]:
            seq_len = x.shape[0]
            mean = prior.expand(seq_len)
            std = torch.ones(seq_len)
            return {"loc": mean, "scale": std}

        likelihood = LikelihoodDistribution(
            base_distribution=base_dist,
            parameterizer=parameterizer,
            input_dim=1,
        )

        prior_sample = torch.tensor(0.0)
        x = torch.randn(3, 1)
        conditioned = likelihood.condition_on_prior_and_input(prior_sample, x)

        # Sample with batch shape
        samples = conditioned.sample((10,))
        assert samples.shape == (10, 3)

    def test_multivariate_input(self) -> None:
        """Test with multivariate inputs and batched operations."""
        base_dist = Normal(0.0, 1.0)

        def parameterizer(
            prior: Float[torch.Tensor, "..."],
            x: Float[torch.Tensor, "seq input_dim"],
        ) -> dict[str, Float[torch.Tensor, "..."]]:
            # x has shape (seq_len, input_dim)
            seq_len, input_dim = x.shape
            # Use x features to modify parameters
            mean = prior.expand(seq_len) + x.mean(dim=-1)
            std = torch.ones(seq_len) + 0.1 * x.std(dim=-1)
            return {"loc": mean, "scale": std}

        likelihood = LikelihoodDistribution(
            base_distribution=base_dist,
            parameterizer=parameterizer,
            input_dim=5,
        )

        prior_sample = torch.tensor(1.0)
        x = torch.randn(8, 5)
        conditioned = likelihood.condition_on_prior_and_input(prior_sample, x)

        samples = conditioned.sample()
        assert samples.shape == (8,)

        log_prob = conditioned.log_prob(samples)
        assert log_prob.shape == (8,)


class TestDistributionParams:
    """Test the DistributionParams type alias (now dict-only)."""

    def test_type_alias_is_dict_only(self) -> None:
        params: DistributionParams = {
            "loc": torch.tensor(0.0),
            "scale": torch.tensor(1.0),
        }
        assert isinstance(params, dict)
