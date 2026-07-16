"""Data generation abstractions for PFN training.

Provides a unified interface for different data generation strategies,
from probabilistic (prior/likelihood) to deterministic functions to fixed datasets.

Example usage:
    Single sequence generation:
        >>> gen = DeterministicFunctionGenerator(...)
        >>> x, y = gen.generate(seq_len=64)  # returns single sequence

    Batch generation (use the standalone sample_batch function):
        >>> from pfn_transformerlens.sampler.dataloader import sample_batch
        >>> x_batch, y_batch = sample_batch(gen, batch_size=32, seq_len=64)
        >>> # x_batch shape: (32, 64, input_dim), y_batch shape: (32, 64)

    Note: Generators do NOT have a sample_batch method. Use the standalone
    sample_batch function from dataloader module for batched sampling.
"""

from typing import Any, Callable, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch.distributions import Distribution
from torch.utils._pytree import tree_map

from pfn_transformerlens.sampler.prior_likelihood import (
    LikelihoodDistribution,
    PriorDistribution,
)


@runtime_checkable
class SupervisedDataGenerator(Protocol):
    """Protocol for generating (x, y) training sequences for supervised learning.

    Any supervised data generator must implement this interface to be compatible
    with the PFN training pipeline.

    Example usage:
        Single sequence generation:
            >>> gen = DeterministicFunctionGenerator(...)
            >>> x, y = gen.generate(seq_len=64)  # returns single sequence

        Batch generation (use the standalone sample_batch function):
            >>> from pfn_transformerlens.sampler.dataloader import sample_batch
            >>> x_batch, y_batch = sample_batch(gen, batch_size=32, seq_len=64)
            >>> # x_batch shape: (32, 64, input_dim), y_batch shape: (32, 64)

        Note: Generators do NOT have a sample_batch method. Use the standalone
        sample_batch function from dataloader module for batched sampling.
    """

    input_dim: int

    def generate(
        self, seq_len: int
    ) -> tuple[Float[torch.Tensor, "seq input_dim"], Float[torch.Tensor, "seq"]]:
        """Generate one sequence of (x, y) pairs.

        Args:
            seq_len: Length of sequence to generate.

        Returns:
            Tuple of (x, y) where:
                x: Input features of shape (seq_len, input_dim)
                y: Target values of shape (seq_len,)
        """
        ...


@runtime_checkable
class UnsupervisedDataGenerator(Protocol):
    """Protocol for generating y sequences for unsupervised learning.

    Any unsupervised data generator must implement this interface to be compatible
    with the PFN training pipeline. No x values are needed for unsupervised models.


    Example usage:
        Single sequence generation:
            >>> gen = UnsupervisedProbabilisticGenerator(...)
            >>> y = gen.generate(seq_len=64)  # returns single sequence

        Batch generation (use the standalone sample_batch function):
            >>> from pfn_transformerlens.sampler.dataloader import sample_batch
            >>> x_batch, y_batch = sample_batch(gen, batch_size=32, seq_len=64)
            >>> # x_batch is None, y_batch shape: (32, 64)

        Note: Generators do NOT have a sample_batch method. Use the standalone
        sample_batch function from dataloader module for batched sampling.
    """

    input_dim: int

    def generate(self, seq_len: int) -> Float[torch.Tensor, "seq"]:
        """Generate one sequence of observations.

        Args:
            seq_len: Length of sequence to generate.

        Returns:
            y: Observation values of shape (seq_len,)
        """
        ...


DataGenerator = SupervisedDataGenerator | UnsupervisedDataGenerator


@runtime_checkable
class ParameterizedDataGenerator(Protocol):
    """Generator that exposes sampled parameters for baseline comparisons.

    This optional protocol allows generators to return the parameters they sampled,
    which is useful for computing baseline predictions or analyzing model behavior
    relative to the true data-generating parameters.
    """

    input_dim: int

    def generate_with_params(
        self, seq_len: int
    ) -> tuple[
        tuple[Float[torch.Tensor, "seq input_dim"], Float[torch.Tensor, "seq"]]
        | Float[torch.Tensor, "seq"],
        dict[str, Any],
    ]:
        """Generate sequence and return the parameters used.

        Args:
            seq_len: Length of sequence to generate.

        Returns:
            Tuple of (data, params) where:
                data: Either (x, y) tuple for supervised or just y for unsupervised
                params: Dictionary of parameters like {"theta": ...}
        """
        ...


class SupervisedProbabilisticGenerator:
    """Generates supervised (x, y) data from prior/likelihood distributions.

    Implements the Bayesian workflow for supervised learning:
    sample task from prior, then generate data conditioned on that task.
    """

    def __init__(
        self,
        prior: PriorDistribution,
        likelihood: LikelihoodDistribution,
        x_distribution: Distribution | None = None,
    ):
        """Initialize supervised probabilistic generator.

        Args:
            prior: Distribution over task parameters θ.
            likelihood: Distribution over y given θ and x.
            x_distribution: Distribution over input features x. Defaults to N(0,1).
        """
        self.prior = prior
        self.likelihood = likelihood
        self.x_distribution = x_distribution or torch.distributions.Normal(0.0, 1.0)
        self.input_dim = likelihood.input_dim

    def generate(
        self, seq_len: int
    ) -> tuple[Float[torch.Tensor, "seq input_dim"], Float[torch.Tensor, "seq"]]:
        """Generate one supervised sequence.

        Returns:
            Tuple of (x, y) where x is inputs and y is targets.
        """
        theta = self.prior.sample(torch.Size([]))
        x = self.x_distribution.sample((seq_len, self.input_dim))
        conditioned_likelihood = self.likelihood.condition_on_prior_and_input(theta, x)
        y = conditioned_likelihood.sample()
        return x, y

    def generate_with_params(
        self, seq_len: int
    ) -> tuple[
        tuple[Float[torch.Tensor, "seq input_dim"], Float[torch.Tensor, "seq"]],
        dict[str, Any],
    ]:
        """Generate sequence and return sampled theta parameter.

        Returns:
            Tuple of ((x, y), {"theta": theta})
        """
        theta = self.prior.sample(torch.Size([]))
        x = self.x_distribution.sample((seq_len, self.input_dim))
        conditioned_likelihood = self.likelihood.condition_on_prior_and_input(theta, x)
        y = conditioned_likelihood.sample()
        return (x, y), {"theta": theta}


class UnsupervisedProbabilisticGenerator:
    """Generates unsupervised y sequences from prior/likelihood distributions.

    Implements the Bayesian workflow for unsupervised learning:
    sample task from prior, then generate observations conditioned on that task.
    No x values are generated - only y observations.
    """

    def __init__(
        self,
        prior: PriorDistribution,
        likelihood: LikelihoodDistribution,
    ):
        """Initialize unsupervised probabilistic generator.

        Args:
            prior: Distribution over task parameters θ.
            likelihood: Distribution over y given θ. The likelihood's parameterizer
                should only depend on theta, not x (x will be dummy zeros).
        """
        self.prior = prior
        self.likelihood = likelihood
        self.input_dim = likelihood.input_dim

    def generate(self, seq_len: int) -> Float[torch.Tensor, "seq"]:
        """Generate one unsupervised sequence.

        Returns:
            Tensor y of observations (no x).
        """
        theta = self.prior.sample(torch.Size([]))
        # the likelihood API always accepts x, so unsupervised generators pass zeros
        x = torch.zeros((seq_len, self.input_dim))
        conditioned_likelihood = self.likelihood.condition_on_prior_and_input(theta, x)
        y = conditioned_likelihood.sample()
        return y

    def generate_with_params(
        self, seq_len: int
    ) -> tuple[Float[torch.Tensor, "seq"], dict[str, Any]]:
        """Generate sequence and return sampled theta parameter.

        Returns:
            Tuple of (y, {"theta": theta})
        """
        theta = self.prior.sample(torch.Size([]))
        x = torch.zeros((seq_len, self.input_dim))
        conditioned_likelihood = self.likelihood.condition_on_prior_and_input(theta, x)
        y = conditioned_likelihood.sample()
        return y, {"theta": theta}


# Backward compatibility alias
ProbabilisticGenerator = SupervisedProbabilisticGenerator


class DeterministicFunctionGenerator:
    """Generates data from deterministic function with optional noise.

    Supports both noisy and noiseless regression:
        y = f(x; θ) + ε  where θ ~ prior, ε ~ N(0, noise_std)

    For noiseless case, set noise_std=None.
    """

    def __init__(
        self,
        prior: Distribution,
        function: Callable[
            [Float[torch.Tensor, "seq input_dim"], Any],
            Float[torch.Tensor, "seq"],
        ],
        input_dim: int,
        noise_std: float | None = 0.0,
        x_distribution: Distribution = torch.distributions.Normal(0.0, 1.0),
        device: str | torch.device | None = None,
    ):
        """Initialize deterministic function generator.

        The prior can return any structure (scalar, tensor, tuple, dict), and the function
        must accept that structure as its second argument.

        Args:
            prior: Distribution over function parameters. Can return any structure:
                - Scalar tensor: prior.sample() -> theta
                - Tuple: prior.sample() -> (alpha, beta)
                - Dict: prior.sample() -> {"mean": ..., "scale": ...}
            function: Deterministic function with signature f(x, params) -> y where:
                - x: Input tensor of shape (seq_len, input_dim)
                - params: Parameters sampled from prior (matches prior.sample() output)
                - Returns: y of shape (seq_len,)
            input_dim: Dimension of input features.
            noise_std: Standard deviation of Gaussian noise. Set to None for noiseless.
            x_distribution: Distribution for sampling x. Defaults to N(0,1).
            device: Optional device to place sampled tensors on. Defaults to prior device.

        Examples:
            Single parameter:
                prior = torch.distributions.Normal(0, 1)  # returns scalar theta
                function = lambda x, theta: (x * theta).sum(dim=-1)

            Multiple parameters (tuple):
                prior = ...  # returns (alpha, beta)
                function = lambda x, params: (x * params[0]).sum(dim=-1) + params[1]

            Multiple parameters (dict):
                prior = ...  # returns {"slope": ..., "intercept": ...}
                function = lambda x, params: (x * params["slope"]).sum(dim=-1) + params["intercept"]
        """
        self.prior = prior
        self.function = function
        self.input_dim = input_dim
        self.noise_std = noise_std
        self.x_distribution = x_distribution
        self.device = torch.device(device) if device is not None else None

    def generate(
        self, seq_len: int
    ) -> tuple[Float[torch.Tensor, "seq input_dim"], Float[torch.Tensor, "seq"]]:
        theta = self.prior.sample()
        prior_device = getattr(self.prior, "device", None)
        if prior_device is not None and not isinstance(prior_device, torch.device):
            prior_device = torch.device(prior_device)
        target_device = self.device or prior_device

        if target_device is not None:
            theta = tree_map(
                lambda t: t.to(target_device) if torch.is_tensor(t) else t, theta
            )

        x = self.x_distribution.sample((seq_len, self.input_dim))
        if isinstance(x, torch.Tensor) and target_device is not None:
            x = x.to(target_device)

        y = self.function(x, theta)

        if self.noise_std is not None and self.noise_std > 0:
            noise = torch.randn_like(y) * self.noise_std
            y = y + noise

        return x, y

    def generate_with_params(
        self, seq_len: int
    ) -> tuple[
        tuple[Float[torch.Tensor, "seq input_dim"], Float[torch.Tensor, "seq"]],
        dict[str, Any],
    ]:
        """Generate sequence and return sampled parameters.

        Returns:
            Tuple of ((x, y), {"params": params}) where params matches prior.sample() output
        """
        params = self.prior.sample()
        prior_device = getattr(self.prior, "device", None)
        if prior_device is not None and not isinstance(prior_device, torch.device):
            prior_device = torch.device(prior_device)
        target_device = self.device or prior_device

        if target_device is not None:
            params = tree_map(
                lambda t: t.to(target_device) if torch.is_tensor(t) else t, params
            )

        x = self.x_distribution.sample((seq_len, self.input_dim))
        if isinstance(x, torch.Tensor) and target_device is not None:
            x = x.to(target_device)

        y = self.function(x, params)

        if self.noise_std is not None and self.noise_std > 0:
            noise = torch.randn_like(y) * self.noise_std
            y = y + noise

        return (x, y), {"params": params}


class FixedDatasetGenerator:
    """Samples sequences from a fixed dataset.

    Useful for supervised learning from static data rather than
    generative distributions.
    """

    def __init__(
        self,
        x_data: Float[torch.Tensor, "N input_dim"],
        y_data: Float[torch.Tensor, "N"],
        sequential: bool = False,
    ):
        """Initialize fixed dataset generator.

        Args:
            x_data: Input features, shape (N, input_dim).
            y_data: Target values, shape (N,).
            sequential: If True, sample consecutive subsequences.
                If False, sample random indices.
        """
        if len(x_data) != len(y_data):
            raise ValueError(
                f"x_data and y_data must have same length, got {len(x_data)} and {len(y_data)}"
            )
        if len(x_data) == 0:
            raise ValueError("Dataset cannot be empty")

        self.x_data = x_data
        self.y_data = y_data
        self.input_dim = x_data.shape[1]
        self.sequential = sequential
        self.dataset_size = len(x_data)

    def generate(
        self, seq_len: int
    ) -> tuple[Float[torch.Tensor, "seq input_dim"], Float[torch.Tensor, "seq"]]:
        if self.sequential:
            start_idx = torch.randint(
                0, max(1, self.dataset_size - seq_len + 1), (1,)
            ).item()
            end_idx = min(start_idx + seq_len, self.dataset_size)
            indices = torch.arange(start_idx, end_idx)
            if len(indices) < seq_len:
                indices = torch.cat(
                    [
                        indices,
                        torch.randint(0, self.dataset_size, (seq_len - len(indices),)),
                    ]
                )
        else:
            indices = torch.randint(0, self.dataset_size, (seq_len,))

        return self.x_data[indices], self.y_data[indices]
