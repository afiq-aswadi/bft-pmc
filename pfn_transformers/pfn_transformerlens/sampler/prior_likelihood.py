"""
Prior and Likelihood distribution classes for PFN sampling.
"""

from collections.abc import Callable

import torch
from jaxtyping import Float
from torch.distributions import Distribution

# Type alias for distribution parameters (single, consistent type)
DistributionParams = dict[str, Float[torch.Tensor, "..."]]


class DiscreteTaskDistribution(Distribution):
    """
    Distribution over discrete tasks.
    When sampled, returns the actual task values instead of indices.
    """

    arg_constraints = {}

    def __init__(self, tasks: Float[torch.Tensor, "..."]):
        """
        Initialize with a tensor of task values.

        Args:
            tasks: Tensor of shape (num_tasks,) or (num_tasks, dim) containing task values
        """
        if tasks.ndim not in {1, 2} or tasks.shape[0] < 1:
            raise ValueError("tasks must have shape (num_tasks,) or (num_tasks, dim).")
        if not tasks.is_floating_point() or not torch.isfinite(tasks).all():
            raise ValueError("tasks must be a finite floating-point tensor.")
        self.tasks = tasks
        self.num_tasks = tasks.shape[0]

        # Create uniform categorical distribution over task indices
        probs = torch.ones(self.num_tasks, device=tasks.device) / self.num_tasks
        self.categorical = torch.distributions.Categorical(probs=probs)

        # Set event_shape based on task dimensionality
        if tasks.dim() == 1:
            event_shape = torch.Size([])
        else:
            event_shape = torch.Size([tasks.shape[1]])

        super().__init__(
            batch_shape=torch.Size([]),
            event_shape=event_shape,
            validate_args=False,
        )

    def sample(
        self, sample_shape: torch.Size | tuple[int, ...] = torch.Size()
    ) -> Float[torch.Tensor, "..."]:
        """
        Sample task values by first sampling indices then looking up tasks.

        Args:
            sample_shape: Shape of the sample to draw. Defaults to empty tuple (single sample).

        Returns:
            Sampled task values with shape ``sample_shape + event_shape``.

        Example:
            >>> tasks = torch.tensor([1.0, 2.0, 3.0])
            >>> dist = DiscreteTaskDistribution(tasks)
            >>> # Single sample
            >>> sample = dist.sample()
            >>> # Shape: ()
            >>> # Multiple samples
            >>> samples = dist.sample((5,))
            >>> # Shape: (5,)
            >>> # Batch of samples
            >>> batch_samples = dist.sample((3, 4))
            >>> # Shape: (3, 4)
            >>> # For multi-dimensional tasks
            >>> tasks_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            >>> dist_2d = DiscreteTaskDistribution(tasks_2d)
            >>> sample_2d = dist_2d.sample((10,))
            >>> # Shape: (10, 2)
        """
        indices = self.categorical.sample(sample_shape)
        return self.tasks[indices]

    def log_prob(self, value: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        """
        Compute log probability of a task value.
        Returns uniform log probability if value matches any task, -inf otherwise.
        """
        # For each value, check if it matches any task
        # This uses approximate equality for floating point comparison
        if self.tasks.dim() == 1:
            matches = torch.isclose(value.unsqueeze(-1), self.tasks, rtol=1e-5).any(
                dim=-1
            )
        else:
            matches = (
                torch.isclose(value.unsqueeze(-2), self.tasks, rtol=1e-5)
                .all(dim=-1)
                .any(dim=-1)
            )
        log_prob_uniform = -torch.log(
            torch.tensor(
                self.num_tasks,
                dtype=self.tasks.dtype,
                device=value.device,
            )
        )
        negative_infinity = torch.full(
            matches.shape,
            float("-inf"),
            dtype=self.tasks.dtype,
            device=value.device,
        )
        return torch.where(matches, log_prob_uniform, negative_infinity)


class PriorDistribution(Distribution):
    """
    Vanilla prior distribution class.
    This is a simple wrapper around standard PyTorch distributions.
    """

    arg_constraints = {}  # No constraints as we delegate to base distribution

    def __init__(self, base_distribution: Distribution):
        """
        Initialize with a base PyTorch distribution.

        Args:
            base_distribution: The underlying PyTorch distribution
        """
        self.base_distribution = base_distribution
        super().__init__(
            batch_shape=base_distribution.batch_shape,
            event_shape=base_distribution.event_shape,
            validate_args=False,
        )

    def sample(
        self, sample_shape: torch.Size | tuple[int, ...] = torch.Size()
    ) -> Float[torch.Tensor, "..."]:
        """Sample from the prior distribution."""
        return self.base_distribution.sample(sample_shape)

    def log_prob(self, value: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        """Compute log probability of the value under the prior."""
        return self.base_distribution.log_prob(value)

    @property
    def mean(self) -> Float[torch.Tensor, "..."]:
        """Mean of the prior distribution."""
        return self.base_distribution.mean

    @property
    def variance(self) -> Float[torch.Tensor, "..."]:
        """Variance of the prior distribution."""
        return self.base_distribution.variance


class LikelihoodDistribution(Distribution):
    """
    Likelihood distribution that can be parameterized by a function of the prior.
    This allows the likelihood to depend on the sampled prior parameters.
    """

    arg_constraints = {}

    def __init__(
        self,
        base_distribution: Distribution,
        parameterizer: Callable[
            [Float[torch.Tensor, "..."], Float[torch.Tensor, "seq input_dim"]],
            DistributionParams,
        ],
        input_dim: int,
    ):
        """
        Initialize the likelihood distribution.

        Args:
            base_distribution: Base distribution to use (e.g., Normal)
            parameterizer: Function that takes (prior_sample, x) and returns
                a dict of distribution parameters (keyword args).
                allows users to provide functions without subclassing.
            input_dim: Dimension of input features x
        """
        self.base_distribution = base_distribution
        self.distribution_class = type(base_distribution)
        self.parameterizer = parameterizer
        self.input_dim = input_dim
        self._current_params = None

        # Initialize with default parameters
        super().__init__(
            batch_shape=base_distribution.batch_shape,
            event_shape=base_distribution.event_shape,
            validate_args=False,
        )

    def _create_distribution_from_params(
        self, params: DistributionParams
    ) -> Distribution:
        """
        Create a distribution instance from parameters.

        Args:
            params: Dictionary of keyword arguments to construct the distribution.

        Returns:
            Distribution instance
        """
        if isinstance(params, dict):
            return self.distribution_class(**params)
        raise ValueError(
            f"Unsupported parameter type: {type(params)}. Expected dict[str, torch.Tensor]."
        )

    def condition_on_prior_and_input(
        self,
        prior_sample: Float[torch.Tensor, "..."],
        x: Float[torch.Tensor, "seq input_dim"],
    ) -> "LikelihoodDistribution":
        """
        Create a new likelihood distribution conditioned on prior sample and input.

        Args:
            prior_sample: Sample from the prior distribution
            x: Input features

        Returns:
            New LikelihoodDistribution with updated parameters
        """
        # Get the parameters for this specific prior sample and input
        params = self.parameterizer(prior_sample, x)

        # Create a new instance with the conditioned parameters
        new_dist = LikelihoodDistribution(
            base_distribution=self.base_distribution,
            parameterizer=self.parameterizer,
            input_dim=self.input_dim,
        )
        new_dist._current_params = params
        return new_dist

    def sample(
        self, sample_shape: torch.Size | tuple[int, ...] = torch.Size()
    ) -> Float[torch.Tensor, "..."]:
        """Sample from the likelihood distribution."""
        if self._current_params is None:
            raise RuntimeError(
                "LikelihoodDistribution is unconditioned. Call condition_on_prior_and_input first."
            )
        # Create a new distribution with current parameters and sample
        conditioned_dist = self._create_distribution_from_params(self._current_params)
        sample = conditioned_dist.sample(sample_shape)

        # ensure discrete distributions return integer tensors
        if isinstance(
            conditioned_dist,
            (torch.distributions.Bernoulli, torch.distributions.Categorical),
        ):
            sample = sample.long()

        return sample

    def log_prob(self, value: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        """Compute log probability under the current parameters."""
        if self._current_params is None:
            raise RuntimeError(
                "LikelihoodDistribution is unconditioned. Call condition_on_prior_and_input first."
            )
        conditioned_dist = self._create_distribution_from_params(self._current_params)
        return conditioned_dist.log_prob(value)

    @property
    def mean(self) -> Float[torch.Tensor, "..."]:
        """Mean of the current distribution."""
        if self._current_params is None:
            raise RuntimeError(
                "LikelihoodDistribution is unconditioned. Call condition_on_prior_and_input first."
            )
        conditioned_dist = self._create_distribution_from_params(self._current_params)
        return conditioned_dist.mean

    @property
    def variance(self) -> Float[torch.Tensor, "..."]:
        """Variance of the current distribution."""
        if self._current_params is None:
            raise RuntimeError(
                "LikelihoodDistribution is unconditioned. Call condition_on_prior_and_input first."
            )
        conditioned_dist = self._create_distribution_from_params(self._current_params)
        return conditioned_dist.variance

    def __repr__(self) -> str:
        """String representation showing current parameters."""
        dist_name = self.distribution_class.__name__
        if self._current_params is None:
            params_str = "unconditioned"
        else:
            # Format parameters nicely
            params_items = []
            for key, value in self._current_params.items():
                shape = tuple(value.shape)
                params_items.append(f"{key}={shape}")
            params_str = ", ".join(params_items)
        return (
            f"LikelihoodDistribution({dist_name}, "
            f"input_dim={self.input_dim}, params=[{params_str}])"
        )
