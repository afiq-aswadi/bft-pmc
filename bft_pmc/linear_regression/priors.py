"""Discrete prior distribution for memorization experiments.

Implements a discrete uniform distribution over a finite set of weight vectors,
compatible with pfn_transformerlens.DeterministicGenerator.
"""

import torch
from torch.distributions import Distribution


class DiscretePrior(Distribution):
    """Uniform distribution over a fixed finite set of weight vectors.

    Samples uniformly with replacement from num_tasks weight vectors,
    each sampled once at initialization from N(0, I).

    Compatible with DeterministicGenerator for studying memorization vs
    generalization: models trained on few tasks should memorize the finite set,
    while models trained on many tasks should generalize.
    """

    # No constructor arguments to validate; silence torch arg_constraints warning.
    arg_constraints: dict = {}

    def __init__(
        self,
        task_size: int,
        num_tasks: int | None = None,
        device: str = "cpu",
        tasks: torch.Tensor | None = None,
        validate_args: bool | None = None,
    ):
        """Initialize discrete prior over num_tasks weight vectors.

        Args:
            task_size: Dimensionality of weight vectors
            num_tasks: Number of distinct weight vectors in the finite set
            device: Device to store weight vectors on
            tasks: Optional fixed task pool tensor of shape (num_tasks, task_size)
            validate_args: Whether to validate distribution arguments
        """
        if tasks is not None:
            if tasks.dim() != 2 or tasks.shape[1] != task_size:
                raise ValueError(
                    f"tasks must have shape (num_tasks, {task_size}), got {tasks.shape}"
                )
            self.tasks = tasks.to(device)
            self.num_tasks = tasks.shape[0]
        else:
            if num_tasks is None:
                raise ValueError("num_tasks must be provided when tasks is None")
            self.num_tasks = num_tasks
            self.tasks = torch.normal(
                mean=0.0,
                std=1.0,
                size=(num_tasks, task_size),
                device=device,
            )

        self.task_size = task_size
        self._device = device

        batch_shape = torch.Size()
        event_shape = torch.Size([task_size])
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(
        self,
        sample_shape: torch.Size | tuple[int, ...] | list[int] = torch.Size(),
    ) -> torch.Tensor:
        """Sample weight vectors uniformly with replacement.

        Args:
            sample_shape: Shape of samples to return

        Returns:
            Tensor of shape (*sample_shape, task_size)
        """
        sample_shape = torch.Size(sample_shape)
        indices = torch.randint(
            high=self.num_tasks,
            size=sample_shape if sample_shape else torch.Size([1]),
            device=self._device,
        )

        if sample_shape:
            return self.tasks[indices]

        # Unbatched draw should return a single weight vector, not shape (1, task_size)
        return self.tasks[indices[0]]

    @property
    def device(self) -> torch.device:
        """Device this distribution is on."""
        return self.tasks.device

    def to(self, device: str | torch.device) -> "DiscretePrior":
        """Move distribution to different device."""
        self.tasks = self.tasks.to(device)
        self._device = str(device)
        return self
