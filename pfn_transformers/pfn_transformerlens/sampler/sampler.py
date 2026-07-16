from dataclasses import dataclass, field
from collections.abc import Iterator

import torch
from jaxtyping import Float
from torch.distributions import Distribution
from torch.utils.data import IterableDataset

from pfn_transformerlens.model.configs import BasePFNConfig

from pfn_transformerlens.sampler.data_generator import (
    DataGenerator,
    SupervisedProbabilisticGenerator,
)
from pfn_transformerlens.sampler.prior_likelihood import (
    LikelihoodDistribution,
    PriorDistribution,
)


@dataclass
class SamplerConfig:
    """Config for backward-compatible prior/likelihood sampling."""

    prior: PriorDistribution
    likelihood: LikelihoodDistribution
    model_config: BasePFNConfig
    x_distribution: Distribution = field(
        default_factory=lambda: torch.distributions.Normal(0.0, 1.0)
    )


class Sampler(IterableDataset):
    """Iterable dataset that generates infinite (x, y) sequences.

    Supports both legacy prior/likelihood interface and new DataGenerator protocol.
    """

    def __init__(
        self,
        seq_len: int,
        config: SamplerConfig | None = None,
        internal_batch_size: int = 32,
        data_generator: DataGenerator | None = None,
    ) -> None:
        """Initialize sampler.

        Args:
            seq_len: Length of sequences to generate.
            config: Legacy SamplerConfig (for prior/likelihood).
                Mutually exclusive with data_generator.
            internal_batch_size: Number of sequences to generate per batch (for efficiency).
            data_generator: DataGenerator instance. Mutually exclusive with config.
        """
        if config is not None and data_generator is not None:
            raise ValueError("Provide either config or data_generator, not both")
        if config is None and data_generator is None:
            raise ValueError("Must provide either config or data_generator")

        self.seq_len = seq_len
        self.internal_batch_size = internal_batch_size

        if config is not None:
            self.generator = SupervisedProbabilisticGenerator(
                prior=config.prior,
                likelihood=config.likelihood,
                x_distribution=config.x_distribution,
            )
        else:
            self.generator = data_generator

    def __iter__(
        self,
    ) -> Iterator[
        tuple[Float[torch.Tensor, "seq input_dim"] | None, Float[torch.Tensor, " seq"]]
    ]:
        while True:
            for _ in range(self.internal_batch_size):
                result = self.generator.generate(self.seq_len)
                if isinstance(result, tuple):
                    x, y = result
                    yield x, y
                else:
                    yield None, result
