"""Data generation for the Dirichlet-Multinomial (balls and urns) setting.

Prior: Dirichlet(alpha) over probability simplexes in R^V
Likelihood: Categorical(theta) where theta ~ Dirichlet(alpha)
Sequence: x_1, ..., x_n ~ i.i.d. Categorical(theta)
"""

import torch
from jaxtyping import Float

from pfn_transformerlens import DiscreteTask, Likelihood, Prior, UnsupervisedBayesian


def _categorical_parameterizer(
    theta: Float[torch.Tensor, " V"], x: Float[torch.Tensor, "seq 1"]
) -> dict[str, Float[torch.Tensor, "seq V"]]:
    seq_len = x.shape[0]
    return {"probs": theta.unsqueeze(0).expand(seq_len, -1)}


def _make_likelihood(vocab_size: int) -> Likelihood:
    return Likelihood(
        base_distribution=torch.distributions.Categorical(
            probs=torch.ones(vocab_size) / vocab_size
        ),
        parameterizer=_categorical_parameterizer,
        input_dim=1,
    )


def make_bau_generator(
    alpha: Float[torch.Tensor, " V"],
) -> UnsupervisedBayesian:
    """Create a generator with true Dirichlet prior (for eval).

    Args:
        alpha: Dirichlet concentration parameters, shape (V,).
    """
    prior = Prior(base_distribution=torch.distributions.Dirichlet(alpha))
    return UnsupervisedBayesian(prior=prior, likelihood=_make_likelihood(len(alpha)))


def make_discrete_bau_generator(
    alpha: Float[torch.Tensor, " V"],
    num_tasks: int,
) -> tuple[UnsupervisedBayesian, Float[torch.Tensor, "M V"]]:
    """Create a generator with discrete prior (M fixed simplexes for training).

    Args:
        alpha: Dirichlet concentration parameters, shape (V,).
        num_tasks: Number of fixed simplexes to sample.

    Returns:
        Tuple of (generator, theta_pool) where theta_pool has shape (num_tasks, V).
    """
    theta_pool = torch.distributions.Dirichlet(alpha).sample((num_tasks,))
    prior = Prior(base_distribution=DiscreteTask(theta_pool))
    return UnsupervisedBayesian(
        prior=prior, likelihood=_make_likelihood(len(alpha))
    ), theta_pool


def make_generator_from_pool(
    theta_pool: Float[torch.Tensor, "M V"],
) -> UnsupervisedBayesian:
    """Create a generator from an existing theta pool (e.g. loaded from a checkpoint)."""
    vocab_size = theta_pool.shape[1]
    prior = Prior(base_distribution=DiscreteTask(theta_pool))
    return UnsupervisedBayesian(prior=prior, likelihood=_make_likelihood(vocab_size))


class BOSGenerator:
    """Wraps an UnsupervisedBayesian generator to prepend a BOS token.

    Generates seq_len - 1 data tokens and prepends the BOS token, so the
    total sequence length is seq_len.
    """

    def __init__(self, base: UnsupervisedBayesian, bos_token: int) -> None:
        self.base = base
        self.bos_token = bos_token
        self.input_dim = base.input_dim
        # expose prior for checkpoint saving
        self.prior = base.prior

    def generate(self, seq_len: int) -> Float[torch.Tensor, " seq"]:
        tokens = self.base.generate(seq_len - 1)
        bos = torch.tensor([self.bos_token], dtype=tokens.dtype, device=tokens.device)
        return torch.cat([bos, tokens])
