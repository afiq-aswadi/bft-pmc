"""Likelihood functions for deterministic function generators.

Task functions define p(y|x,w) - the likelihood in the Bayesian framework.
"""

import torch


def linear_regression(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Linear regression: y = w^T x.

    Args:
        x: Input tensor of shape (..., input_dim)
        w: Weight tensor of shape (input_dim,)

    Returns:
        Output tensor of shape (...)
    """
    return (w * x).sum(dim=-1)
