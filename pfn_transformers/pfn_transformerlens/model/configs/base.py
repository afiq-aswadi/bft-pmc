"""Base configuration class for PFN models."""

from dataclasses import dataclass

import torch
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def validate_bucket_config(
    bucket_type: str,
    y_min: float | None,
    y_max: float | None,
    riemann_borders: torch.Tensor | None,
    d_vocab: int,
) -> None:
    """Validate parameters shared by continuous distribution bucketizers."""
    if bucket_type == "uniform":
        if y_min is None or y_max is None:
            raise ValueError("Uniform bucketing requires both y_min and y_max")
        if y_min >= y_max:
            raise ValueError(f"y_min ({y_min}) must be less than y_max ({y_max})")
        return

    if bucket_type == "riemann":
        if riemann_borders is None:
            raise ValueError("Riemann bucketing requires riemann_borders")
        if riemann_borders.ndim != 1:
            raise ValueError(
                f"riemann_borders must be one-dimensional, got {riemann_borders.shape}"
            )
        expected_length = d_vocab + 1
        if len(riemann_borders) != expected_length:
            raise ValueError(
                f"riemann_borders must have length {expected_length}, "
                f"got {len(riemann_borders)}"
            )
        return

    raise ValueError(f"Unknown bucket_type: {bucket_type!r}")


@dataclass
class BasePFNConfig(HookedTransformerConfig):
    """Base configuration for all PFN model types.

    Contains common transformer architecture parameters shared across
    regression, classification, and unsupervised models.

    Attributes:
        input_dim: Dimension of input features.
        use_pos_emb: Whether to use position embeddings.
        normalization_type: Type of normalization (default: "LN").
    """

    input_dim: int = 16
    use_pos_emb: bool = True
    normalization_type: str = "LN"

    def __post_init__(self) -> None:
        super().__post_init__()
