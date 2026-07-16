"""Configuration for supervised regression PFN models."""

from dataclasses import dataclass
from typing import Literal

import torch

from pfn_transformerlens.model.configs.base import BasePFNConfig, validate_bucket_config


@dataclass
class SupervisedRegressionPFNConfig(BasePFNConfig):
    """Configuration for supervised regression PFN models.

    Supports two prediction modes:
    - "distribution": Predicts probability distribution over buckets
    - "point": Direct scalar regression (no bucketing)

    For distribution predictions, supports two bucketing strategies:
    - "uniform": Evenly-spaced buckets (requires y_min, y_max)
    - "riemann": Quantile-based buckets (requires riemann_borders)

    Attributes:
        mask_type: Attention mask type ("autoregressive-pfn" or "gpt2").
        prediction_type: Output type ("distribution" or "point").
        bucket_type: Bucketing strategy (only for distribution mode).
        bucket_support: Support type ("unbounded" or "bounded").
        y_min: Minimum value for uniform buckets.
        y_max: Maximum value for uniform buckets.
        riemann_borders: Precomputed borders for riemann buckets.
    """

    mask_type: Literal["autoregressive-pfn", "gpt2"] = "autoregressive-pfn"
    prediction_type: Literal["distribution", "point"] = "distribution"
    bucket_type: Literal["uniform", "riemann"] | None = None
    bucket_support: Literal["unbounded", "bounded"] = "unbounded"
    y_min: float | None = None
    y_max: float | None = None
    riemann_borders: torch.Tensor | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.mask_type not in {"autoregressive-pfn", "gpt2"}:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")
        if self.prediction_type == "point":
            self.d_vocab_out = 1
            if (
                self.bucket_type is not None
                or self.y_min is not None
                or self.y_max is not None
            ):
                raise ValueError(
                    "Point prediction mode does not use bucketing. "
                    "Set prediction_type='distribution' to use buckets."
                )
        elif self.prediction_type == "distribution":
            if self.bucket_type is None:
                raise ValueError(
                    "Distribution prediction mode requires bucket_type to be specified. "
                    "Use 'uniform' or 'riemann'."
                )
            validate_bucket_config(
                self.bucket_type,
                self.y_min,
                self.y_max,
                self.riemann_borders,
                self.d_vocab,
            )
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
