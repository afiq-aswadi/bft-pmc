"""Configuration for unsupervised (next-token prediction) PFN models."""

from dataclasses import dataclass
from typing import Literal

import torch

from pfn_transformerlens.model.configs.base import BasePFNConfig, validate_bucket_config


@dataclass
class UnsupervisedPFNConfig(BasePFNConfig):
    """Configuration for unsupervised next-token prediction models.

    This mode trains a standard GPT-2 style transformer without x/y interleaving
    or special PFN attention masks. Pure sequence modeling for approximating
    posterior predictive distributions p(x*|x_1:n).

    Supports both discrete and continuous sequences with point or distributional predictions.

    Attributes:
        d_vocab: Vocabulary size (discrete) or number of buckets (continuous distribution).
        input_type: Whether inputs are discrete tokens or continuous values.
        prediction_type: Whether to predict points or distributions.
        bucket_type: Bucketing strategy for continuous distribution predictions.
        mask_type: Must be "gpt2" (causal attention only).
        act_fn: Activation function (default: "gelu").

    Valid combinations:
        - discrete + point: rare, predicts single token index
        - discrete + distribution: standard language modeling (d_vocab = vocabulary size)
        - continuous + point: next-value regression (output shape: batch x seq x 1)
        - continuous + distribution: probabilistic continuous modeling (requires bucket_type)
    """

    d_vocab: int = 2
    input_type: Literal["discrete", "continuous"] = "discrete"
    prediction_type: Literal["point", "distribution"] = "distribution"
    bucket_type: Literal["uniform", "riemann"] | None = None
    bucket_support: Literal["unbounded", "bounded"] = "unbounded"
    y_min: float | None = None
    y_max: float | None = None
    riemann_borders: torch.Tensor | None = None
    mask_type: Literal["gpt2"] = "gpt2"
    act_fn: str = "gelu"

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.mask_type != "gpt2":
            raise ValueError(
                f"Unsupervised mode only supports mask_type='gpt2', got '{self.mask_type}'. "
                "Use SupervisedRegressionPFNConfig or ClassificationPFNConfig for "
                "PFN-style attention."
            )

        if self.d_vocab <= 0:
            raise ValueError(f"d_vocab must be positive, got {self.d_vocab}")
        if self.input_type not in {"discrete", "continuous"}:
            raise ValueError(f"Unknown input_type: {self.input_type}")
        if self.prediction_type not in {"point", "distribution"}:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Validate bucket_type requirements
        if self.input_type == "continuous" and self.prediction_type == "distribution":
            if self.bucket_type is None:
                raise ValueError(
                    "continuous inputs with distribution predictions require bucket_type "
                    "(either 'uniform' or 'riemann')"
                )

            validate_bucket_config(
                self.bucket_type,
                self.y_min,
                self.y_max,
                self.riemann_borders,
                self.d_vocab,
            )
        elif self.bucket_type is not None:
            raise ValueError(
                f"bucket_type should only be set for continuous distribution predictions, "
                f"got input_type={self.input_type}, prediction_type={self.prediction_type}"
            )

        # Set output dimension based on prediction type
        if self.prediction_type == "point":
            self.d_vocab_out = 1

        self.input_dim = 1
