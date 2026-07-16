"""Configuration for classification PFN models."""

from dataclasses import dataclass
from typing import Literal

from pfn_transformerlens.model.configs.base import BasePFNConfig


@dataclass
class ClassificationPFNConfig(BasePFNConfig):
    """Configuration for multi-class classification PFN models.

    Uses discrete categorical outputs (no bucketing). The model outputs
    logits over num_classes, trained with cross-entropy loss.

    Attributes:
        num_classes: Number of classification classes.
        y_type: Type of y values ("continuous" or "categorical").
        mask_type: Attention mask type ("autoregressive-pfn" or "gpt2").
    """

    num_classes: int = 2
    y_type: Literal["continuous", "categorical"] = "continuous"
    mask_type: Literal["autoregressive-pfn", "gpt2"] = "autoregressive-pfn"

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        if self.y_type not in {"continuous", "categorical"}:
            raise ValueError(f"Unknown y_type: {self.y_type}")
        if self.mask_type not in {"autoregressive-pfn", "gpt2"}:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

        self.d_vocab_out = self.num_classes
