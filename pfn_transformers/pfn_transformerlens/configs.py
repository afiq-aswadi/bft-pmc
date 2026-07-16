"""Model configuration exports with intuitive naming."""

from pfn_transformerlens.model.configs.base import BasePFNConfig as BaseConfig
from pfn_transformerlens.model.configs.regression import (
    SupervisedRegressionPFNConfig as Regression,
)
from pfn_transformerlens.model.configs.classification import (
    ClassificationPFNConfig as Classification,
)
from pfn_transformerlens.model.configs.unsupervised import (
    UnsupervisedPFNConfig as Unsupervised,
)

__all__ = [
    "BaseConfig",
    "Regression",
    "Classification",
    "Unsupervised",
]
