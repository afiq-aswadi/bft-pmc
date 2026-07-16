"""PFN configuration classes for different task types."""

from pfn_transformerlens.model.configs.base import BasePFNConfig
from pfn_transformerlens.model.configs.classification import ClassificationPFNConfig
from pfn_transformerlens.model.configs.regression import SupervisedRegressionPFNConfig
from pfn_transformerlens.model.configs.unsupervised import UnsupervisedPFNConfig

__all__ = [
    "BasePFNConfig",
    "SupervisedRegressionPFNConfig",
    "ClassificationPFNConfig",
    "UnsupervisedPFNConfig",
]
