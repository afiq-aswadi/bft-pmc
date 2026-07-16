"""Data generator exports with intuitive naming."""

from pfn_transformerlens.sampler.data_generator import (
    DeterministicFunctionGenerator as Deterministic,
    FixedDatasetGenerator as Dataset,
    SupervisedProbabilisticGenerator as Bayesian,
    UnsupervisedProbabilisticGenerator as UnsupervisedBayesian,
    SupervisedDataGenerator,
    UnsupervisedDataGenerator,
    DataGenerator,
)

__all__ = [
    "Deterministic",
    "Bayesian",
    "Dataset",
    "UnsupervisedBayesian",
    "SupervisedDataGenerator",
    "UnsupervisedDataGenerator",
    "DataGenerator",
]
