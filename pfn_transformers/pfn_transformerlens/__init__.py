"""PFN TransformerLens: Prior-Fitted Networks with transformer-lens."""

# Core training
# Submodules
from pfn_transformerlens import bayes, checkpointing, configs, generators, wandb_utils
from pfn_transformerlens.model.bucketizer import estimate_riemann_borders
from pfn_transformerlens.model.configs.classification import (
    ClassificationPFNConfig as ClassificationConfig,
)

# Configs (short names)
from pfn_transformerlens.model.configs.regression import (
    SupervisedRegressionPFNConfig as RegressionConfig,
)
from pfn_transformerlens.model.configs.unsupervised import (
    UnsupervisedPFNConfig as UnsupervisedConfig,
)

# Model factory
from pfn_transformerlens.model.PFN import PFNModel as PFN

# Generators (short names)
from pfn_transformerlens.sampler.data_generator import (
    DeterministicFunctionGenerator as DeterministicGenerator,
)
from pfn_transformerlens.sampler.data_generator import (
    FixedDatasetGenerator as DatasetGenerator,
)
from pfn_transformerlens.sampler.data_generator import (
    SupervisedProbabilisticGenerator as BayesianGenerator,
)
from pfn_transformerlens.sampler.data_generator import (
    UnsupervisedProbabilisticGenerator as UnsupervisedBayesian,
)

# Utilities
from pfn_transformerlens.sampler.dataloader import sample_batch
from pfn_transformerlens.sampler.prior_likelihood import (
    DiscreteTaskDistribution as DiscreteTask,
)
from pfn_transformerlens.sampler.prior_likelihood import (
    LikelihoodDistribution as Likelihood,
)
from pfn_transformerlens.sampler.prior_likelihood import (
    PriorDistribution as Prior,
)
from pfn_transformerlens.train import TrainingConfig, train
from pfn_transformerlens.wandb_logger import WandbLogger

# Rename wandb_utils to wandb for cleaner imports
wandb = wandb_utils

__version__ = "0.1.0"

__all__ = [
    # Training
    "train",
    "TrainingConfig",
    "WandbLogger",
    # Model
    "PFN",
    # Configs
    "RegressionConfig",
    "ClassificationConfig",
    "UnsupervisedConfig",
    # Generators
    "DeterministicGenerator",
    "BayesianGenerator",
    "DatasetGenerator",
    "UnsupervisedBayesian",
    # Utilities
    "sample_batch",
    "estimate_riemann_borders",
    # Submodules
    "checkpointing",
    "wandb",
    "wandb_utils",
    "generators",
    "configs",
    "bayes",
    # Prior and Likelihood
    "Prior",
    "Likelihood",
    "DiscreteTask",
]
