# pfn-transformerlens API Reference

> AI-generated. This file is intended for AI consumption (coding agents, RAG, etc.).

**Package:** `pfn_transformerlens`
**Purpose:** Train Prior-Fitted Networks (PFNs) -- transformers that perform in-context Bayesian inference without parameter updates. Built on TransformerLens.

---

## Top-Level Imports

```python
from pfn_transformerlens import (
    # training
    train, TrainingConfig, WandbLogger,
    # model factory
    PFN,
    # configs
    RegressionConfig, ClassificationConfig, UnsupervisedConfig,
    # generators
    DeterministicGenerator, BayesianGenerator, DatasetGenerator, UnsupervisedBayesian,
    # bayesian primitives
    Prior, Likelihood, DiscreteTask,
    # utilities
    sample_batch, estimate_riemann_borders,
    # package modules
    checkpointing, wandb_utils, generators, configs, bayes,
)
```

Short-name aliases for module imports:

```python
from pfn_transformerlens.generators import Deterministic, Bayesian, Dataset, UnsupervisedBayesian
from pfn_transformerlens.configs import Regression, Classification, Unsupervised
from pfn_transformerlens.bayes import Prior, Likelihood, DiscreteTask
```

---

## Config Dataclasses

All configs inherit from `BasePFNConfig(HookedTransformerConfig)`, which adds:

| Field | Type | Default |
|-------|------|---------|
| `input_dim` | `int` | `16` |
| `use_pos_emb` | `bool` | `True` |
| `normalization_type` | `str` | `"LN"` |

Plus all `HookedTransformerConfig` fields: `d_model`, `n_layers`, `n_heads`, `d_head`, `d_vocab`, `act_fn`, etc.

### RegressionConfig (`SupervisedRegressionPFNConfig`)

| Field | Type | Default |
|-------|------|---------|
| `mask_type` | `Literal["autoregressive-pfn", "gpt2"]` | `"autoregressive-pfn"` |
| `prediction_type` | `Literal["distribution", "point"]` | `"distribution"` |
| `bucket_type` | `Literal["uniform", "riemann"] \| None` | `None` |
| `bucket_support` | `Literal["unbounded", "bounded"]` | `"unbounded"` |
| `y_min` | `float \| None` | `None` |
| `y_max` | `float \| None` | `None` |
| `riemann_borders` | `Tensor \| None` | `None` |

### ClassificationConfig (`ClassificationPFNConfig`)

| Field | Type | Default |
|-------|------|---------|
| `num_classes` | `int` | `2` |
| `y_type` | `Literal["continuous", "categorical"]` | `"continuous"` |
| `mask_type` | `Literal["autoregressive-pfn", "gpt2"]` | `"autoregressive-pfn"` |

### UnsupervisedConfig (`UnsupervisedPFNConfig`)

| Field | Type | Default |
|-------|------|---------|
| `d_vocab` | `int` | `2` |
| `input_type` | `Literal["discrete", "continuous"]` | `"discrete"` |
| `prediction_type` | `Literal["point", "distribution"]` | `"distribution"` |
| `bucket_type` | `Literal["uniform", "riemann"] \| None` | `None` |
| `bucket_support` | `Literal["unbounded", "bounded"]` | `"unbounded"` |
| `y_min` | `float \| None` | `None` |
| `y_max` | `float \| None` | `None` |
| `riemann_borders` | `Tensor \| None` | `None` |
| `mask_type` | `Literal["gpt2"]` | `"gpt2"` (enforced) |
| `act_fn` | `str` | `"gelu"` |

---

## TrainingConfig

```python
@dataclass
class TrainingConfig:
    # data
    batch_size: int = 32
    seq_len: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False

    # optimization
    num_steps: int = 10000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    use_warmup: bool = True
    warmup_steps: int = 500
    use_grad_clip: bool = True
    grad_clip: float = 1.0
    seed: int | None = None

    # logging
    log_every: int = 100
    log_distributional_mse: bool = False
    log_file: str | None = None

    # checkpointing
    save_checkpoint: bool = True
    checkpoint_schedule: str = "linear"        # "linear" or "logarithmic"
    save_every: int = 1000
    linear_checkpoint_interval: int = 100
    n_log_checkpoints: int = 1000
    checkpoint_dir: str = "checkpoints"

    # evaluation
    eval_every: int | None = None
    eval_batches: int = 10

    # device
    device: str = "auto"                       # "auto" -> cuda > mps > cpu

    # wandb
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_log_model: bool = True
    wandb_tags: list[str] | None = None
    wandb_notes: str | None = None

    def get_device(self) -> str
```

---

## train()

```python
def train(
    data_generator: DataGenerator,
    model_config: BasePFNConfig,
    training_config: TrainingConfig,
    *,
    resume_from: str | None = None,
    eval_data_generator: DataGenerator | None = None,
    data_config: Any = None,          # dataclass, logged to wandb
) -> BasePFN
```

---

## Data Generators

Type alias: `DataGenerator = SupervisedDataGenerator | UnsupervisedDataGenerator`

### DeterministicGenerator (`DeterministicFunctionGenerator`)

```python
class DeterministicFunctionGenerator:
    def __init__(
        self,
        prior: Distribution,
        function: Callable[[Tensor, Any], Tensor],
        input_dim: int,
        noise_std: float | None = 0.0,
        x_distribution: Distribution = Normal(0, 1),
        device: str | torch.device | None = None,
    )
    def generate(self, seq_len: int) -> tuple[Tensor, Tensor]
    def generate_with_params(self, seq_len: int) -> tuple[tuple[Tensor, Tensor], dict]
```

### BayesianGenerator (`SupervisedProbabilisticGenerator`)

```python
class SupervisedProbabilisticGenerator:
    def __init__(
        self,
        prior: PriorDistribution,
        likelihood: LikelihoodDistribution,
        x_distribution: Distribution | None = None,
    )
    def generate(self, seq_len: int) -> tuple[Tensor, Tensor]
    def generate_with_params(self, seq_len: int) -> tuple[tuple[Tensor, Tensor], dict]
```

### DatasetGenerator (`FixedDatasetGenerator`)

```python
class FixedDatasetGenerator:
    def __init__(
        self,
        x_data: Tensor,              # [N, input_dim]
        y_data: Tensor,              # [N]
        sequential: bool = False,
    )
    def generate(self, seq_len: int) -> tuple[Tensor, Tensor]
```

### UnsupervisedBayesian (`UnsupervisedProbabilisticGenerator`)

```python
class UnsupervisedProbabilisticGenerator:
    def __init__(
        self,
        prior: PriorDistribution,
        likelihood: LikelihoodDistribution,
    )
    def generate(self, seq_len: int) -> Tensor
    def generate_with_params(self, seq_len: int) -> tuple[Tensor, dict]
```

### Bayesian Primitives

```python
class PriorDistribution(Distribution):
    def __init__(self, base_distribution: Distribution)

class LikelihoodDistribution(Distribution):
    def __init__(
        self,
        base_distribution: Distribution,
        parameterizer: Callable[[Tensor, Tensor], dict[str, Tensor]],
        input_dim: int,
    )

class DiscreteTaskDistribution(Distribution):
    def __init__(self, tasks: Tensor)
```

---

## Model Classes

### Factory

```python
def PFNModel(config: BasePFNConfig) -> BasePFN
# aliased as PFN at top level
```

### BasePFN (abstract)

```python
class BasePFN(nn.Module, ABC):
    transformer: HookedTransformer
    config: BasePFNConfig

    def get_bucket_values(self, y: Tensor) -> Tensor
    def get_y_values(self, bucket_indices: Tensor) -> Tensor
    def log_bucket_densities(self, logits: Tensor) -> Tensor
    @property
    def bucketizer(self) -> Bucketizer
```

### SupervisedPFN

```python
class SupervisedPFN(BasePFN):
    config: SupervisedRegressionPFNConfig | ClassificationPFNConfig

    def forward(
        self,
        x: Tensor,   # [batch, seq, input_dim]
        y: Tensor,   # [batch, seq]
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, ActivationCache]
    # returns logits [batch, seq, d_vocab]

    def predict_on_prompt(
        self,
        x: Tensor,   # [..., seq, input_dim]
        y: Tensor,   # [..., seq]
        *,
        temperature: float = 1.0,
        return_logits: bool = False,
        return_cache: bool = False,
    ) -> DistributionPrediction | PointPrediction | ClassificationPrediction

    def generate(
        self,
        x_distribution: Distribution,
        num_generate: int,
        prompt_x: Tensor | None = None,
        prompt_y: Tensor | None = None,
        sample: bool = True,
        temperature: float = 1.0,
        num_rollouts: int = 1,
    ) -> tuple[Tensor, Tensor]
    # returns (x, y) for generated sequences
```

### UnsupervisedPFN

```python
class UnsupervisedPFN(BasePFN):
    config: UnsupervisedPFNConfig

    def forward(
        self,
        y: Tensor,   # [batch, seq]
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, ActivationCache]
    # returns logits [batch, seq, d_vocab]

    def predict_on_prompt(
        self,
        y: Tensor,   # [..., seq]
        *,
        temperature: float = 1.0,
        return_logits: bool = False,
        return_cache: bool = False,
    ) -> DistributionPrediction | PointPrediction

    def generate(
        self,
        num_generate: int,
        prompt: Tensor | None = None,
        sample: bool = True,
        temperature: float = 1.0,
        num_rollouts: int = 1,
    ) -> Tensor
```

---

## Prediction Output Types

```python
@dataclass
class DistributionPrediction:
    probs: Tensor      # [..., seq, d_vocab]
    y_grid: Tensor     # [d_vocab]
    logits: Tensor | None = None

@dataclass
class PointPrediction:
    preds: Tensor      # [..., seq, 1]

@dataclass
class ClassificationPrediction:
    probs: Tensor      # [..., seq, num_classes]
    logits: Tensor | None = None
```

---

## Checkpointing

```python
from pfn_transformerlens.checkpointing import save_checkpoint, load_checkpoint, CheckpointMetadata

@dataclass
class CheckpointMetadata:
    timestamp: str
    wandb_run_id: str | None
    wandb_run_name: str | None
    wandb_run_url: str | None
    git_hash: str | None

def save_checkpoint(
    checkpoint_path: Path,
    step: int,
    model_state: dict,
    optimizer_state: dict,
    model_config: BasePFNConfig,
    training_config: TrainingConfig,
    metadata: CheckpointMetadata,
    scheduler_state: dict | None = None,
    task_distribution: dict | None = None,
) -> None

def load_checkpoint(
    checkpoint_path: Path | str,
    device: str = "auto",
    load_optimizer: bool = False,
) -> tuple[BasePFN, dict | None, CheckpointMetadata]

def get_logarithmic_checkpoint_steps(
    training_steps: int,
    n_log_checkpoints: int = 1000,
    linear_interval: int = 100,
) -> list[int]
```

---

## wandb_utils

Only contains run naming utilities. Model loading/listing functions have been removed.

```python
from pfn_transformerlens.wandb_utils import RunNameScheme, create_run_name

@dataclass
class RunNameScheme:
    model_fields: tuple[str, ...] = ()
    training_fields: tuple[str, ...] = ()
    data_fields: tuple[str, ...] = ()

    @classmethod
    def from_templates(
        cls,
        model: Any | None = None,
        training: Any | None = None,
        data: Any | None = None,
    ) -> RunNameScheme

def create_run_name(
    *,
    base: str,
    model_config: Any | None = None,
    training_config: Any | None = None,
    data_config: Any | None = None,
    scheme: RunNameScheme | None = None,
    include_fields: dict[str, Sequence[str]] | None = None,
    extra: Mapping[str, Any] | None = None,
    max_length: int = 128,
) -> str
```

---

## WandbLogger

```python
from pfn_transformerlens.wandb_logger import WandbLogger

class WandbLogger:
    def __init__(
        self,
        training_config: TrainingConfig,
        model_config: BasePFNConfig,
        data_config: Any = None,
    )
    def log(self, metrics: dict[str, float], step: int) -> None
    def log_checkpoint(
        self,
        checkpoint_path: str | Path,
        step: int,
        metadata: CheckpointMetadata | None = None,
    ) -> None
    def finish(self) -> None

    # properties set after init
    enabled: bool
    run_id: str | None
    run_name: str | None
    run_url: str | None
```

---

## Utilities

### sample_batch

```python
from pfn_transformerlens.sampler.dataloader import sample_batch

def sample_batch(
    data_generator: DataGenerator,
    batch_size: int,
    seq_len: int,
) -> tuple[Tensor | None, Tensor]
# returns (x, y) where x is None for unsupervised generators
```

### estimate_riemann_borders

```python
from pfn_transformerlens.model.bucketizer import estimate_riemann_borders

def estimate_riemann_borders(
    ys: Tensor,
    *,
    num_buckets: int,
    widen_borders_factor: float = 1.0,
) -> Tensor  # [num_buckets + 1]
```

---

## End-to-End Example

```python
import torch
from pfn_transformerlens import (
    train, TrainingConfig, RegressionConfig, DeterministicGenerator,
    estimate_riemann_borders, sample_batch,
)

# 1. data generator
def linear_fn(x, w):
    return (w * x).sum(dim=-1)

data_gen = DeterministicGenerator(
    prior=torch.distributions.Normal(0.0, 1.0),
    function=linear_fn,
    input_dim=5,
    noise_std=0.1,
)

# 2. estimate bucket borders from sample data
_, sample_y = sample_batch(data_gen, batch_size=100, seq_len=64)
borders = estimate_riemann_borders(sample_y.flatten(), num_buckets=100)

# 3. model config
model_cfg = RegressionConfig(
    d_model=128, n_layers=4, n_heads=4, d_head=32,
    input_dim=5,
    prediction_type="distribution",
    bucket_type="riemann",
    d_vocab=100,
    riemann_borders=borders,
)

# 4. train
train_cfg = TrainingConfig(
    batch_size=32, seq_len=64, num_steps=5000,
    use_wandb=False,
)
model = train(data_gen, model_cfg, train_cfg)

# 5. inference
device = next(model.parameters()).device
pred = model.predict_on_prompt(
    x=torch.randn(1, 20, 5).to(device),
    y=torch.randn(1, 20).to(device),
)
# pred is a DistributionPrediction with .probs [1, 20, 100] and .y_grid [100]
```

---

## Package Structure

```
pfn_transformerlens/
  __init__.py              top-level exports
  train.py                 training loop, TrainingConfig
  checkpointing.py         save/load checkpoints
  wandb_utils.py           RunNameScheme, create_run_name
  wandb_logger.py          WandbLogger
  generators.py            short-name generator re-exports
  configs.py               short-name config re-exports
  bayes.py                 short-name bayesian re-exports
  model/
    PFN.py                 BasePFN, SupervisedPFN, UnsupervisedPFN, PFNModel
    PFNMasks.py            attention mask implementations
    bucketizer.py          Bucketizer, estimate_riemann_borders
    configs/
      base.py              BasePFNConfig
      regression.py        SupervisedRegressionPFNConfig
      classification.py    ClassificationPFNConfig
      unsupervised.py      UnsupervisedPFNConfig
  sampler/
    data_generator.py      all generator classes + protocols
    prior_likelihood.py    PriorDistribution, LikelihoodDistribution, DiscreteTaskDistribution
    dataloader.py          sample_batch, build_dataloader
```
