# PFN TransformerLens

Library for training Prior-Fitted Networks (PFNs) with TransformerLens. This
directory is the vendored library version used by the experiments in the parent
repository and is covered by its own MIT license.

## Installation

### Local Development

From the parent workspace:

```bash
cd pfn_transformers

# Basic installation
uv sync --extra examples

# With W&B support
uv sync --extra wandb
```

## Usage

### Training

#### Using DeterministicFunctionGenerator (function-based tasks)

```python
import torch
from pfn_transformerlens.model.configs.regression import SupervisedRegressionPFNConfig
from pfn_transformerlens.train import train, TrainingConfig
from pfn_transformerlens.sampler.data_generator import DeterministicFunctionGenerator

# Define task function (e.g., linear regression)
def linear_function(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (w * x).sum(dim=-1)

# Setup data generator
data_gen = DeterministicFunctionGenerator(
    prior=torch.distributions.Normal(0.0, 1.0),  # distribution over function parameters
    function=linear_function,
    input_dim=10,
    noise_std=0.1,  # None for noiseless
    x_distribution=torch.distributions.Normal(0.0, 1.0)  # optional, defaults to N(0,1)
)

# Configure model
model_cfg = SupervisedRegressionPFNConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_head=32,
    input_dim=10
)

# Configure training
train_cfg = TrainingConfig(
    batch_size=32,
    seq_len=64,
    num_steps=10000,
    learning_rate=1e-4,
    use_wandb=True,
    wandb_project="my-project",  # or set WANDB_PROJECT env var
    wandb_entity="my-team",      # or set WANDB_ENTITY env var
    save_checkpoint=True,
    checkpoint_dir="checkpoints"
)

# Train
model = train(data_gen, model_cfg, train_cfg)
```

#### Using SupervisedProbabilisticGenerator (Bayesian workflow)

```python
from pfn_transformerlens.sampler.data_generator import SupervisedProbabilisticGenerator
from pfn_transformerlens.sampler.prior_likelihood import (
    PriorDistribution,
    LikelihoodDistribution,
    DiscreteTaskDistribution
)

# Define discrete tasks
tasks = torch.randn(1024)  # 1024 different task parameters
prior = PriorDistribution(DiscreteTaskDistribution(tasks))

# Define likelihood parameterizer
def normal_parameterizer(theta: torch.Tensor, x: torch.Tensor) -> dict:
    return {
        "loc": x.squeeze(-1) * theta,
        "scale": torch.ones_like(x.squeeze(-1)) * 0.1
    }

likelihood = LikelihoodDistribution(
    base_distribution=torch.distributions.Normal(0.0, 1.0),
    parameterizer=normal_parameterizer,
    input_dim=1
)

# Create generator
data_gen = SupervisedProbabilisticGenerator(
    prior=prior,
    likelihood=likelihood,
    x_distribution=torch.distributions.Normal(0.0, 1.0)  # optional
)
```

#### Other available generators

- `UnsupervisedProbabilisticGenerator` - for unsupervised learning (generates y only)
- `FixedDatasetGenerator` - sample from static dataset

### Sampling Data from Generators

Generators provide two ways to sample data:

#### Single sequence generation (use `.generate()` method)

```python
# Generate a single sequence
x, y = data_gen.generate(seq_len=64)
# x shape: (64, input_dim), y shape: (64,)
```

#### Batch generation (use standalone `sample_batch` function)

**Important**: Generators do NOT have a `.sample_batch()` method. Use the standalone function from the dataloader module:

```python
from pfn_transformerlens.sampler.dataloader import sample_batch

# Generate a batch of sequences
x_batch, y_batch = sample_batch(data_gen, batch_size=32, seq_len=64)
# x_batch shape: (32, 64, input_dim), y_batch shape: (32, 64)

# For unsupervised generators, x_batch will be None
unsupervised_gen = UnsupervisedProbabilisticGenerator(prior, likelihood)
x_batch, y_batch = sample_batch(unsupervised_gen, batch_size=32, seq_len=64)
# x_batch is None, y_batch shape: (32, 64)
```

#### Using dataloaders in training

The `train()` function handles batching automatically. You don't need to call `sample_batch` manually:

```python
# The train function uses build_dataloader internally
model = train(data_gen, model_cfg, train_cfg)
```

### Loading Models from Checkpoints

Load from local checkpoint:

```python
from pfn_transformerlens.checkpointing import load_checkpoint

model, optimizer_state, metadata = load_checkpoint(
    "checkpoints/checkpoint_step_5000.pt",
    device="cuda"
)

print(f"Loaded model trained at: {metadata.timestamp}")
```

### Structured W&B Run Names

```python
from pfn_transformerlens.wandb_utils import create_run_name, RunNameScheme

scheme = RunNameScheme(
    model_fields=("n_layers", "d_model"),
    training_fields=("learning_rate",)
)

run_name = create_run_name(
    base="pfn",
    model_config=model_cfg,
    training_config=train_cfg,
    scheme=scheme
)
# Result: "pfn_n4_d128_lr0.0001"
```

## Development

### Code Quality Checks

After making changes, run these checks:

```bash
# Format and lint
ruff check --fix . && ruff format .

# Type check
uvx ty check

# Tests
uv run pytest
```
