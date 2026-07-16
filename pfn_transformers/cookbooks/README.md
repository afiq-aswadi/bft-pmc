# PFN Transformer Cookbooks

This directory contains executable cookbook examples demonstrating various capabilities of the PFN (Prior-Fitted Network) transformer library.

## Running the Cookbooks

Each cookbook is a standalone Python script that can be run with:

```bash
uv run python cookbooks/XX_name.py
```

All outputs (plots, checkpoints) are saved to `cookbooks/outputs/XX_name/`.

## Available Cookbooks

### 01_supervised_linear_regression.py
**Supervised Learning: Noisy Linear Regression**

Trains a PFN to learn linear functions `y = w·x + ε` where weights are sampled from a standard normal distribution.

- Demonstrates in-context learning with uncertainty quantification
- Shows predictions with confidence bands (±2σ)
- Generates plots comparing predicted vs true values

**Key concepts**: DeterministicFunctionGenerator, SupervisedRegressionPFNConfig, distribution prediction with bucketing

---

### 02_unsupervised_coinflips.py
**Unsupervised Learning: Beta-Bernoulli Model**

Trains a PFN on coinflip sequences where the bias `p ~ Beta(2, 2)` and observations are Bernoulli(p).

- Demonstrates posterior updating as more data is observed
- Compares predicted P(heads) vs empirical average
- Shows convergence to true bias

**Key concepts**: UnsupervisedProbabilisticGenerator, PriorDistribution, LikelihoodDistribution, discrete input type

---

### 03_supervised_polynomial.py
**Supervised Learning: Polynomial Regression**

Trains a PFN on polynomial functions `y = a₀ + a₁x + a₂x²` with random coefficients.

- Shows learning of more complex function families
- Visualizes predictions with confidence bands
- Tests on multiple polynomial instances

**Key concepts**: Higher-order functions, continuous predictions, visualization on dense grids

---

### 04_unsupervised_markov.py
**Unsupervised Learning: Discrete Markov Chains**

Trains a PFN to learn transition dynamics from Markov chain sequences with random transition matrices.

- Demonstrates learning of sequential dependencies
- Compares learned vs true transition matrices using heatmaps
- Shows prediction of next-state probabilities

**Key concepts**: Custom generators, discrete state spaces, transition matrix visualization

---

### 05_wandb_config_interpretability.py
**Advanced: Wandb, Config Loading, and Interpretability**

Demonstrates advanced features for experiment tracking and model introspection:

1. **Wandb logging**: Setup for experiment tracking (set `use_wandb=True`)
2. **Model checkpointing**: Saving with metadata and loading by config
3. **Interpretability**: Accessing internal transformer activations
   - Attention pattern visualization
   - Residual stream norm analysis across layers

**Key concepts**: save_model, load_model_by_config, return_cache=True, TransformerLens hooks

---

### 06_supervised_binary_classification.py
**Supervised Learning: Binary Classification**

Trains a PFN to perform binary classification on 2D points:
- Class 0: points inside a circle
- Class 1: points outside the circle

- Demonstrates in-context learning of non-linear decision boundaries
- Shows decision boundary visualization with confidence regions
- Includes calibration analysis and prediction accuracy plots

**Key concepts**: ClassificationPFNConfig, DeterministicGenerator with discrete outputs, non-linear decision boundaries

---

### 07_continuous_sampling_demo.py
**Sampling: Continuous Buckets and Tails**

Shows how `Bucketizer.sample` draws continuous values from bucketed distributions.

- Compares bounded vs unbounded support using uniform buckets
- Visualizes histograms and bucket borders to highlight tails
- Prints summary statistics and contrasts with midpoint decoding

**Key concepts**: Bucketizer.sample, half-normal tails, temperature scaling, histogram visualization

## Common Patterns

### Training Configuration
All cookbooks use small models for fast training:
- 2-4 layers, 64-128 dimensional embeddings
- 100-500 training steps
- Batch size 64-128

### Model Types
- **Supervised**: Input-output pairs `(x, y)` with custom attention masks
- **Unsupervised**: Sequence modeling without x/y separation

### Prediction Types
- **Distribution**: Outputs probability distributions (with bucketing for continuous values)
- **Point**: Direct scalar predictions

## Tips

- Increase `num_steps` for better convergence (at the cost of longer training)
- Set `use_wandb=True` in TrainingConfig to enable experiment tracking
- Use `return_cache=True` in predictions for mechanistic interpretability
- Adjust `y_min`, `y_max`, and `d_vocab` (number of buckets) for distribution predictions

## Dependencies

All scripts use:
- `torch` for tensors and distributions
- `matplotlib` and `seaborn` for visualization
- `pfn_transformerlens` for models and training

Run `uv sync` to install dependencies.
