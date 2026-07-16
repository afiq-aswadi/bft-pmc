"""
Cookbook 1: Supervised Linear Regression

Training a PFN (Prior-data Fitted Network) to learn
linear functions through in-context learning.

We train a PFN to perform regression on linear functions of the form:
    y = w·x + ε

where:
- w is a weight vector sampled from N(0, I)
- x is the input feature vector
- ε is Gaussian noise with std=0.1

The trained PFN learns to identify which linear function it's seeing from a few
examples, then predict on new points with uncertainty quantification.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import tyro
from einops import einsum
from jaxtyping import Float

from pfn_transformerlens import (
    DeterministicGenerator,
    RegressionConfig,
    TrainingConfig,
    sample_batch,
    train,
)


def linear_function(
    x: Float[torch.Tensor, "seq_len input_dim"], w: Float[torch.Tensor, "input_dim"]
) -> Float[torch.Tensor, "seq_len"]:
    """Linear function: y = w·x (element-wise product, then sum over features).

    Args:
        x: input features of shape (seq_len, input_dim)
        w: weight vector of shape (input_dim,)

    Returns:
        y: output of shape (seq_len,)
    """
    return einsum(x, w, "seq_len input_dim, input_dim -> seq_len")


@dataclass
class ExpConfig:
    """Configuration for linear regression cookbook experiment."""

    # data generation parameters
    input_dim: int = 5
    noise_std: float = 0.1

    # model architecture parameters
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_head: int = 32
    n_ctx: int = 128
    d_vocab: int = 50  # number of buckets for distribution prediction

    # bucketing parameters for distribution prediction
    y_min: float = -5.0
    y_max: float = 5.0

    # training parameters
    batch_size: int = 128
    num_steps: int = 500  # reduced for faster demo
    learning_rate: float = 1e-3
    warmup_steps: int = 50
    log_every: int = 100
    use_wandb: bool = False

    # visualization parameters
    num_test_functions: int = 3  # number of functions to plot
    seq_len: int = 50  # sequence length for visualization


def main(config: ExpConfig) -> None:
    output_dir = Path(__file__).parent / "outputs" / "01_linear_regression"
    output_dir.mkdir(parents=True, exist_ok=True)

    # set up data generator
    # we use MultivariateNormal to sample weight vectors from N(0, I)
    prior = torch.distributions.MultivariateNormal(
        loc=torch.zeros(config.input_dim),
        covariance_matrix=torch.eye(config.input_dim),
    )
    data_gen = DeterministicGenerator(
        prior=prior,
        function=linear_function,
        input_dim=config.input_dim,
        noise_std=config.noise_std,
    )

    # configure model architecture
    # the model uses bucket-based distribution prediction with uniform buckets
    model_config = RegressionConfig(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_head=config.d_head,
        n_ctx=config.n_ctx,
        d_vocab=config.d_vocab,
        input_dim=config.input_dim,
        prediction_type="distribution",
        bucket_type="uniform",
        y_min=config.y_min,
        y_max=config.y_max,
        mask_type="autoregressive-pfn",
        act_fn="gelu",
    )

    # configure training
    training_config = TrainingConfig(
        batch_size=config.batch_size,
        num_steps=config.num_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        log_every=config.log_every,
        use_wandb=config.use_wandb,
    )

    # train the model
    model = train(data_gen, model_config, training_config)

    # move model to appropriate device for inference
    device = next(model.parameters()).device

    # visualize in-context learning on multiple random linear functions
    # we generate real sequences with noise and show the model's autoregressive predictions
    # uncertainty should decrease as the model sees more examples
    fig, axes = plt.subplots(1, config.num_test_functions, figsize=(15, 4))

    for idx in range(config.num_test_functions):
        # sample a real sequence from the data generator (with noise)
        x_batch, y_batch = sample_batch(data_gen, batch_size=1, seq_len=config.seq_len)
        assert x_batch is not None
        x_seq = x_batch[0].to(device)  # (seq_len, input_dim)
        y_seq = y_batch[0].to(device)  # (seq_len,)

        # run model to get predictions at all positions
        # the model predicts autoregressively: prediction at position i uses
        # information from positions 0, 1, ..., i-1
        with torch.no_grad():
            pred = model.predict_on_prompt(x_seq, y_seq)

        # extract predictions for all positions
        probs = pred.probs.cpu()  # (seq_len, d_vocab)
        y_grid = pred.y_grid.cpu()  # (d_vocab,)
        y_true = y_seq.cpu()  # (seq_len,)

        # compute mean and std from predicted distribution
        y_pred_mean = (probs * y_grid).sum(dim=-1)
        y_pred_std = torch.sqrt(
            (probs * (y_grid - y_pred_mean.unsqueeze(-1)) ** 2).sum(dim=-1)
        )

        # visualize predictions vs ground truth
        # the uncertainty bands should get thinner as we move along the sequence
        ax = axes[idx]
        x_plot = torch.arange(config.seq_len)
        ax.plot(x_plot, y_true, "o", label="True", alpha=0.7, markersize=4)
        ax.plot(x_plot, y_pred_mean, "s", label="Predicted", alpha=0.7, markersize=4)
        ax.fill_between(
            x_plot,
            y_pred_mean - 2 * y_pred_std,
            y_pred_mean + 2 * y_pred_std,
            alpha=0.3,
            label=r"$\pm 2\sigma$",
        )
        ax.set_xlabel("Position in sequence")
        ax.set_ylabel("y")
        ax.set_title(f"Function {idx + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "predictions_comparison.png", dpi=400, bbox_inches="tight")
    plt.close()

    # create detailed analysis plot for one function
    # this shows (1) prediction accuracy and (2) predicted distributions at different positions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # sample a new sequence for detailed analysis
    x_batch, y_batch = sample_batch(data_gen, batch_size=1, seq_len=config.seq_len)
    assert x_batch is not None
    x_seq = x_batch[0].to(device)
    y_seq = y_batch[0].to(device)

    with torch.no_grad():
        pred = model.predict_on_prompt(x_seq, y_seq)

    probs = pred.probs.cpu()
    y_grid = pred.y_grid.cpu()
    y_true = y_seq.cpu()

    y_pred_mean = (probs * y_grid).sum(dim=-1)
    y_pred_std = torch.sqrt(
        (probs * (y_grid - y_pred_mean.unsqueeze(-1)) ** 2).sum(dim=-1)
    )

    # plot 1: scatter plot of predicted vs true values
    # points on the diagonal line indicate perfect predictions
    ax = axes[0]
    ax.scatter(y_true, y_pred_mean, alpha=0.6)
    ax.errorbar(y_true, y_pred_mean, yerr=2 * y_pred_std, fmt="none", alpha=0.3)
    lim_min = min(y_true.min().item(), y_pred_mean.min().item())
    lim_max = max(y_true.max().item(), y_pred_mean.max().item())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.5, label="Perfect")
    ax.set_xlabel("True y")
    ax.set_ylabel("Predicted y")
    ax.set_title("Prediction accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # plot 2: show predicted distributions at different positions in the sequence
    # early positions should have wider distributions (more uncertainty)
    # later positions should have narrower distributions (less uncertainty)
    ax = axes[1]
    # show distributions at positions: 0, 10, 20, 30, 40 (evenly spaced)
    positions_to_plot = [
        0,
        config.seq_len // 4,
        config.seq_len // 2,
        3 * config.seq_len // 4,
        config.seq_len - 1,
    ]
    for pos in positions_to_plot:
        ax.plot(y_grid, probs[pos], alpha=0.7, label=f"Position {pos}")
        ax.axvline(
            y_true[pos].item(),
            color=f"C{positions_to_plot.index(pos)}",
            linestyle="--",
            alpha=0.5,
        )
    ax.set_xlabel("y")
    ax.set_ylabel("Probability")
    ax.set_title("Predicted distributions at different positions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "detailed_analysis.png", dpi=400, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    tyro.cli(main)
