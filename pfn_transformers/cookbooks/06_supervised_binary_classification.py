"""
Cookbook 6: Supervised Binary Classification

Training a PFN to perform binary classification on 2D points:
- Class 0: points inside a circle of radius r
- Class 1: points outside the circle

The PFN learns to classify new points based on a few labeled examples,
demonstrating in-context learning for non-linear decision boundaries.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from pfn_transformerlens import (
    DeterministicGenerator,
    ClassificationConfig,
    TrainingConfig,
    sample_batch,
    train,
)


def circle_classification_function(
    x: torch.Tensor, center: torch.Tensor, radius: float
) -> torch.Tensor:
    """Binary classification: 1 if point is outside circle, 0 if inside.

    Args:
        x: points of shape (seq_len, 2) - 2D coordinates
        center: center point of shape (2,)
        radius: radius of the circle

    Returns:
        labels: binary labels of shape (seq_len,) - 0 inside, 1 outside
    """
    distances = torch.sqrt(((x - center) ** 2).sum(dim=-1))
    return (distances > radius).long()


@dataclass
class ExpConfig:
    """Configuration for binary classification cookbook experiment."""

    # data generation parameters
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = 1.0

    # model architecture parameters
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_head: int = 32
    n_ctx: int = 1024  # must be 2*seq_len for supervised models (x/y interleaving)

    # training parameters
    batch_size: int = 128
    num_steps: int = 500
    learning_rate: float = 1e-3
    warmup_steps: int = 50
    log_every: int = 100
    use_wandb: bool = False

    # visualization parameters
    grid_resolution: int = 20  # for decision boundary visualization
    num_test_examples: int = 3  # number of test sequences to visualize
    num_context_examples: int = (
        8  # number of examples to use as context for grid prediction
    )


def main(config: ExpConfig) -> None:
    output_dir = Path(__file__).parent / "outputs" / "06_binary_classification"
    output_dir.mkdir(parents=True, exist_ok=True)

    # create center tensor
    center = torch.tensor([config.center_x, config.center_y])

    # set up data generator
    def classification_fn(x: torch.Tensor, _theta: torch.Tensor) -> torch.Tensor:
        # _theta is not used for this deterministic classification
        return circle_classification_function(x, center, config.radius)

    # we use a uniform distribution over a square containing the circle
    # to ensure we get examples from both classes
    prior = torch.distributions.Uniform(
        low=-config.radius * 1.5, high=config.radius * 1.5
    )

    # DeterministicGenerator samples circle parameters and evaluates labels exactly
    data_gen = DeterministicGenerator(
        prior=prior,
        function=classification_fn,
        input_dim=2,  # 2D points
        noise_std=None,  # no noise for classification labels
    )

    # configure model architecture for binary classification
    model_config = ClassificationConfig(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_head=config.d_head,
        n_ctx=config.n_ctx,
        input_dim=2,
        d_vocab=1000,  # required by HookedTransformer (not used for classification)
        num_classes=2,  # binary classification
        y_type="categorical",  # use categorical embeddings for class labels
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

    # create visualization of decision boundaries
    _, axes = plt.subplots(1, config.num_test_examples, figsize=(15, 4))

    for idx in range(config.num_test_examples):
        # sample a sequence of labeled examples
        x_batch, y_batch = sample_batch(
            data_gen, batch_size=1, seq_len=config.n_ctx // 2
        )
        assert x_batch is not None
        x_seq = x_batch[0].to(device)  # (seq_len, 2)
        y_seq = y_batch[0].to(device)  # (seq_len,)

        # get predictions on the sequence
        with torch.no_grad():
            pred = model.predict_on_prompt(x_seq, y_seq)

        # extract predicted probabilities for class 1 (outside circle)
        probs_class_1 = pred.probs[:, 1].cpu()  # (seq_len,)

        # create grid for decision boundary visualization
        x_grid = np.linspace(
            -config.radius * 1.5, config.radius * 1.5, config.grid_resolution
        )
        y_grid = np.linspace(
            -config.radius * 1.5, config.radius * 1.5, config.grid_resolution
        )
        xx, yy = np.meshgrid(x_grid, y_grid)

        # create test points on grid
        grid_points = torch.tensor(
            np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32
        )
        grid_points = grid_points.to(device)

        # get predictions for all grid points using a subset of examples as context
        # use only num_context_examples to stay within context window
        context_x = x_seq[: config.num_context_examples]
        context_y = y_seq[: config.num_context_examples]

        full_x = torch.cat([context_x, grid_points], dim=0)
        full_y = torch.cat(
            [
                context_y,
                torch.zeros(len(grid_points), dtype=torch.long, device=device),
            ],
            dim=0,
        )

        with torch.no_grad():
            grid_pred = model.predict_on_prompt(full_x, full_y)

        # extract predicted probabilities for grid points (last part of sequence)
        grid_probs = grid_pred.probs[-len(grid_points) :, 1].cpu().reshape(xx.shape)

        # visualize
        ax = axes[idx] if config.num_test_examples > 1 else axes

        # plot decision boundary (p=0.5 contour)
        contour = ax.contourf(
            xx, yy, grid_probs, levels=np.linspace(0, 1, 11), cmap="RdYlBu_r", alpha=0.6
        )
        ax.contour(xx, yy, grid_probs, levels=[0.5], colors="black", linewidths=2)

        # plot training examples
        x_seq_cpu = x_seq.cpu().numpy()
        y_seq_cpu = y_seq.cpu().numpy()
        ax.scatter(
            x_seq_cpu[y_seq_cpu == 0, 0],
            x_seq_cpu[y_seq_cpu == 0, 1],
            c="blue",
            marker="o",
            s=80,
            edgecolors="black",
            label="Inside (train)",
            alpha=0.8,
        )
        ax.scatter(
            x_seq_cpu[y_seq_cpu == 1, 0],
            x_seq_cpu[y_seq_cpu == 1, 1],
            c="red",
            marker="s",
            s=80,
            edgecolors="black",
            label="Outside (train)",
            alpha=0.8,
        )

        # plot true circle boundary
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = config.center_x + config.radius * np.cos(theta)
        circle_y = config.center_y + config.radius * np.sin(theta)
        ax.plot(circle_x, circle_y, "k--", linewidth=2, label="True boundary")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Example {idx + 1}")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # add colorbar
    cbar = plt.colorbar(contour, ax=axes, shrink=0.8)
    cbar.set_label("Predicted P(Outside)")

    plt.tight_layout()
    plt.savefig(output_dir / "decision_boundaries.png", dpi=400, bbox_inches="tight")
    plt.close()

    # create spatial classification plot
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # sample a new sequence for analysis
    x_batch, y_batch = sample_batch(data_gen, batch_size=1, seq_len=config.n_ctx // 2)
    assert x_batch is not None
    x_seq = x_batch[0].to(device)
    y_seq = y_batch[0].to(device)

    with torch.no_grad():
        pred = model.predict_on_prompt(x_seq, y_seq)

    probs_class_1 = pred.probs[:, 1].cpu().numpy()
    y_pred = (probs_class_1 > 0.5).astype(int)
    y_true = y_seq.cpu().numpy()
    x_points = x_seq.cpu().numpy()

    # plot 1: true labels in 2D space
    ax = axes[0]
    # plot circle boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = config.center_x + config.radius * np.cos(theta)
    circle_y = config.center_y + config.radius * np.sin(theta)
    ax.plot(circle_x, circle_y, "k--", linewidth=2, label="True boundary")

    # plot points colored by true label
    inside_mask = y_true == 0
    outside_mask = y_true == 1
    ax.scatter(
        x_points[inside_mask, 0],
        x_points[inside_mask, 1],
        c="blue",
        marker="o",
        s=50,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        label="Inside (true)",
    )
    ax.scatter(
        x_points[outside_mask, 0],
        x_points[outside_mask, 1],
        c="red",
        marker="s",
        s=50,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        label="Outside (true)",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("True labels")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # plot 2: predicted labels in 2D space
    ax = axes[1]
    # plot circle boundary
    ax.plot(circle_x, circle_y, "k--", linewidth=2, label="True boundary")

    # plot points colored by predicted label
    pred_inside_mask = y_pred == 0
    pred_outside_mask = y_pred == 1
    ax.scatter(
        x_points[pred_inside_mask, 0],
        x_points[pred_inside_mask, 1],
        c="blue",
        marker="o",
        s=50,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        label="Inside (predicted)",
    )
    ax.scatter(
        x_points[pred_outside_mask, 0],
        x_points[pred_outside_mask, 1],
        c="red",
        marker="s",
        s=50,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        label="Outside (predicted)",
    )

    # highlight misclassified points with X markers
    misclassified = y_pred != y_true
    if np.any(misclassified):
        ax.scatter(
            x_points[misclassified, 0],
            x_points[misclassified, 1],
            c="yellow",
            marker="x",
            s=100,
            linewidth=2,
            label="Misclassified",
            zorder=10,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Predicted labels")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # add accuracy text
    accuracy = np.mean(y_pred == y_true)
    ax.text(
        0.02,
        0.98,
        f"Accuracy: {accuracy:.1%}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(output_dir / "prediction_analysis.png", dpi=400, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    tyro.cli(main)
