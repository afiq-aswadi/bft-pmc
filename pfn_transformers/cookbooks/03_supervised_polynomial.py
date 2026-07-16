"""Cookbook 3: Supervised Polynomial Regression."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import tyro
from einops import einsum

from pfn_transformerlens import (
    DeterministicGenerator,
    DiscreteTask,
    RegressionConfig,
    TrainingConfig,
    train,
)
from pfn_transformerlens.model.PFN import PointPrediction


@dataclass
class ExpConfig:
    """Configuration for supervised polynomial regression."""

    num_tasks: int = 20
    coeff_std: float = 1.0
    noise_std: float = 0.0  # noiseless

    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 2
    d_head: int = 64
    n_ctx: int = 128
    d_vocab: int = 1  # unused for point predictions but required by config

    batch_size: int = 128
    seq_len: int = 64
    num_steps: int = 3000
    learning_rate: float = 1e-3
    warmup_steps: int = 30
    log_every: int = 100
    use_wandb: bool = False
    save_checkpoint: bool = False

    num_eval_tasks: int = 3
    eval_x_min: float = -2.0
    eval_x_max: float = 2.0
    num_eval_points: int = 59
    num_context_points: int = 5
    context_x_min: float = -2
    context_x_max: float = 2


def polynomial_function(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Polynomial: y = a0 + a1*x + a2*x^2."""
    x_flat = x.squeeze(-1)
    # Design matrix: [1, x, x^2] for each x
    design = torch.stack([torch.ones_like(x_flat), x_flat, x_flat**2], dim=-1)
    # einsum: seq_len coeffs, coeffs -> seq_len
    return einsum(design, coeffs, "seq_len coeffs, coeffs -> seq_len")


def main(config: ExpConfig) -> None:
    output_dir = Path(__file__).parent / "outputs" / "03_polynomial"
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = torch.randn(config.num_tasks, 3) * config.coeff_std
    prior = DiscreteTask(
        tasks=tasks,
    )

    data_gen = DeterministicGenerator(
        prior=prior,
        function=polynomial_function,
        input_dim=1,
        noise_std=config.noise_std,
    )

    model_config = RegressionConfig(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_head=config.d_head,
        n_ctx=config.n_ctx,
        input_dim=1,
        d_vocab=config.d_vocab,
        prediction_type="point",
        mask_type="gpt2",
        act_fn="gelu",
    )

    training_config = TrainingConfig(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        num_steps=config.num_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        log_every=config.log_every,
        use_wandb=config.use_wandb,
        save_checkpoint=config.save_checkpoint,
    )

    model = train(data_gen, model_config, training_config, data_config=config)

    eval_x = torch.linspace(
        config.eval_x_min, config.eval_x_max, config.num_eval_points
    ).unsqueeze(-1)

    if config.num_context_points > 0:
        context_x = torch.linspace(
            config.context_x_min,
            config.context_x_max,
            config.num_context_points,
        ).unsqueeze(-1)
        prompt_x = torch.cat([context_x, eval_x], dim=0)
    else:
        context_x = torch.empty((0, 1))
        prompt_x = eval_x

    fig, axes = plt.subplots(1, config.num_eval_tasks, figsize=(15, 4))
    axes_list = [axes] if config.num_eval_tasks == 1 else list(axes)

    for idx, ax in enumerate(axes_list):
        coeffs = prior.sample()
        prompt_y = polynomial_function(prompt_x, coeffs)

        with torch.no_grad():
            pred = model.predict_on_prompt(prompt_x, prompt_y)
            assert isinstance(pred, PointPrediction)

        preds = pred.preds.squeeze(-1).cpu()
        true_vals = prompt_y.cpu()

        if config.num_context_points > 0:
            context_y = true_vals[: config.num_context_points]
            ax.scatter(
                context_x.squeeze(),
                context_y,
                color="black",
                label="Context",
            )

        eval_preds = preds[config.num_context_points :]
        eval_true = true_vals[config.num_context_points :]
        ax.plot(
            eval_x.squeeze(),
            eval_true,
            "g--",
            label="True",
            alpha=0.7,
        )
        ax.plot(
            eval_x.squeeze(),
            eval_preds,
            "b-",
            label="Predicted",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Polynomial {idx + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "polynomial_predictions.png", dpi=400)
    plt.close()

    print(f"Saved: {output_dir / 'polynomial_predictions.png'}")


if __name__ == "__main__":
    tyro.cli(main)
