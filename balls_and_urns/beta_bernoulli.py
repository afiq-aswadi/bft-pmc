"""Train and evaluate the paper's continuous-prior Beta-Bernoulli transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float, Int
from scipy.stats import beta as beta_distribution
import torch
import tyro

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from balls_and_urns.data import BOSGenerator, make_bau_generator
from balls_and_urns.predictive_monte_carlo import (
    predictive_monte_carlo_theta_chunked,
)
from pfn_transformerlens import TrainingConfig, UnsupervisedConfig, train
from pfn_transformerlens.checkpointing import load_checkpoint
from pfn_transformerlens.model.PFN import UnsupervisedPFN


@dataclass(frozen=True)
class BetaBernoulliDataConfig:
    """Continuous Beta prior used to generate Bernoulli sequences."""

    prior_alpha: float
    prior_beta: float


@dataclass
class BetaBernoulliConfig:
    """Paper configuration with smaller values available through CLI overrides."""

    output_dir: Path = Path("outputs/bau/beta_bernoulli")
    checkpoint_path: Path | None = None

    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    d_model: int = 128
    d_mlp: int = 512
    n_layers: int = 2
    n_heads: int = 2
    d_head: int = 32
    seq_len: int = 512

    batch_size: int = 128
    num_steps: int = 100_000
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    num_workers: int = 4
    log_every: int = 100

    prompt_len: int = 32
    forward_recursion_steps: int = 450
    num_rollouts: int = 1_000
    chunk_size: int = 200
    theta_stars: tuple[float, ...] = (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    )

    seed: int = 0
    pmc_seed: int = 1
    device: str = "auto"
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    plot_dpi: int = 400

    def validate(self) -> None:
        positive_integers = (
            self.d_model,
            self.d_mlp,
            self.n_layers,
            self.n_heads,
            self.d_head,
            self.seq_len,
            self.batch_size,
            self.num_steps,
            self.log_every,
            self.prompt_len,
            self.forward_recursion_steps,
            self.num_rollouts,
            self.chunk_size,
            self.plot_dpi,
        )
        if min(positive_integers) < 1:
            raise ValueError("Dimensions, counts, and plotting DPI must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if len(self.theta_stars) != 9:
            raise ValueError("The paper figure requires exactly nine theta values.")
        if any(theta <= 0 or theta >= 1 for theta in self.theta_stars):
            raise ValueError("theta_stars must lie strictly between zero and one.")
        required_context = 1 + self.prompt_len + self.forward_recursion_steps
        if required_context > self.seq_len:
            raise ValueError(
                f"BOS, prompt, and rollout require context {required_context}, "
                f"but seq_len is {self.seq_len}."
            )


@dataclass(frozen=True)
class PMCResults:
    """Samples and sufficient statistics behind the appendix figure."""

    theta_stars: Float[np.ndarray, " prompt"]
    prompts: Int[np.ndarray, "prompt prompt_len"]
    prior_samples: Float[np.ndarray, " rollout"]
    posterior_samples: Float[np.ndarray, "prompt rollout"]
    posterior_alpha: Float[np.ndarray, " prompt"]
    posterior_beta: Float[np.ndarray, " prompt"]


def train_beta_bernoulli(config: BetaBernoulliConfig) -> UnsupervisedPFN:
    """Train the continuous-prior model and save its final checkpoint."""
    alpha = torch.tensor(
        [config.prior_beta, config.prior_alpha],
        dtype=torch.float32,
    )
    bos_token = 2
    data_generator = BOSGenerator(make_bau_generator(alpha), bos_token)
    model_config = UnsupervisedConfig(
        d_model=config.d_model,
        d_mlp=config.d_mlp,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_head=config.d_head,
        n_ctx=config.seq_len,
        d_vocab=3,
        d_vocab_out=2,
        input_type="discrete",
        prediction_type="distribution",
        act_fn="gelu",
    )
    training_config = TrainingConfig(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        num_workers=config.num_workers,
        num_steps=config.num_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        log_every=config.log_every,
        save_checkpoint=True,
        checkpoint_schedule="linear",
        save_every=config.num_steps,
        checkpoint_dir=str(config.output_dir / "checkpoints"),
        log_file=str(config.output_dir / "training_log.json"),
        device=config.device,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        wandb_run_name="beta-bernoulli",
        wandb_log_model=False,
        seed=config.seed,
    )
    model = train(
        data_generator,
        model_config,
        training_config,
        data_config=BetaBernoulliDataConfig(
            prior_alpha=config.prior_alpha,
            prior_beta=config.prior_beta,
        ),
    )
    assert isinstance(model, UnsupervisedPFN)
    return model


def sample_prompts(
    config: BetaBernoulliConfig,
) -> Int[torch.Tensor, "prompt prompt_len"]:
    """Sample one fixed prompt for each ground-truth Bernoulli probability."""
    return torch.stack(
        [
            torch.bernoulli(
                torch.full((config.prompt_len,), theta),
                generator=torch.Generator().manual_seed(config.seed + index + 1),
            ).long()
            for index, theta in enumerate(config.theta_stars)
        ]
    )


def compute_pmc_results(
    model: UnsupervisedPFN, config: BetaBernoulliConfig
) -> PMCResults:
    """Recover the model's implicit prior and prompt-conditioned posteriors."""
    if hasattr(model, "task_distribution"):
        raise ValueError(
            "Checkpoint contains a finite task pool, not a continuous Beta prior."
        )
    if (
        model.config.input_type != "discrete"
        or model.config.prediction_type != "distribution"
        or model.config.d_vocab != 3
        or model.config.d_vocab_out != 2
    ):
        raise ValueError("Checkpoint is not a binary BOS Beta-Bernoulli model.")
    required_context = 1 + config.prompt_len + config.forward_recursion_steps
    if model.config.n_ctx < required_context:
        raise ValueError(
            f"Model context {model.config.n_ctx} is shorter than the required "
            f"context {required_context}."
        )

    model.eval()
    prompts = sample_prompts(config)
    torch.manual_seed(config.pmc_seed)
    prior_theta = predictive_monte_carlo_theta_chunked(
        model=model,
        vocab_size=2,
        forward_recursion_steps=config.forward_recursion_steps,
        num_rollouts=config.num_rollouts,
        prompt=None,
        bos_token=2,
        chunk_size=config.chunk_size,
    )
    assert prior_theta.shape == (config.num_rollouts, 2)

    posterior_samples = np.empty(
        (len(config.theta_stars), config.num_rollouts),
        dtype=np.float32,
    )
    for index, prompt in enumerate(prompts):
        posterior_theta = predictive_monte_carlo_theta_chunked(
            model=model,
            vocab_size=2,
            forward_recursion_steps=config.forward_recursion_steps,
            num_rollouts=config.num_rollouts,
            prompt=prompt,
            bos_token=2,
            chunk_size=config.chunk_size,
        )
        assert posterior_theta.shape == (config.num_rollouts, 2)
        posterior_samples[index] = posterior_theta[:, 1]

    num_ones = prompts.sum(dim=1).numpy()
    posterior_alpha = config.prior_alpha + num_ones
    posterior_beta = config.prior_beta + config.prompt_len - num_ones
    return PMCResults(
        theta_stars=np.asarray(config.theta_stars),
        prompts=prompts.numpy(),
        prior_samples=prior_theta[:, 1],
        posterior_samples=posterior_samples,
        posterior_alpha=posterior_alpha,
        posterior_beta=posterior_beta,
    )


def plot_pmc_grid(
    results: PMCResults, config: BetaBernoulliConfig, output_stem: Path
) -> None:
    """Save the nine-panel PMC comparison as raster and vector figures."""
    assert results.prompts.shape == (9, config.prompt_len)
    assert results.posterior_samples.shape == (9, config.num_rollouts)
    assert results.prior_samples.shape == (config.num_rollouts,)

    x = np.linspace(1e-4, 1.0 - 1e-4, 500)
    bins = np.linspace(0.0, 1.0, 61)
    colors = plt.get_cmap("viridis")
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(9, 7.8),
        dpi=config.plot_dpi,
        constrained_layout=True,
    )

    for index, ax in enumerate(axes.flat):
        num_ones = int(results.prompts[index].sum())
        num_zeros = config.prompt_len - num_ones
        ax.hist(
            results.prior_samples,
            bins=bins,
            density=True,
            color="grey",
            alpha=0.25,
            label="PMC prior",
        )
        ax.plot(
            x,
            beta_distribution.pdf(x, config.prior_alpha, config.prior_beta),
            color="grey",
            linestyle="--",
            linewidth=1.5,
            label="Beta prior",
        )
        ax.hist(
            results.posterior_samples[index],
            bins=bins,
            density=True,
            color=colors(0.65),
            alpha=0.35,
            label="PMC posterior",
        )
        ax.plot(
            x,
            beta_distribution.pdf(
                x,
                results.posterior_alpha[index],
                results.posterior_beta[index],
            ),
            color=colors(0.2),
            linewidth=1.5,
            label="Beta posterior",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_title(
            rf"$\theta^\star={results.theta_stars[index]:.1f}$: "
            f"{num_ones} ones, {num_zeros} zeros",
            fontsize=10,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\theta$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Beta-Bernoulli PMC prior and posterior")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=config.plot_dpi,
        bbox_inches="tight",
    )
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(config: BetaBernoulliConfig) -> None:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, default=str)

    if config.checkpoint_path is None:
        model = train_beta_bernoulli(config)
    else:
        model, _, _ = load_checkpoint(config.checkpoint_path, device=config.device)
        assert isinstance(model, UnsupervisedPFN)

    results = compute_pmc_results(model, config)
    np.savez_compressed(
        config.output_dir / "beta_bernoulli_pmc_samples.npz",
        theta_stars=results.theta_stars,
        prompts=results.prompts,
        prior_samples=results.prior_samples,
        posterior_samples=results.posterior_samples,
        posterior_alpha=results.posterior_alpha,
        posterior_beta=results.posterior_beta,
    )
    plot_pmc_grid(
        results,
        config,
        config.output_dir / "beta_bernoulli_pmc_grid",
    )


if __name__ == "__main__":
    main(tyro.cli(BetaBernoulliConfig))
