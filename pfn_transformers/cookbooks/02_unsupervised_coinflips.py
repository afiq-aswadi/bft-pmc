"""
Cookbook 2: Unsupervised Beta-Bernoulli (Coinflips)

Demonstrates training a PFN to learn coinflip sequences with varying bias:
    p ~ Beta(2, 2)
    observations ~ Bernoulli(p)

The PFN learns to perform posterior updating: as it sees more flips,
predictions converge to the true bias.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import tyro

from pfn_transformerlens import (
    Likelihood,
    Prior,
    TrainingConfig,
    UnsupervisedBayesian,
    UnsupervisedConfig,
    sample_batch,
    train,
)


@dataclass
class ExpConfig:
    """Configuration for coinflips experiment."""

    # prior parameters
    prior_alpha: float = 10.0
    prior_beta: float = 2.0

    # model architecture parameters
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_head: int = 32
    n_ctx: int = 32
    d_vocab: int = 2  # binary outcomes: 0 or 1

    # training parameters
    batch_size: int = 128
    seq_len: int = 32
    num_steps: int = 300
    learning_rate: float = 1e-3
    warmup_steps: int = 30
    log_every: int = 100
    use_wandb: bool = False

    # evaluation parameters
    true_bias: float = 0.7
    num_flips: int = 16


def main(config: ExpConfig) -> None:
    output_dir = Path(__file__).parent / "outputs" / "02_coinflips"
    output_dir.mkdir(parents=True, exist_ok=True)

    # prior: Beta distribution over bias
    prior = Prior(
        base_distribution=torch.distributions.Beta(
            torch.tensor(config.prior_alpha), torch.tensor(config.prior_beta)
        )
    )

    # likelihood: Bernoulli parameterized by sampled bias
    def bernoulli_parameterizer(theta: torch.Tensor, x: torch.Tensor) -> dict:
        # broadcast theta (bias) to match sequence length
        seq_len = x.shape[0]
        return {"probs": theta.expand(seq_len)}

    likelihood = Likelihood(
        base_distribution=torch.distributions.Bernoulli(0.5),  # dummy base
        parameterizer=bernoulli_parameterizer,
        input_dim=1,
    )

    # data generator for discrete Bernoulli observations
    data_gen = UnsupervisedBayesian(prior=prior, likelihood=likelihood)

    # model config: discrete inputs (0 or 1)
    model_config = UnsupervisedConfig(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_head=config.d_head,
        n_ctx=config.n_ctx,
        d_vocab=config.d_vocab,  # binary outcomes: 0 or 1
        input_type="discrete",
        prediction_type="distribution",
        act_fn="gelu",
    )

    # training
    training_config = TrainingConfig(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        num_steps=config.num_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        log_every=config.log_every,
        use_wandb=config.use_wandb,
    )

    model = train(data_gen, model_config, training_config)

    # test: show posterior updating
    true_bias = config.true_bias
    num_flips = config.num_flips

    # generate coinflips
    _, flips = sample_batch(data_gen, 1, num_flips)

    # get autoregressive predictions for all positions at once
    with torch.no_grad():
        pred = model.predict_on_prompt(flips)

    # extract predicted probabilities of heads (class 1) for each position
    predicted_probs = pred.probs[0, :, 1].cpu().numpy()

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # plot 1: predicted probability vs observations
    ax = axes[0]
    ax.plot(range(1, num_flips + 1), predicted_probs, label="Predicted P(heads)")
    ax.axhline(true_bias, color="r", linestyle="--", label="True bias")
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Predicted P(heads)")
    ax.set_title("Posterior Updating with More Data")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # plot 2: Bayes PPD vs predicted
    # For Beta-Bernoulli, posterior predictive is Beta(alpha + heads, beta + tails)
    # Mean of Beta(a,b) is a/(a+b)
    flips_np = flips[0, :].cpu().numpy()  # shape: (num_flips,)
    heads_cumsum = flips_np.cumsum()  # shape: (num_flips,)
    bayes_ppd_mean = (config.prior_alpha + heads_cumsum) / (
        config.prior_alpha + config.prior_beta + torch.arange(1, num_flips + 1)
    )

    ax = axes[1]
    ax.plot(range(1, num_flips + 1), bayes_ppd_mean, label="Bayes PPD mean")
    ax.plot(range(1, num_flips + 1), predicted_probs, label="PFN prediction")
    ax.axhline(true_bias, color="r", linestyle="--", label="True bias")
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Probability")
    ax.set_title("Bayes PPD vs Predicted Probabilities")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "posterior_updating.png", dpi=400)
    plt.close()

    print(f"Saved: {output_dir / 'posterior_updating.png'}")


if __name__ == "__main__":
    tyro.cli(main)
