"""Train one linear-regression PFN on a finite pool of Gaussian tasks.

    uv run train.py lr --num-tasks 32 --pos-encoding none --seed 42

The paper's sweep trains one model per task diversity M and positional
encoding: see slurm/train.slurm, or configs/sweeps/lr_task_diversity_sweep.yaml
for ``uv run train.py sweep``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from bft_pmc.linear_regression.likelihoods import linear_regression
from bft_pmc.linear_regression.priors import DiscretePrior
from bft_pmc.pos_encoding import PositionalEncoding, positional_encoding_kwargs
from pfn_transformerlens import sample_batch
from pfn_transformerlens.model.bucketizer import estimate_riemann_borders
from pfn_transformerlens.model.configs.regression import SupervisedRegressionPFNConfig
from pfn_transformerlens.sampler.data_generator import DeterministicFunctionGenerator
from pfn_transformerlens.train import TrainingConfig, train
from pfn_transformerlens.wandb_utils import RunNameScheme, create_run_name


@dataclass
class DataConfig:
    """Run metadata used for naming and logging."""

    prior_type: str
    function_type: str
    num_tasks: int
    input_dim: int
    noise_std: float


# RunNameScheme defines which config fields are used for model identification.
# Used by create_run_name() to generate consistent wandb run names.
SCHEME = RunNameScheme.from_templates(
    model={
        "d_model": None,
        "n_layers": None,
        "mask_type": None,
    },
    data={
        "prior_type": None,
        "function_type": None,
        "num_tasks": None,
    },
)


def estimate_borders_from_data(
    data_gen: DeterministicFunctionGenerator,
    num_samples: int,
    num_buckets: int,
    seq_len: int,
) -> torch.Tensor:
    """Estimate riemann borders by sampling from data generator."""
    if num_samples < 1 or num_buckets < 2 or seq_len < 1:
        raise ValueError(
            "num_samples and seq_len must be positive; num_buckets must be at least 2."
        )
    num_batches = (num_samples + seq_len - 1) // seq_len
    all_ys: list[torch.Tensor] = []

    for _ in range(num_batches):
        x, y = sample_batch(data_gen, batch_size=1, seq_len=seq_len)
        all_ys.append(y)

    ys = torch.cat(all_ys, dim=0).flatten()
    return estimate_riemann_borders(ys, num_buckets=num_buckets)


@dataclass
class TrainConfig:
    """Train one linear-regression PFN on a finite pool of Gaussian tasks.

    The defaults are the paper's architecture and optimiser; the paper's runs
    (slurm/train.slurm) also set --num-tasks, --pos-encoding and --seed 42.
    """

    d_model: int = 512
    """Residual stream width."""
    n_layers: int = 2
    """Transformer blocks."""
    n_heads: int = 4
    """Attention heads per block."""
    d_mlp: int = 512
    """MLP hidden width."""
    d_head: int = 128
    """Width of each attention head."""
    d_vocab: int = 256
    """Response buckets of the output distribution."""
    n_ctx: int = 512
    """Token context: 512 positions hold 256 interleaved (x, y) pairs, the same
    256 observations as the balls-and-urns and Markov models."""
    act_fn: str = "gelu"
    """MLP activation."""
    input_dim: int = 8
    """Dimension of x, and of each task's weight vector."""
    pos_encoding: PositionalEncoding = "learned"
    """learned (absolute table), rope (rotary) or none."""
    mask_type: Literal["autoregressive-pfn", "gpt2"] = "autoregressive-pfn"
    """Attention mask over the interleaved (x, y) tokens."""
    prediction_type: Literal["distribution", "point"] = "distribution"
    """distribution predicts a categorical over response buckets; point a
    scalar."""
    bucket_type: Literal["uniform", "riemann"] = "uniform"
    """uniform buckets between y_min and y_max, or riemann quantile buckets
    estimated from sampled data."""
    bucket_support: Literal["unbounded", "bounded"] = "unbounded"
    """Whether the outermost buckets extend to infinity."""
    y_min: float = -10.0
    """Lower edge of the uniform buckets."""
    y_max: float = 10.0
    """Upper edge of the uniform buckets."""
    num_riemann_samples: int = 10000
    """Responses sampled to place the riemann buckets."""
    prior_type: Literal["discrete"] = "discrete"
    """Task prior: uniform over a fixed pool of num_tasks Gaussian draws."""
    function_type: Literal["linear"] = "linear"
    """Likelihood: y = w^T x + noise."""
    num_tasks: int = 65536
    """Task diversity M: size of the training task pool."""
    noise_std: float = 0.5
    """Observation-noise standard deviation."""
    batch_size: int = 256
    """Sequences per optimisation step."""
    seq_len: int = 256
    """(x, y) pairs per training sequence."""
    num_steps: int = 150000
    """Optimisation steps."""
    learning_rate: float = 1e-4
    """Peak AdamW learning rate."""
    warmup_steps: int = 1000
    """Linear learning-rate warmup."""
    save_checkpoint: bool = True
    """Write checkpoints during training."""
    checkpoint_schedule: str = "linear"
    """linear saves every save_every steps; logarithmic saves every
    linear_checkpoint_interval steps plus n_log_checkpoints log-spaced steps,
    the dense-early schedule for training dynamics."""
    save_every: int = 500
    """Steps between checkpoints of the linear schedule."""
    linear_checkpoint_interval: int = 500
    """Linear part of the logarithmic schedule."""
    n_log_checkpoints: int = 200
    """Log-spaced checkpoints of the logarithmic schedule."""
    checkpoint_root: str = "checkpoints/lr"
    """Directory the checkpoints are written to (a W&B run-id subdirectory of
    it when W&B is on)."""
    eval_every: int = 1000
    """Steps between evaluations on fresh tasks from the population prior."""
    eval_batches: int = 10
    """Batches averaged per evaluation."""
    log_file: str | None = None
    """Write the training and evaluation metrics to this JSON file."""
    use_wandb: bool = False
    """Log to Weights & Biases (needs the wandb extra)."""
    wandb_project: str | None = None
    """W&B project; falls back to WANDB_PROJECT."""
    wandb_entity: str | None = None
    """W&B entity; falls back to WANDB_ENTITY."""
    wandb_log_model: bool = False
    """Upload checkpoints to W&B as artifacts."""
    seed: int | None = None
    """Random seed; the paper uses 42. None leaves torch unseeded."""

    def validate(self) -> None:
        if (
            min(
                self.d_model,
                self.n_layers,
                self.n_heads,
                self.d_head,
                self.d_vocab,
                self.n_ctx,
                self.input_dim,
                self.num_tasks,
                self.batch_size,
                self.seq_len,
                self.num_steps,
            )
            < 1
        ):
            raise ValueError("Model, data, and training dimensions must be positive.")
        if self.d_model != self.n_heads * self.d_head:
            raise ValueError("d_model must equal n_heads times d_head.")
        # x and y occupy one position each, so a length-S sequence needs 2S positions.
        if 2 * self.seq_len > self.n_ctx:
            raise ValueError("2 * seq_len must fit within the model context n_ctx.")
        if self.learning_rate <= 0 or self.noise_std < 0:
            raise ValueError(
                "learning_rate must be positive and noise_std non-negative."
            )
        if self.bucket_type not in {"uniform", "riemann"}:
            raise ValueError(f"Unsupported bucket_type: {self.bucket_type!r}.")
        if self.bucket_type == "uniform" and self.y_min >= self.y_max:
            raise ValueError("y_min must be smaller than y_max for uniform buckets.")


def _validate_config(cfg: TrainConfig) -> None:
    """Reject stale config values that this CLI does not actually support."""
    cfg.validate()
    if cfg.prior_type != "discrete":
        raise ValueError(
            "bft_pmc/linear_regression/train.py currently supports only the discrete prior."
        )
    if cfg.function_type != "linear":
        raise ValueError(
            "bft_pmc/linear_regression/train.py currently supports only the linear likelihood."
        )


def main(cfg: TrainConfig) -> None:
    _validate_config(cfg)
    prior = DiscretePrior(num_tasks=cfg.num_tasks, task_size=cfg.input_dim)
    x_dist = torch.distributions.Normal(0.0, 1.0)
    data_gen = DeterministicFunctionGenerator(
        prior=prior,
        function=linear_regression,
        input_dim=cfg.input_dim,
        noise_std=cfg.noise_std,
        x_distribution=x_dist,
    )

    riemann_borders = None
    y_min = None
    y_max = None

    if cfg.bucket_type == "riemann":
        riemann_borders = estimate_borders_from_data(
            data_gen, cfg.num_riemann_samples, cfg.d_vocab, cfg.seq_len
        )
    elif cfg.bucket_type == "uniform":
        y_min = cfg.y_min
        y_max = cfg.y_max

    model_cfg = SupervisedRegressionPFNConfig(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_head=cfg.d_head,
        d_mlp=cfg.d_mlp,
        d_vocab=cfg.d_vocab,
        input_dim=cfg.input_dim,
        n_ctx=cfg.n_ctx,
        act_fn=cfg.act_fn,
        mask_type=cfg.mask_type,
        prediction_type=cfg.prediction_type,
        bucket_type=cfg.bucket_type,
        bucket_support=cfg.bucket_support,
        riemann_borders=riemann_borders,
        y_min=y_min,
        y_max=y_max,
        **positional_encoding_kwargs(cfg.pos_encoding),
    )

    # eval generator uses Gaussian (continuous) prior for generalization measurement
    eval_prior = torch.distributions.Normal(0.0, 1.0)
    eval_data_gen = DeterministicFunctionGenerator(
        prior=eval_prior,
        function=linear_regression,
        input_dim=cfg.input_dim,
        noise_std=cfg.noise_std,
        x_distribution=x_dist,
    )

    data_cfg = DataConfig(
        prior_type=cfg.prior_type,
        function_type=cfg.function_type,
        num_tasks=cfg.num_tasks,
        input_dim=cfg.input_dim,
        noise_std=cfg.noise_std,
    )

    run_name = create_run_name(
        base="sweep",
        model_config=model_cfg,
        data_config=data_cfg,
        scheme=SCHEME,
        extra={"pos": cfg.pos_encoding},
    )

    train_cfg = TrainingConfig(
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        num_steps=cfg.num_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        eval_every=cfg.eval_every,
        eval_batches=cfg.eval_batches,
        log_file=cfg.log_file,
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        wandb_run_name=run_name,
        wandb_log_model=cfg.wandb_log_model,
        save_checkpoint=cfg.save_checkpoint,
        checkpoint_schedule=cfg.checkpoint_schedule,
        save_every=cfg.save_every,
        linear_checkpoint_interval=cfg.linear_checkpoint_interval,
        n_log_checkpoints=cfg.n_log_checkpoints,
        checkpoint_dir=cfg.checkpoint_root,
        seed=cfg.seed,
    )

    train(
        data_gen,
        model_cfg,
        train_cfg,
        eval_data_generator=eval_data_gen,
        data_config=data_cfg,
    )


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
