"""Standalone training CLI driven by YAML sweep configs via sweeps/run_yaml_sweep.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from pfn_transformerlens import sample_batch
from pfn_transformerlens.model.bucketizer import estimate_riemann_borders
from pfn_transformerlens.model.configs.regression import SupervisedRegressionPFNConfig
from pfn_transformerlens.sampler.data_generator import DeterministicFunctionGenerator
from pfn_transformerlens.train import TrainingConfig, train
from pfn_transformerlens.wandb_utils import RunNameScheme, create_run_name
from linear_regression.likelihoods import linear_regression
from linear_regression.priors import DiscretePrior


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
    # architecture
    d_model: int = 512
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 512
    d_head: int = 128
    d_vocab: int = 256
    n_ctx: int = 128
    act_fn: str = "gelu"
    input_dim: int = 8

    # model behavior
    mask_type: Literal["autoregressive-pfn", "gpt2"] = "autoregressive-pfn"
    prediction_type: Literal["distribution", "point"] = "distribution"
    bucket_type: Literal["uniform", "riemann"] = "uniform"
    bucket_support: Literal["unbounded", "bounded"] = "unbounded"
    y_min: float = -10.0
    y_max: float = 10.0
    num_riemann_samples: int = 10000

    # data
    prior_type: Literal["discrete"] = "discrete"
    function_type: Literal["linear"] = "linear"
    num_tasks: int = 65536
    noise_std: float = 0.5

    # training
    batch_size: int = 256
    seq_len: int = 64
    num_steps: int = 150000
    learning_rate: float = 1e-4
    warmup_steps: int = 1000

    # checkpointing
    save_checkpoint: bool = True
    checkpoint_schedule: str = "logarithmic"
    save_every: int = 1000
    linear_checkpoint_interval: int = 500
    n_log_checkpoints: int = 200
    checkpoint_root: str = "checkpoints/lr"

    # eval
    eval_every: int = 1000
    eval_batches: int = 10

    # logging
    log_file: str | None = None

    # wandb
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_log_model: bool = False
    seed: int | None = None

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
            "linear_regression/train.py currently supports only the discrete prior."
        )
    if cfg.function_type != "linear":
        raise ValueError(
            "linear_regression/train.py currently supports only the linear likelihood."
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
