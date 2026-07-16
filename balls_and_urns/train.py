"""Training CLI for the Dirichlet-Multinomial (balls and urns) setting."""

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import tyro

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from balls_and_urns.data import (
    BOSGenerator,
    make_bau_generator,
    make_discrete_bau_generator,
)
from pfn_transformerlens import UnsupervisedConfig
from pfn_transformerlens.train import TrainingConfig, train
from pfn_transformerlens.wandb_utils import RunNameScheme, create_run_name


@dataclass
class DataConfig:
    """Data configuration for wandb logging."""

    prior_type: str
    vocab_size: int
    alpha_value: float
    num_tasks: int


SCHEME = RunNameScheme.from_templates(
    model={
        "d_model": None,
        "n_layers": None,
    },
    data={
        "prior_type": None,
        "vocab_size": None,
        "num_tasks": None,
    },
)


@dataclass
class TrainConfig:
    # architecture
    d_model: int = 128
    d_mlp: int | None = None
    n_layers: int = 2
    n_heads: int = 4
    d_head: int = 32
    n_ctx: int = 128
    act_fn: str = "gelu"

    # task
    vocab_size: int = 4
    alpha_value: float = 1.0
    num_tasks: int = 1024

    # training
    batch_size: int = 128
    seq_len: int = 64
    num_steps: int = 50000
    learning_rate: float = 1e-3
    warmup_steps: int = 500

    # checkpointing
    save_checkpoint: bool = True
    checkpoint_schedule: str = "logarithmic"
    save_every: int = 1000
    linear_checkpoint_interval: int = 500
    n_log_checkpoints: int = 200
    checkpoint_root: str = "checkpoints/bau"

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
        dimensions = [
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.d_head,
            self.n_ctx,
            self.vocab_size,
            self.num_tasks,
            self.batch_size,
            self.seq_len,
            self.num_steps,
        ]
        if min(dimensions) < 1:
            raise ValueError("Model, data, and training dimensions must be positive.")
        if self.d_model != self.n_heads * self.d_head:
            raise ValueError("d_model must equal n_heads times d_head.")
        if self.alpha_value <= 0 or self.learning_rate <= 0:
            raise ValueError("alpha_value and learning_rate must be positive.")


def main(cfg: TrainConfig) -> None:
    cfg.validate()
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    alpha = torch.ones(cfg.vocab_size) * cfg.alpha_value
    bos_token = cfg.vocab_size  # BOS is one beyond data tokens

    # training data: discrete prior (M fixed simplexes), with BOS prepended
    base_gen, theta_pool = make_discrete_bau_generator(alpha, cfg.num_tasks)
    data_gen = BOSGenerator(base_gen, bos_token)
    # eval data: true Dirichlet prior, with BOS prepended
    eval_gen = BOSGenerator(make_bau_generator(alpha), bos_token)

    model_cfg = UnsupervisedConfig(
        d_model=cfg.d_model,
        d_mlp=cfg.d_mlp,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_head=cfg.d_head,
        n_ctx=cfg.n_ctx,
        d_vocab=cfg.vocab_size + 1,  # +1 for BOS token (input embedding)
        d_vocab_out=cfg.vocab_size,  # output only data tokens, not BOS
        input_type="discrete",
        prediction_type="distribution",
        act_fn=cfg.act_fn,
    )

    data_cfg = DataConfig(
        prior_type="discrete",
        vocab_size=cfg.vocab_size,
        alpha_value=cfg.alpha_value,
        num_tasks=cfg.num_tasks,
    )

    run_name = create_run_name(
        base="bau",
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
        eval_data_generator=eval_gen,
        data_config=data_cfg,
    )


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
