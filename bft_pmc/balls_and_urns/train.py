"""Training CLI for the Dirichlet-Multinomial (balls and urns) setting."""

from dataclasses import dataclass

import torch
import tyro

from bft_pmc.balls_and_urns.data import (
    BOSGenerator,
    make_bau_generator,
    make_discrete_bau_generator,
)
from bft_pmc.pos_encoding import PositionalEncoding, positional_encoding_kwargs
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
    """Train one balls-and-urns PFN on a finite pool of Dirichlet urns.

    The defaults are a small smoke-test model, not the paper's: the paper's
    runs (slurm/train.slurm) use --vocab-size 12 --d-model 64 --d-mlp 256
    --n-heads 1 --d-head 64 --batch-size 64 --num-steps 100000
    --learning-rate 5e-4 --warmup-steps 0 --seed 42.
    """

    d_model: int = 128
    """Residual stream width."""
    d_mlp: int | None = None
    """MLP hidden width; None lets TransformerLens use 4 x d_model."""
    n_layers: int = 2
    """Transformer blocks."""
    n_heads: int = 4
    """Attention heads per block."""
    d_head: int = 32
    """Width of each attention head."""
    n_ctx: int = 256
    """Token context: BOS plus 255 observations, the same 256 positions as the
    linear-regression and Markov models."""
    act_fn: str = "gelu"
    """MLP activation."""
    pos_encoding: PositionalEncoding = "learned"
    """learned (absolute table), rope (rotary) or none."""
    vocab_size: int = 4
    """Ball colours V; the latent task is a point on the (V - 1)-simplex."""
    alpha_value: float = 1.0
    """Concentration of the symmetric Dirichlet the urn pool is drawn from."""
    num_tasks: int = 1024
    """Task diversity M: urns in the training pool."""
    batch_size: int = 128
    """Sequences per optimisation step."""
    seq_len: int = 256
    """Tokens per training sequence, BOS included."""
    num_steps: int = 50000
    """Optimisation steps."""
    learning_rate: float = 1e-3
    """Peak AdamW learning rate."""
    warmup_steps: int = 500
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
    checkpoint_root: str = "checkpoints/bau"
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
        if self.seq_len > self.n_ctx:
            raise ValueError("seq_len must fit within the model context n_ctx.")
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
        **positional_encoding_kwargs(cfg.pos_encoding),
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
        eval_data_generator=eval_gen,
        data_config=data_cfg,
    )


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
