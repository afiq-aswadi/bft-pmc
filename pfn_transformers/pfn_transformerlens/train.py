"""General training loop for Prior-Fitted Network (PFN).

This module provides a reusable training pipeline where callers can provide:
- Prior distribution
- Likelihood distribution
- PFN model configuration (PFNConfig)
- Training configuration (TrainingConfig)

It returns a trained PFNModel and handles checkpointing, logging, and basic
metrics (cross-entropy loss).

For training w/ multiple GPUs, just use simple-gpu-scheduler (https://pypi.org/project/simple-gpu-scheduler/)

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from pfn_transformerlens.checkpointing import (
    CheckpointMetadata,
    _get_git_hash,
    get_logarithmic_checkpoint_steps,
    save_checkpoint,
)
from pfn_transformerlens.model.configs import (
    BasePFNConfig,
    ClassificationPFNConfig,
    SupervisedRegressionPFNConfig,
    UnsupervisedPFNConfig,
)
from pfn_transformerlens.model.PFN import BasePFN, PFNModel, UnsupervisedPFN
from pfn_transformerlens.model.bucketizer import Bucketizer
from pfn_transformerlens.sampler.data_generator import DataGenerator
from pfn_transformerlens.sampler.dataloader import build_dataloader, sample_batch
from pfn_transformerlens.wandb_logger import WandbLogger


@dataclass
class TrainingConfig:
    """Training configuration for PFN.

    Attributes:
        batch_size: DataLoader batch size.
        seq_len: Sequence length (x,y are interleaved internally by the model).
        num_workers: DataLoader worker processes for data generation.
        pin_memory: Whether to pin host memory for faster host-to-device copies.
        prefetch_factor: Number of batches prefetched per worker (num_workers > 0).
        persistent_workers: Keep workers alive between iterations (num_workers > 0).
        num_steps: Number of optimization steps.
        learning_rate: Base learning rate.
        weight_decay: Weight decay for AdamW.
        use_warmup: Whether to use linear LR warmup.
        warmup_steps: Linear LR warmup steps (only used if use_warmup=True).
        use_grad_clip: Whether to apply gradient clipping.
        grad_clip: Gradient clipping max norm (only used if use_grad_clip=True).
        log_every: Steps between progress logs (avg over window).
        log_distributional_mse: Whether to compute an approximate MSE metric when
            training density models (adds an argmax over buckets each step).
        save_checkpoint: Whether to save checkpoints during training.
        checkpoint_schedule: Checkpoint scheduling strategy. Options:
            - "linear": Save every save_every steps (default, backwards compatible)
            - "logarithmic": Combined linear + log spacing per https://arxiv.org/pdf/2501.17745
        save_every: Steps between checkpoints (only used for "linear" schedule).
        linear_checkpoint_interval: Interval for linear checkpoints in logarithmic schedule.
        n_log_checkpoints: Number of logarithmically spaced checkpoints (N parameter).
        checkpoint_dir: Directory to save checkpoints.
        eval_every: Steps between evaluations (None = no evaluation).
        eval_batches: Number of batches to average for eval metrics.
        device: "auto", "cuda", "mps", or "cpu".
        use_wandb: Whether to enable wandb logging.
        wandb_project: Wandb project name (falls back to WANDB_PROJECT env var).
        wandb_entity: Wandb entity/username (falls back to WANDB_ENTITY env var).
        wandb_run_name: Optional run name (wandb auto-generates if None).
            Note: The parameter is wandb_run_name, not run_name.
        wandb_log_model: Whether to upload checkpoints as wandb artifacts.
        wandb_tags: Optional list of tags for the wandb run.
        wandb_notes: Optional notes/description for the wandb run.
        log_file: Path to write JSON metrics log. If None, no file is written.
        seed: Random seed for reproducibility. If None, no seed is set.
    """

    batch_size: int = 32
    seq_len: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False
    num_steps: int = 10000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    use_warmup: bool = True
    warmup_steps: int = 500
    use_grad_clip: bool = True
    grad_clip: float = 1.0
    log_every: int = 100
    log_distributional_mse: bool = False
    save_checkpoint: bool = True
    checkpoint_schedule: str = "linear"
    save_every: int = 1000
    linear_checkpoint_interval: int = 100
    n_log_checkpoints: int = 1000
    checkpoint_dir: str = "checkpoints"
    eval_every: int | None = None
    eval_batches: int = 10
    device: str = "auto"
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_log_model: bool = True
    wandb_tags: list[str] | None = None
    wandb_notes: str | None = None
    log_file: str | None = None
    seed: int | None = None

    def get_device(self) -> str:
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.device


def _compute_distributional_nll(
    logits: Float[torch.Tensor, "batch seq d_vocab"],
    targets: Float[torch.Tensor, "batch seq"],
    bucketizer: Bucketizer,
    compute_mse: bool = False,
) -> tuple[Float[torch.Tensor, ""], dict[str, float | str]]:
    """Compute distributional NLL loss and optional MSE metric.

    Args:
        logits: Model output logits over bucket vocabulary.
        targets: Target continuous values.
        bucketizer: Bucketizer for converting logits to densities.
        compute_mse: Whether to compute approximate MSE via argmax over buckets.

    Returns:
        Tuple of (loss, metrics_dict).
    """

    log_density_at_targets = bucketizer.log_density_at_values(logits, targets)
    loss = -log_density_at_targets.mean()
    metrics = {"loss": float(loss.item()), "loss_type": "NLL"}

    if compute_mse:
        with torch.no_grad():
            log_densities = bucketizer.log_bucket_densities(logits)
            pred_buckets = log_densities.argmax(dim=-1)
            pred_y = bucketizer.decode(pred_buckets)
            mse = nn.functional.mse_loss(pred_y, targets, reduction="mean")
        metrics["mse"] = float(mse.item())

    return loss, metrics


def compute_unsupervised_loss(
    model: UnsupervisedPFN,
    y: Float[torch.Tensor, "batch seq"],
    *,
    log_distributional_mse: bool = False,
) -> tuple[Float[torch.Tensor, ""], dict[str, float | str]]:
    """Compute loss for unsupervised (next-token/next-value prediction) models.

    Args:
        model: Unsupervised PFN model.
        y: Observation sequence.
        log_distributional_mse: Whether to compute MSE for distributional predictions.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    logits = model(y)
    y = y.to(logits.device)

    if (
        model.config.input_type == "discrete"
        and model.config.prediction_type == "distribution"
    ):
        # Discrete next-token prediction with cross-entropy
        logits_flat = logits[:, :-1, :].reshape(-1, model.config.d_vocab_out)
        targets_flat = y[:, 1:].long().reshape(-1)
        loss = nn.functional.cross_entropy(logits_flat, targets_flat, reduction="mean")
        metrics = {"loss": float(loss.item()), "loss_type": "CE"}

        with torch.no_grad():
            preds = logits[:, :-1, :].argmax(dim=-1)
            acc = (preds == y[:, 1:].long()).float().mean()
        metrics["accuracy"] = float(acc.item())

        return loss, metrics

    elif (
        model.config.input_type == "continuous"
        and model.config.prediction_type == "point"
    ):
        # Continuous next-value prediction with MSE
        preds = logits[:, :-1, :].squeeze(-1)
        targets = y[:, 1:]
        mse = nn.functional.mse_loss(preds, targets, reduction="mean")
        metrics = {
            "loss": float(mse.item()),
            "mse": float(mse.item()),
            "loss_type": "MSE",
        }
        return mse, metrics

    elif (
        model.config.input_type == "continuous"
        and model.config.prediction_type == "distribution"
    ):
        # Continuous distribution prediction with NLL
        logits_shifted = logits[:, :-1, :]
        targets_shifted = y[:, 1:]
        return _compute_distributional_nll(
            logits_shifted, targets_shifted, model.bucketizer, log_distributional_mse
        )

    else:
        # Discrete + point prediction (rare but valid)
        preds = logits[:, :-1, :].squeeze(-1)
        targets = y[:, 1:].float()
        mse = nn.functional.mse_loss(preds, targets, reduction="mean")
        metrics = {
            "loss": float(mse.item()),
            "mse": float(mse.item()),
            "loss_type": "MSE",
        }
        return mse, metrics


def compute_supervised_loss(
    model: BasePFN,
    x: Float[torch.Tensor, "batch seq input_dim"],
    y: Float[torch.Tensor, "batch seq"],
    *,
    log_distributional_mse: bool = False,
) -> tuple[Float[torch.Tensor, ""], dict[str, float | str]]:
    """Compute loss for supervised (classification or regression) models.

    Args:
        model: Supervised PFN model.
        x: Input features.
        y: Target values or class labels.
        log_distributional_mse: Whether to compute MSE for distributional predictions.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    logits = model(x, y)

    if isinstance(model.config, ClassificationPFNConfig):
        # Classification with cross-entropy
        logits_flat = logits.view(-1, model.config.num_classes)
        targets_flat = y.long().view(-1)
        loss = nn.functional.cross_entropy(logits_flat, targets_flat, reduction="mean")
        metrics = {"loss": float(loss.item()), "loss_type": "CE"}

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == y.long()).float().mean()
        metrics["accuracy"] = float(acc.item())

        return loss, metrics

    if (
        isinstance(model.config, SupervisedRegressionPFNConfig)
        and model.config.prediction_type == "point"
    ):
        # Point regression with MSE
        preds = logits.squeeze(-1)
        mse = nn.functional.mse_loss(preds, y, reduction="mean")
        metrics = {
            "loss": float(mse.item()),
            "mse": float(mse.item()),
            "loss_type": "MSE",
        }
        return mse, metrics

    # Distributional regression with NLL
    return _compute_distributional_nll(
        logits, y, model.bucketizer, log_distributional_mse
    )


def compute_loss(
    model: BasePFN,
    x: Float[torch.Tensor, "batch seq input_dim"] | None,
    y: Float[torch.Tensor, "batch seq"],
    *,
    log_distributional_mse: bool = False,
) -> tuple[Float[torch.Tensor, ""], dict[str, float | str]]:
    """Compute PFN loss and metrics.

    Delegates to specialized functions for supervised vs unsupervised models.

    Args:
        model: PFN model (supervised or unsupervised).
        x: Input features for supervised models. Must be None for unsupervised models.
        y: Target values or observations.
        log_distributional_mse: Whether to compute MSE for distributional predictions.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    if isinstance(model, UnsupervisedPFN):
        assert x is None, "x must be None for unsupervised models"
        return compute_unsupervised_loss(
            model, y, log_distributional_mse=log_distributional_mse
        )
    else:
        assert x is not None, "x must be provided for supervised models"
        return compute_supervised_loss(
            model, x, y, log_distributional_mse=log_distributional_mse
        )


def evaluate_model(
    model: BasePFN,
    data_generator: DataGenerator,
    eval_batches: int,
    seq_len: int,
    batch_size: int,
    device: str,
    log_distributional_mse: bool = False,
) -> dict[str, float | str]:
    """Evaluate model on validation data.

    Args:
        model: PFN model to evaluate.
        data_generator: DataGenerator for validation data.
        eval_batches: Number of batches to average over.
        seq_len: Sequence length for evaluation.
        batch_size: Batch size for evaluation.
        device: Device to run evaluation on.
        log_distributional_mse: Whether to compute MSE for distributional predictions.

    Returns:
        Dictionary of averaged evaluation metrics.
    """
    model.eval()
    total_metrics: dict[str, float | str] = {}

    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = sample_batch(data_generator, batch_size, seq_len)
            if x is not None:
                x = x.to(device)
            y = y.to(device)

            _, metrics = compute_loss(
                model, x, y, log_distributional_mse=log_distributional_mse
            )

            for key, value in metrics.items():
                if isinstance(value, str):
                    # non-numeric values (like loss_type) - just store directly
                    total_metrics[key] = value
                elif isinstance(value, (int, float)):
                    # numeric values - accumulate for averaging
                    existing = total_metrics.get(key, 0.0)
                    if isinstance(existing, (int, float)):
                        total_metrics[key] = existing + value

    # Average metrics (skip non-numeric values)
    avg_metrics = {
        key: value / eval_batches if isinstance(value, (int, float)) else value
        for key, value in total_metrics.items()
    }

    model.train()
    return avg_metrics


def train(
    data_generator: DataGenerator,
    model_config: BasePFNConfig,
    training_config: TrainingConfig,
    *,
    resume_from: str | None = None,
    eval_data_generator: DataGenerator | None = None,
    data_config: Any | None = None,
) -> BasePFN:
    """Run a general PFN training loop.

    Args:
        data_generator: DataGenerator instance that produces training sequences.
            Use ProbabilisticGenerator to wrap prior/likelihood if needed.
        model_config: BasePFNConfig for the transformer backbone.
        training_config: Optimization + runtime configuration.
        resume_from: Optional path to a checkpoint to resume from.
        eval_data_generator: Optional separate DataGenerator for validation.
            If None and eval_every is set, uses training generator for eval.
        data_config: Optional dataclass with data config (e.g., num_tasks, input_dim).
            Logged to wandb for config-based model loading with RunNameScheme.
            Kept as dataclass throughout, only converted to dict at wandb boundary.

    Returns:
        Trained PFN model (BasePFN subclass).
    """
    device = training_config.get_device()
    device_type = torch.device(device).type
    autocast_enabled = device_type in {"cuda", "mps"}
    autocast_device = device_type if autocast_enabled else "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=device_type == "cuda")

    if training_config.seed is not None:
        torch.manual_seed(training_config.seed)

    # wandb logging
    logger = WandbLogger(training_config, model_config, data_config)

    # derive checkpoint directory (make run-scoped when wandb logging enabled)
    checkpoint_root = Path(training_config.checkpoint_dir)
    if logger.enabled and logger.run_id:
        checkpoint_root = checkpoint_root / logger.run_id
    training_config.checkpoint_dir = str(checkpoint_root)

    # Model + data
    model = PFNModel(model_config)
    model = model.to(device)

    dataloader = build_dataloader(data_generator, training_config)

    data_iter = iter(dataloader)

    # AdamW is the supported optimizer for this training loop
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    # LR scheduler with warmup
    if training_config.use_warmup:

        def warmup_lambda(step: int) -> float:
            if step < training_config.warmup_steps:
                return (step + 1) / training_config.warmup_steps
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    else:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    start_step = 0
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_step = int(state.get("step", 0))

    metrics_log: list[dict[str, float | int]] = []

    model.train()
    pbar = tqdm(range(start_step, training_config.num_steps), desc="Training")
    run_loss = 0.0
    run_mse = 0.0
    run_acc = 0.0
    mse_count = 0
    acc_count = 0

    # precompute checkpoint steps based on schedule
    if training_config.checkpoint_schedule == "logarithmic":
        checkpoint_steps_list = get_logarithmic_checkpoint_steps(
            training_steps=training_config.num_steps,
            n_log_checkpoints=training_config.n_log_checkpoints,
            linear_interval=training_config.linear_checkpoint_interval,
        )
        checkpoint_steps_set: set[int] = set(checkpoint_steps_list)
    else:
        checkpoint_steps_set = set()

    for step in pbar:
        x, y = next(data_iter)

        # validate data matches config type (only check first batch)
        if step == start_step:
            if isinstance(model.config, UnsupervisedPFNConfig):
                assert x is None, (
                    "unsupervised models must receive x=None from dataloader, "
                    f"got x with shape {x.shape if x is not None else None}"
                )
                assert y.ndim == 2, (
                    f"y must be 2D (batch, seq) for unsupervised, got shape {y.shape}"
                )

                if (
                    model.config.input_type == "discrete"
                    and model.config.prediction_type == "distribution"
                ):
                    assert y.dtype in (torch.long, torch.int32, torch.int64), (
                        f"discrete unsupervised models require integer y (token indices), got dtype {y.dtype}"
                    )
                    assert y.min() >= 0 and y.max() < model.config.d_vocab, (
                        f"y values must be in [0, {model.config.d_vocab}), "
                        f"got range [{y.min().item()}, {y.max().item()}]"
                    )
            else:
                # supervised models (classification or regression)
                assert x is not None, (
                    "supervised models require x from dataloader, got x=None. "
                    "use UnsupervisedPFNConfig for models without x"
                )
                assert x.ndim == 3, (
                    f"x must be 3D (batch, seq, input_dim), got shape {x.shape}"
                )
                assert y.ndim == 2, f"y must be 2D (batch, seq), got shape {y.shape}"
                assert x.shape[0] == y.shape[0], (
                    f"x and y batch sizes must match, got {x.shape[0]} and {y.shape[0]}"
                )
                assert x.shape[1] == y.shape[1], (
                    f"x and y sequence lengths must match, got {x.shape[1]} and {y.shape[1]}"
                )
                assert x.shape[2] == model.config.input_dim, (
                    f"x input_dim must match config.input_dim, "
                    f"got x.shape[2]={x.shape[2]} but config.input_dim={model.config.input_dim}"
                )

                if isinstance(model.config, ClassificationPFNConfig):
                    assert y.dtype in (torch.long, torch.int32, torch.int64), (
                        f"classification models require integer y (class indices), got dtype {y.dtype}"
                    )
                    assert y.min() >= 0 and y.max() < model.config.num_classes, (
                        f"y values must be in [0, {model.config.num_classes}), "
                        f"got range [{y.min().item()}, {y.max().item()}]"
                    )

        if x is not None:
            x = x.to(device)
        y = y.to(device)

        # Forward + backward
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled):
            loss, metrics = compute_loss(
                model,
                x,
                y,
                log_distributional_mse=training_config.log_distributional_mse,
            )

        # log loss every step
        loss_value = metrics["loss"]
        assert isinstance(loss_value, (int, float))
        logger.log({"loss": loss_value}, step)

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if training_config.use_grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if training_config.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.grad_clip
                )
            optimizer.step()
        scheduler.step()

        # Get current LR for logging
        lr = optimizer.param_groups[0]["lr"]

        # Metrics window
        run_loss += loss_value
        mse_value = metrics.get("mse")
        if mse_value is not None and isinstance(mse_value, (int, float)):
            run_mse += mse_value
            mse_count += 1
        acc_value = metrics.get("accuracy")
        if acc_value is not None and isinstance(acc_value, (int, float)):
            run_acc += acc_value
            acc_count += 1

        if (step + 1) % training_config.log_every == 0:
            avg_loss = run_loss / training_config.log_every
            loss_type = metrics.get("loss_type", "loss")
            postfix = {
                "loss": f"{loss_type} {avg_loss:.4f}",
                "lr": f"{lr:.6f}",
            }

            # prepare wandb metrics
            wandb_metrics = {
                "train/loss": avg_loss,
                "train/lr": lr,
            }

            if mse_count > 0:
                avg_mse = run_mse / mse_count
                postfix["mse"] = f"{avg_mse:.4f}"
                wandb_metrics["train/mse"] = avg_mse

            if acc_count > 0:
                avg_acc = run_acc / acc_count
                postfix["acc"] = f"{avg_acc:.4f}"
                wandb_metrics["train/accuracy"] = avg_acc

            pbar.set_postfix(postfix)
            logger.log(wandb_metrics, step + 1)

            if training_config.log_file is not None:
                log_entry: dict[str, float | int] = {"step": step + 1}
                log_entry.update(wandb_metrics)
                metrics_log.append(log_entry)

            run_loss = 0.0
            run_mse = 0.0
            run_acc = 0.0
            mse_count = 0
            acc_count = 0

        # determine if we should save checkpoint at this step
        should_save_checkpoint = False
        if training_config.save_checkpoint:
            if training_config.checkpoint_schedule == "logarithmic":
                should_save_checkpoint = (step + 1) in checkpoint_steps_set
            else:
                should_save_checkpoint = (step + 1) % training_config.save_every == 0

        if should_save_checkpoint:
            from datetime import datetime

            ckpt_dir = checkpoint_root
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"checkpoint_step_{step + 1}.pt"

            task_distribution = None
            prior = getattr(data_generator, "prior", None)
            tasks = getattr(prior, "tasks", None)
            if tasks is None:
                base_dist = getattr(prior, "base_distribution", None)
                tasks = getattr(base_dist, "tasks", None)
            if tasks is not None:
                source = (
                    prior
                    if hasattr(prior, "task_size")
                    else getattr(prior, "base_distribution", prior)
                )
                task_distribution = {
                    "tasks": tasks.detach().cpu(),
                    "task_size": getattr(source, "task_size", None),
                    "num_tasks": getattr(source, "num_tasks", None),
                }

            # Create metadata
            metadata = CheckpointMetadata(
                timestamp=datetime.now().isoformat(),
                wandb_run_id=logger.run_id if logger.enabled else None,
                wandb_run_name=logger.run_name if logger.enabled else None,
                wandb_run_url=logger.run_url if logger.enabled else None,
                git_hash=_get_git_hash(),
            )

            # Save checkpoint with metadata
            save_checkpoint(
                checkpoint_path=ckpt_path,
                step=step + 1,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                model_config=model_config,
                training_config=training_config,
                metadata=metadata,
                task_distribution=task_distribution,
            )
            pbar.write(f"Saved checkpoint to {ckpt_path}")
            logger.log_checkpoint(ckpt_path, step + 1, metadata)

        # Evaluation
        if (
            training_config.eval_every is not None
            and (step + 1) % training_config.eval_every == 0
        ):
            eval_gen = (
                eval_data_generator
                if eval_data_generator is not None
                else data_generator
            )
            eval_metrics = evaluate_model(
                model=model,
                data_generator=eval_gen,
                eval_batches=training_config.eval_batches,
                seq_len=training_config.seq_len,
                batch_size=training_config.batch_size,
                device=device,
                log_distributional_mse=training_config.log_distributional_mse,
            )

            # Format and display eval metrics (skip non-numeric values like loss_type)
            eval_str = "Eval: " + ", ".join(
                f"{key}={value:.4f}"
                for key, value in eval_metrics.items()
                if isinstance(value, (int, float))
            )
            pbar.write(eval_str)

            # log to wandb with eval/ prefix (skip non-numeric values like loss_type)
            wandb_eval_metrics = {
                f"eval/{key}": value
                for key, value in eval_metrics.items()
                if isinstance(value, (int, float))
            }
            logger.log(wandb_eval_metrics, step + 1)

            if training_config.log_file is not None:
                # merge eval metrics into the last log entry if same step, else append new
                if metrics_log and metrics_log[-1]["step"] == step + 1:
                    metrics_log[-1].update(wandb_eval_metrics)
                else:
                    log_entry = {"step": step + 1}
                    log_entry.update(wandb_eval_metrics)
                    metrics_log.append(log_entry)

    if training_config.log_file is not None:
        log_path = Path(training_config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics_log, handle, indent=2)
        pbar.write(f"Saved metrics log to {log_path}")

    logger.finish()
    return model


__all__ = [
    "TrainingConfig",
    "compute_loss",
    "evaluate_model",
    "train",
]
