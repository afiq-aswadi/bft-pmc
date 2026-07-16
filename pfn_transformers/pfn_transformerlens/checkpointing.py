"""Checkpoint save/load utilities. Decoupled from training logic."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import torch
import numpy as np

from pfn_transformerlens.model.PFN import BasePFN, PFNModel
from pfn_transformerlens.model.configs.base import BasePFNConfig


@dataclass
class CheckpointMetadata:
    """Structured metadata for checkpoints."""

    timestamp: str
    wandb_run_id: str | None
    wandb_run_name: str | None
    wandb_run_url: str | None
    git_hash: str | None


def _get_git_hash() -> str | None:
    """Get current git commit hash, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _get_device(device_hint: str) -> str:
    """
    Resolve device hint to actual device string.

    Args:
        device_hint: "auto", "cuda", "mps", or "cpu"

    Returns:
        Device string for torch
    """
    if device_hint == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_hint


def get_logarithmic_checkpoint_steps(
    training_steps: int, n_log_checkpoints: int = 1000, linear_interval: int = 100
) -> list[int]:
    """
    Build checkpoint indices using combined linear and logarithmic spacing.

    Implements the checkpointing scheme from Appendix A.4 of:
    "Dynamics of Transient Structure in In-Context Linear Regression Transformers"
    https://arxiv.org/pdf/2501.17745

    The checkpoint set C = C_linear U C_log where:
    - C_linear = {0, linear_interval, 2*linear_interval, ..., T}
    - C_log = {floor(T^(j/(N-1))) : j = 0, 1, ..., N-1}

    This combines:
    1. Evenly spaced checkpoints to capture regular progress
    2. Logarithmically spaced checkpoints for early-stage rapid changes

    Args:
        training_steps: Total number of training steps (T)
        n_log_checkpoints: Number of logarithmically spaced checkpoints (N)
        linear_interval: Interval for linear checkpoints (e.g., 100 means every 100 steps)

    Returns:
        Sorted list of unique checkpoint step indices in [0, training_steps].

    Example:
        >>> steps = get_logarithmic_checkpoint_steps(150000, n_log_checkpoints=1000, linear_interval=100)
        >>> len(steps)  # approximately 2203 for these settings
        2203
        >>> steps[:5]  # early checkpoints are dense
        [0, 1, 2, 3, 4]
        >>> steps[-5:]  # later checkpoints follow linear spacing
        [149700, 149800, 149900, 149999, 150000]
    """
    if training_steps < 0:
        raise ValueError("training_steps must be non-negative")
    if n_log_checkpoints < 1:
        raise ValueError("n_log_checkpoints must be at least 1")
    if linear_interval <= 0:
        raise ValueError("linear_interval must be positive")

    T = int(training_steps)
    N = int(n_log_checkpoints)
    interval = int(linear_interval)

    if T == 0:
        return [0]

    # C_linear: evenly spaced checkpoints at specified interval, including T.
    linear_steps = np.arange(0, T + 1, interval, dtype=int)
    if linear_steps[-1] != T:
        linear_steps = np.append(linear_steps, T)
    linear_set = set(linear_steps.tolist())

    # C_log: logarithmically spaced checkpoints
    if T == 1:
        log_array = np.ones(N, dtype=int)
    elif N == 1:
        log_array = np.array([T], dtype=int)
    else:
        exponents = np.linspace(0, 1, N)
        log_array = np.floor(np.power(T, exponents)).astype(int)

    # ensure all log steps are in valid range [0, T]
    log_array = np.clip(log_array, 0, T)
    log_set = set(log_array.tolist())

    # union of both sets
    checkpoint_set = linear_set.union(log_set)
    return sorted(checkpoint_set)


def save_checkpoint(
    checkpoint_path: Path,
    step: int,
    model_state: dict,
    optimizer_state: dict,
    model_config: BasePFNConfig,
    training_config,  # TrainingConfig - avoid circular import
    metadata: CheckpointMetadata,
    scheduler_state: dict | None = None,
    task_distribution: dict | None = None,
) -> None:
    """
    Save checkpoint with v2 format.

    Args:
        checkpoint_path: Path to save checkpoint
        step: Current training step
        model_state: model.state_dict()
        optimizer_state: optimizer.state_dict()
        model_config: Model configuration
        training_config: Training configuration
        metadata: Checkpoint metadata with wandb/git info
        scheduler_state: Optional scheduler.state_dict()
        task_distribution: Optional discrete prior info (e.g., {"tasks": tensor, ...})
    """
    checkpoint = {
        "checkpoint_version": 2,
        "step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "model_config": model_config,
        "training_config": training_config,
        "metadata": {
            "timestamp": metadata.timestamp,
            "wandb_run_id": metadata.wandb_run_id,
            "wandb_run_name": metadata.wandb_run_name,
            "wandb_run_url": metadata.wandb_run_url,
            "git_hash": metadata.git_hash,
        },
    }
    if scheduler_state is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state
    if task_distribution is not None:
        checkpoint["task_distribution"] = task_distribution

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path | str,
    device: str = "auto",
    load_optimizer: bool = False,
) -> tuple[BasePFN, dict | None, CheckpointMetadata]:
    """
    Load checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on. "auto" = cuda > mps > cpu
        load_optimizer: If True, return optimizer state dict

    Returns:
        Tuple of (model, optimizer_state_dict | None, metadata)
        - model: Loaded BasePFN in eval mode on specified device
        - optimizer_state: State dict if load_optimizer=True, else None
        - metadata: CheckpointMetadata with wandb/git info

    Example:
        >>> model, opt_state, metadata = load_checkpoint(
        ...     "checkpoints/checkpoint_step_5000.pt",
        ...     device="cuda",
        ...     load_optimizer=True
        ... )
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract metadata
    metadata_dict = ckpt["metadata"]
    metadata = CheckpointMetadata(
        timestamp=metadata_dict.get("timestamp", ""),
        wandb_run_id=metadata_dict.get("wandb_run_id"),
        wandb_run_name=metadata_dict.get("wandb_run_name"),
        wandb_run_url=metadata_dict.get("wandb_run_url"),
        git_hash=metadata_dict.get("git_hash"),
    )

    # Load model
    config = ckpt["model_config"]
    device_str = _get_device(device)
    config.device = device_str
    model = PFNModel(config).to(device_str)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # attach task distribution if present for downstream use
    if "task_distribution" in ckpt:
        setattr(model, "task_distribution", ckpt["task_distribution"])

    optimizer_state = ckpt["optimizer_state_dict"] if load_optimizer else None

    return model, optimizer_state, metadata
