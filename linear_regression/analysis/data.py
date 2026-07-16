"""Checkpoint, generator, and dataset helpers for LR sweep analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from linear_regression.likelihoods import linear_regression
from linear_regression.priors import DiscretePrior
from pfn_transformerlens import DeterministicGenerator, checkpointing
from pfn_transformerlens.checkpointing import CheckpointMetadata
from pfn_transformerlens.model.PFN import SupervisedPFN


class RunInfo(TypedDict):
    """Typed checkpoint metadata used by linear-regression analyses."""

    model: SupervisedPFN
    num_tasks: int
    task_size: int
    tasks: torch.Tensor
    metadata: CheckpointMetadata
    checkpoint_path: Path


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find the latest checkpoint in a run directory."""
    ckpts = list(run_dir.glob("checkpoint_step_*.pt"))
    if not ckpts:
        return None

    def get_step(path: Path) -> int:
        return int(path.stem.split("_")[-1])

    return max(ckpts, key=get_step)


def load_run_info(
    checkpoint_path: Path,
    device: torch.device | str | None = None,
) -> RunInfo:
    """Load checkpoint and extract run info."""
    resolved_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    model, _, metadata = checkpointing.load_checkpoint(
        checkpoint_path,
        device=str(resolved_device),
        load_optimizer=False,
    )
    assert isinstance(model, SupervisedPFN)

    task_dist = getattr(model, "task_distribution", None)
    if task_dist is None:
        raise ValueError(f"Checkpoint {checkpoint_path} has no task_distribution")

    return {
        "model": model,
        "num_tasks": task_dist["num_tasks"],
        "task_size": task_dist["task_size"],
        "tasks": task_dist["tasks"],
        "metadata": metadata,
        "checkpoint_path": checkpoint_path,
    }


def create_generators(
    prior: DiscretePrior,
    task_size: int,
    noise_std: float,
) -> tuple[DeterministicGenerator, DeterministicGenerator]:
    """Create pretraining (discrete) and true (Gaussian) data generators."""
    pretrain_gen = DeterministicGenerator(
        prior=prior,
        function=linear_regression,
        input_dim=task_size,
        noise_std=noise_std,
        device="cpu",
    )

    gaussian_prior = torch.distributions.Independent(
        torch.distributions.Normal(
            loc=torch.zeros(task_size),
            scale=torch.ones(task_size),
        ),
        reinterpreted_batch_ndims=1,
    )
    true_gen = DeterministicGenerator(
        prior=gaussian_prior,
        function=linear_regression,
        input_dim=task_size,
        noise_std=noise_std,
        device="cpu",
    )

    return pretrain_gen, true_gen


def sample_prompt(
    prior: DiscretePrior,
    prompt_source: str,
    prompt_length: int,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a prompt (context) from the specified source."""
    task_size = prior.task_size
    xs = torch.randn(prompt_length, task_size)

    if prompt_source == "gaussian":
        w = torch.randn(task_size)
        ys = xs @ w + noise_std * torch.randn(prompt_length)
    elif prompt_source == "discrete":
        idx = np.random.randint(0, prior.num_tasks)
        w = prior.tasks[idx].cpu()
        ys = xs @ w + noise_std * torch.randn(prompt_length)
    elif prompt_source == "random":
        ys = torch.randn(prompt_length)
    else:
        raise ValueError(f"unsupported prompt source: {prompt_source!r}")

    return xs, ys


@dataclass
class PromptData:
    """Pre-generated prompt data for consistent evaluation across checkpoints."""

    xs: np.ndarray  # (n_prompts, max_prompt_length, task_size)
    ys_gaussian: np.ndarray  # (n_prompts, max_prompt_length)
    ys_discrete: np.ndarray  # (n_prompts, max_prompt_length)
    ys_random: np.ndarray  # (n_prompts, max_prompt_length)


def generate_prompt_data(
    prior: DiscretePrior,
    max_prompt_length: int,
    n_prompts: int,
    noise_std: float,
) -> PromptData:
    """Generate prompt data from a prior for distribution metrics."""
    task_size = prior.task_size
    tasks = prior.tasks.cpu().numpy()
    num_tasks = prior.num_tasks

    xs = np.random.randn(n_prompts, max_prompt_length, task_size)
    noise = noise_std * np.random.randn(n_prompts, max_prompt_length)

    ws_gaussian = np.random.randn(n_prompts, task_size)
    ys_gaussian = np.einsum("npd,nd->np", xs, ws_gaussian) + noise

    task_indices = np.arange(n_prompts) % num_tasks
    ws_discrete = tasks[task_indices]
    ys_discrete = np.einsum("npd,nd->np", xs, ws_discrete) + noise

    ys_random = np.random.randn(n_prompts, max_prompt_length)

    return PromptData(
        xs=xs.astype(np.float32),
        ys_gaussian=ys_gaussian.astype(np.float32),
        ys_discrete=ys_discrete.astype(np.float32),
        ys_random=ys_random.astype(np.float32),
    )


def get_prompts_for_config(
    prompt_data: PromptData,
    prompt_source: str,
    prompt_length: int,
    n_prompts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract batched prompt tensors for a sweep configuration."""
    all_xs = torch.from_numpy(prompt_data.xs[:n_prompts, :prompt_length])
    if prompt_source == "gaussian":
        all_ys = torch.from_numpy(prompt_data.ys_gaussian[:n_prompts, :prompt_length])
    elif prompt_source == "discrete":
        all_ys = torch.from_numpy(prompt_data.ys_discrete[:n_prompts, :prompt_length])
    elif prompt_source == "random":
        all_ys = torch.from_numpy(prompt_data.ys_random[:n_prompts, :prompt_length])
    else:
        raise ValueError(f"unsupported prompt source: {prompt_source!r}")
    return all_xs, all_ys


def save_shared_dataset(
    path: Path,
    prompt_xs: np.ndarray,
    prompt_ys_generalising: np.ndarray,
    prompt_ys_random: np.ndarray,
    generalising_xs: np.ndarray,
    generalising_ys: np.ndarray,
    random_xs: np.ndarray,
    random_ys: np.ndarray,
    task_size: int,
    noise_std: float,
    eval_batch_size: int,
    seq_len: int,
    max_prompt_length: int,
    max_n_prompts: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        prompt_xs=prompt_xs,
        prompt_ys_generalising=prompt_ys_generalising,
        prompt_ys_random=prompt_ys_random,
        generalising_xs=generalising_xs,
        generalising_ys=generalising_ys,
        random_xs=random_xs,
        random_ys=random_ys,
        task_size=np.int64(task_size),
        noise_std=np.float64(noise_std),
        eval_batch_size=np.int64(eval_batch_size),
        seq_len=np.int64(seq_len),
        max_prompt_length=np.int64(max_prompt_length),
        max_n_prompts=np.int64(max_n_prompts),
    )


def load_shared_dataset(path: Path) -> dict[str, np.ndarray]:
    """Load prior-independent evaluation data."""
    return dict(np.load(path))


def save_prior_dataset(
    path: Path,
    prompt_ys_memorising: np.ndarray,
    memorising_xs: np.ndarray,
    memorising_ys: np.ndarray,
    num_tasks: int,
    run_id: str,
) -> None:
    """Save run-specific memorising evaluation data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        prompt_ys_memorising=prompt_ys_memorising,
        memorising_xs=memorising_xs,
        memorising_ys=memorising_ys,
        num_tasks=np.int64(num_tasks),
        run_id=np.array(run_id),
    )


def load_prior_dataset(path: Path) -> dict[str, np.ndarray]:
    """Load run-specific memorising evaluation data."""
    return dict(np.load(path))
