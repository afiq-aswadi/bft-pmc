"""Shared checkpoint discovery and metadata helpers for experiment analysis."""

from pathlib import Path

from pfn_transformerlens import checkpointing
from pfn_transformerlens.model.PFN import BasePFN


def get_step(p: Path) -> int:
    return int(p.stem.split("_")[-1])


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = list(run_dir.glob("checkpoint_step_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=get_step)


def find_all_checkpoints(run_dir: Path) -> list[Path]:
    ckpts = list(run_dir.glob("checkpoint_step_*.pt"))
    if not ckpts:
        return []
    return sorted(ckpts, key=get_step)


def load_run_info(checkpoint_path: Path, device: str) -> dict:
    model, _, metadata = checkpointing.load_checkpoint(
        checkpoint_path, device=device, load_optimizer=False
    )
    assert isinstance(model, BasePFN)

    task_dist = getattr(model, "task_distribution", None)
    assert task_dist is not None, (
        f"Checkpoint {checkpoint_path} has no task_distribution"
    )

    return {
        "model": model,
        "tasks": task_dist["tasks"],
        "num_tasks": task_dist["num_tasks"],
        "metadata": metadata,
        "checkpoint_path": checkpoint_path,
    }
