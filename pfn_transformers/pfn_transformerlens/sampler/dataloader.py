"""Utilities for constructing PFN dataloaders and sampling prompts."""

from __future__ import annotations

import pickle
import warnings
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from torch.utils.data import DataLoader

from pfn_transformerlens.sampler.data_generator import DataGenerator
from pfn_transformerlens.sampler.sampler import Sampler

if TYPE_CHECKING:
    from pfn_transformerlens.train import TrainingConfig


def _collate_batch(
    batch: list[tuple[torch.Tensor | None, torch.Tensor]],
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Stack batch elements while handling None inputs for unsupervised data."""
    xs, ys = zip(*batch)
    if xs[0] is None:
        return None, torch.stack(ys)
    return torch.stack(xs), torch.stack(ys)


def _is_picklable(obj: object) -> bool:
    """Lightweight check for whether an object can be pickled for worker processes."""
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def build_dataloader(
    data_generator: DataGenerator,
    training_config: TrainingConfig,
) -> DataLoader:
    """Construct a DataLoader over an infinite sampler.

    Args:
        data_generator: DataGenerator instance (supervised or unsupervised).
            Use ProbabilisticGenerator to wrap prior/likelihood if needed.
        training_config: Training configuration with batch_size and seq_len.

    Returns:
        DataLoader that yields (x, y) tuples where x may be None for unsupervised models.
    """
    sampler = Sampler(
        seq_len=training_config.seq_len,
        data_generator=data_generator,
        internal_batch_size=training_config.batch_size,
    )

    num_workers = training_config.num_workers
    pin_memory = training_config.pin_memory
    collate_fn = _collate_batch

    # If the generator contains lambdas/locals (common in notebooks), it cannot be pickled.
    if num_workers > 0 and not _is_picklable(sampler):
        warnings.warn(
            "Data generator is not picklable (e.g., lambda defined in a notebook). "
            "Falling back to num_workers=0; define the generator at module scope or "
            "set TrainingConfig.num_workers=0 to silence this.",
            stacklevel=2,
        )
        num_workers = 0

    # pin_memory only makes sense for CPU tensors being moved to GPU.
    if pin_memory:
        gen_device = getattr(sampler.generator, "device", None)
        gen_device = torch.device(gen_device) if gen_device is not None else None
        if gen_device is not None and gen_device.type != "cpu":
            warnings.warn(
                "pin_memory=True but data generator outputs CUDA tensors; "
                "disabling pin_memory to avoid errors.",
                stacklevel=2,
            )
            pin_memory = False

    if num_workers > 0:
        return DataLoader(
            sampler,
            batch_size=training_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=training_config.prefetch_factor,
            persistent_workers=training_config.persistent_workers,
        )
    return DataLoader(
        sampler,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=pin_memory,
    )


def sample_batch(
    data_generator: DataGenerator,
    batch_size: int,
    seq_len: int,
) -> tuple[
    Float[torch.Tensor, "batch seq input_dim"] | None, Float[torch.Tensor, "batch seq"]
]:
    """Sample a batch of sequences from a data generator.

    Args:
        data_generator: DataGenerator instance (supervised or unsupervised).
        batch_size: Number of sequences to generate.
        seq_len: Length of each sequence.

    Returns:
        Tuple of (x, y) where:
            x: Input features of shape (batch_size, seq_len, input_dim), or None for unsupervised
            y: Target values of shape (batch_size, seq_len)
    """
    xs = []
    ys = []

    for _ in range(batch_size):
        result = data_generator.generate(seq_len)
        if isinstance(result, tuple):
            x, y = result
            xs.append(x)
            ys.append(y)
        else:
            # unsupervised generator returns only y
            ys.append(result)

    if xs:
        # supervised: stack x and y
        return torch.stack(xs), torch.stack(ys)
    else:
        # unsupervised: return None, y
        return None, torch.stack(ys)


__all__ = ["build_dataloader", "sample_batch"]
