"""Persistence helpers for Predictive Monte Carlo rollout traces.

Every PMC estimator reduces a batch of generated sequences to a task estimate
(regression weights, a simplex point, a transition matrix). The raw generated
sequences behind those estimates are the rollouts. They are written next to the
sample bundle they explain, as ``<bundle stem>_rollouts.npz``, so that existing
bundle readers keep their schema and the (much larger) traces can be pruned
independently.

Trace arrays are keyed with the ``rollout_`` prefix inside sample dictionaries
so that :func:`split_rollout_arrays` can route them to the sibling file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROLLOUT_PREFIX = "rollout_"


def rollout_bundle_path(samples_path: str | Path) -> Path:
    """Return the rollout-bundle path that belongs to a sample bundle."""
    path = Path(samples_path)
    return path.with_name(f"{path.stem}_rollouts.npz")


def compact_token_array(tokens: np.ndarray) -> np.ndarray:
    """Cast discrete rollout states to the smallest lossless integer dtype.

    Token streams dominate the on-disk size of the discrete settings, and the
    vocabularies here are tiny, so storing them as int64 wastes a factor of four.
    """
    array = np.asarray(tokens)
    if array.size == 0:
        return array.astype(np.int16)
    minimum = int(array.min())
    maximum = int(array.max())
    for dtype in (np.int16, np.int32):
        info = np.iinfo(dtype)
        if minimum >= info.min and maximum <= info.max:
            return array.astype(dtype)
    return array.astype(np.int64)


def split_rollout_arrays(
    samples: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split a sample dictionary into (estimates, ``rollout_*`` traces)."""
    estimates = {
        key: value
        for key, value in samples.items()
        if not key.startswith(ROLLOUT_PREFIX)
    }
    traces = {
        key: value for key, value in samples.items() if key.startswith(ROLLOUT_PREFIX)
    }
    return estimates, traces


def save_rollout_bundle(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    **metadata: np.ndarray | int | str,
) -> Path | None:
    """Write rollout traces to a compressed archive.

    Passing an empty ``arrays`` mapping is how callers disable rollout saving;
    nothing is written and ``None`` is returned.
    """
    if not arrays:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays, **metadata)
    return output_path


def save_samples_and_rollouts(
    path: str | Path,
    samples: dict[str, np.ndarray],
    *,
    save_rollouts: bool = True,
    **metadata: np.ndarray | int | str,
) -> Path | None:
    """Write a sample bundle and route its ``rollout_*`` traces to the sibling file.

    ``metadata`` is stored alongside the traces so a rollout bundle describes
    itself (prompt length, source, setup) without its sample bundle.

    Returns the rollout-bundle path, or ``None`` when rollout saving is off or
    the bundle carries no traces.
    """
    estimates, traces = split_rollout_arrays(samples)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **estimates)
    return save_rollout_bundle(
        rollout_bundle_path(output_path),
        traces if save_rollouts else {},
        **metadata,
    )


def load_rollout_bundle(path: str | Path) -> dict[str, np.ndarray]:
    """Load every array from a rollout bundle."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
