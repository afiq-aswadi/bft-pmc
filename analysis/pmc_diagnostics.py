"""Shared machinery for PMC rollout-length and rollout-count diagnostics.

Family modules (balls_and_urns/rollout_diagnostics.py,
markov/rollout_diagnostics.py) produce running rollout-statistic arrays of
shape (num_rollouts, len(grid), dim); everything here operates on those flat
sample matrices: recovery-error curves against baseline sample sets, the
path-stability drift diagnostic, the rollout-count subsampling panel with a
fresh-draw Monte Carlo floor, and the corresponding figures.

Each run directory also records how it was produced and how to redraw it:
``run_record.json`` holds the configuration and provenance, and
``figures.json`` maps every figure to its kind, title and data files, which is
all ``analysis/replot_pmc_diagnostics.py`` needs to draw it again. The
posterior rollouts are also saved as ``<run>_rollouts.npz``, the input of the
jellyfish plot (``eval.py path-stability``).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from metrics import energy_distance_multidim, sliced_wasserstein


RUN_RECORD = "run_record.json"
FIGURE_INDEX = "figures.json"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def check_rollouts_fit_context(
    *,
    steps: int,
    prompt_length: int,
    prompt_sources: list[str],
    context_length: int,
    unit: str = "positions",
) -> None:
    """Fail before any rollout if the longest one would overrun the context.

    A rollout of ``steps`` tokens after a ``prompt_length``-token prompt runs the
    model on ``prompt_length + steps`` positions: balls and urns prepend BOS but
    never feed back the last generated token. Linear regression counts in
    (x, y) pairs against half its token context. The prior runs first, so
    without this check a posterior that does not fit only fails after the prior
    rollouts have already been paid for.
    """
    longest_prompt = prompt_length if "posterior" in prompt_sources else 0
    if longest_prompt + steps > context_length:
        raise ValueError(
            f"steps={steps} after a prompt of {longest_prompt} needs "
            f"{longest_prompt + steps} {unit}, but the model context holds "
            f"{context_length}. Pass --steps {context_length - longest_prompt} or "
            "less, with a --length-grid that ends there."
        )


def save_path_stability_bundle(
    output_dir: Path,
    *,
    prompt_length: int,
    checkpoint_path: str,
    arrays: dict[str, np.ndarray],
    **metadata: object,
) -> Path:
    """Write posterior rollouts where ``eval.py path-stability`` reads them.

    ``arrays`` uses the keys of ``analysis/path_stability.py``: ``rollout_tokens``,
    ``rollout_states``, or ``rollout_x`` with ``rollout_y``, each covering the
    prompt and every generated step. The file is named after the run directory,
    so runs given the same label stay separate trajectories in the jellyfish
    plot, and the checkpoint step is recorded when the file name carries it.
    """
    extras = {name: np.asarray(value) for name, value in metadata.items()}
    step = re.search(r"checkpoint_step_(\d+)", Path(checkpoint_path).name)
    if step is not None:
        extras["step"] = np.asarray(int(step.group(1)))
    path = output_dir / f"{output_dir.name}_rollouts.npz"
    np.savez_compressed(
        path, prompt_len=np.asarray(prompt_length), **arrays, **extras
    )
    return path


def distance_metrics(
    model_samples: np.ndarray,
    baseline_samples: dict[str, np.ndarray],
    n_projections: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, references in baseline_samples.items():
        metrics[f"dist/ed_vs_{name}"] = energy_distance_multidim(
            model_samples,
            references,
        )
        metrics[f"dist/sw_vs_{name}"] = sliced_wasserstein(
            model_samples,
            references,
            n_projections=n_projections,
        )
    return metrics


def recovery_curves(
    stats: np.ndarray,
    baselines: dict[str, np.ndarray],
    grid: list[int],
    n_projections: int,
) -> pd.DataFrame:
    rows = []
    for j, n in enumerate(grid):
        metrics = distance_metrics(stats[:, j, :], baselines, n_projections)
        rows.append({"rollout_length": n, **metrics})
    return pd.DataFrame(rows)


def drift_curves(
    stats: np.ndarray,
    grid: list[int],
    anchor_stat: np.ndarray,
    scale: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Scaled L1 drift from the anchor estimate at each grid length.

    anchor_stat: (num_rollouts, dim) for a per-path anchor, or (dim,) for a
    shared anchor. scale defaults to the statistic dimension; pass the free
    parameter count when that differs (e.g. transition matrices).
    Returns the summary frame and the per-path drift matrix (R, len(grid)).
    """
    if scale is None:
        scale = stats.shape[-1]
    if anchor_stat.ndim == 1:
        anchor = anchor_stat[None, None, :]
    else:
        anchor = anchor_stat[:, None, :]
    per_path = np.abs(stats - anchor).sum(axis=-1) / scale

    frame = pd.DataFrame(
        {
            "rollout_length": grid,
            "drift_mean": per_path.mean(axis=0),
            "drift_q10": np.quantile(per_path, 0.10, axis=0),
            "drift_q50": np.quantile(per_path, 0.50, axis=0),
            "drift_q90": np.quantile(per_path, 0.90, axis=0),
        }
    )
    return frame, per_path


RESAMPLE_COLUMNS = ["num_rollouts", "resample", "kind", "metric", "value"]


def count_curves(
    stat_final: np.ndarray,
    baselines: dict[str, np.ndarray],
    fresh_baseline_sampler: Callable[[int], dict[str, np.ndarray]],
    count_grid: list[int],
    n_resamples: int,
    n_projections: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Recovery error vs rollout count R, with a fresh-draw Monte Carlo floor."""
    frame, _ = count_curves_with_resamples(
        stat_final,
        baselines,
        fresh_baseline_sampler,
        count_grid,
        n_resamples,
        n_projections,
        rng,
    )
    return frame


def on_isolated_torch_stream(sampler, seed):
    """Wrap a rollout sampler so it draws from its own torch RNG stream.

    The drivers seed once and then draw each source's prompt from the global
    stream. A sampler that consumed that stream would shift every prompt drawn
    after it, so the prior source's draws would change the posterior source's
    prompt and the run would stop reproducing the others. This keeps a private
    stream for the sampler and restores the global one around every call, on
    CPU and CUDA alike.
    """
    import torch

    saved: dict = {}

    def wrapped(*args, **kwargs):
        cuda = torch.cuda.is_available()
        outer_cpu = torch.get_rng_state()
        outer_cuda = torch.cuda.get_rng_state_all() if cuda else None
        if saved:
            torch.set_rng_state(saved["cpu"])
            if cuda and saved.get("cuda") is not None:
                torch.cuda.set_rng_state_all(saved["cuda"])
        else:
            torch.manual_seed(seed)
        try:
            return sampler(*args, **kwargs)
        finally:
            saved["cpu"] = torch.get_rng_state()
            saved["cuda"] = torch.cuda.get_rng_state_all() if cuda else None
            torch.set_rng_state(outer_cpu)
            if cuda:
                torch.cuda.set_rng_state_all(outer_cuda)

    return wrapped


def count_curves_with_resamples(
    stat_final: np.ndarray,
    baselines: dict[str, np.ndarray],
    fresh_baseline_sampler: Callable[[int], dict[str, np.ndarray]],
    count_grid: list[int],
    n_resamples: int,
    n_projections: int,
    rng: np.random.Generator,
    fresh_model_sampler: Callable[[int], np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """count_curves plus every draw behind its means and standard deviations.

    The second frame has one row per (R, resample, kind, metric), where kind is
    ``model`` for the rollouts and ``floor`` for the fresh baseline draw, so
    the panel can be redrawn with another spread measure without regenerating
    anything.

    With ``fresh_model_sampler`` every draw is its own pool of r rollouts, so
    no rollout is reused across repeats or across counts and the spread is
    Monte Carlo error. Without it the draws are subsets of one saved pool,
    which overlap more as R grows and therefore understate that error.
    """
    rows = []
    resample_rows: list[dict[str, object]] = []
    total = stat_final.shape[0]
    for r in count_grid:
        # With an independent pool per draw every R gets a spread, including
        # the largest; subsetting one saved pool cannot estimate one at
        # r == total, because every draw of that size is the same set.
        resamples = n_resamples if fresh_model_sampler is not None else (
            1 if r == total else n_resamples
        )
        model_metrics: dict[str, list[float]] = {}
        floor_metrics: dict[str, list[float]] = {}
        for resample in range(resamples):
            if fresh_model_sampler is not None:
                # A fresh pool of exactly r rollouts, symmetric with the
                # floor's fresh_baseline_sampler. No rollout is shared between
                # repeats, nor between the counts along the curve.
                sample = fresh_model_sampler(r)
            else:
                sample = stat_final[rng.choice(total, size=r, replace=False)]
            for key, value in distance_metrics(
                sample, baselines, n_projections
            ).items():
                model_metrics.setdefault(key, []).append(value)
                resample_rows.append(
                    {
                        "num_rollouts": r,
                        "resample": resample,
                        "kind": "model",
                        "metric": key,
                        "value": value,
                    }
                )
            fresh = fresh_baseline_sampler(r)
            for name, reference in baselines.items():
                for key, value in distance_metrics(
                    fresh[name], {name: reference}, n_projections
                ).items():
                    floor_metrics.setdefault(key, []).append(value)
                    resample_rows.append(
                        {
                            "num_rollouts": r,
                            "resample": resample,
                            "kind": "floor",
                            "metric": key,
                            "value": value,
                        }
                    )
        row: dict[str, float] = {"num_rollouts": r}
        for key, values in model_metrics.items():
            row[f"{key}/mean"] = float(np.mean(values))
            row[f"{key}/sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for key, values in floor_metrics.items():
            row[f"floor/{key}/mean"] = float(np.mean(values))
            row[f"floor/{key}/sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(resample_rows, columns=RESAMPLE_COLUMNS)


def plot_recovery(frame: pd.DataFrame, out_stem: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for ax, metric, label in zip(
        axes, ["ed", "sw"], ["Energy distance", "Sliced Wasserstein-1"]
    ):
        for baseline, style in [
            ("baseline_generalising", "-o"),
            ("baseline_memorising", "--s"),
        ]:
            ax.plot(
                frame["rollout_length"],
                frame[f"dist/{metric}_vs_{baseline}"],
                style,
                markersize=3,
                label=baseline.replace("baseline_", ""),
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("rollout length $N$")
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_drift(
    frame: pd.DataFrame,
    per_path: np.ndarray,
    grid: list[int],
    out_stem: Path,
    title: str,
    max_traces: int = 200,
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    shown = min(max_traces, per_path.shape[0])
    ax.plot(grid, per_path[:shown].T, color="tab:blue", alpha=0.05, linewidth=0.6)
    ax.plot(
        frame["rollout_length"],
        frame["drift_mean"],
        color="black",
        linewidth=1.8,
        label="mean",
    )
    ax.fill_between(
        frame["rollout_length"],
        frame["drift_q10"],
        frame["drift_q90"],
        color="black",
        alpha=0.12,
        label="10–90%",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("rollout length $N$")
    ax.set_ylabel("scaled L1 drift from anchor")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_counts(frame: pd.DataFrame, out_stem: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for ax, metric, label in zip(
        axes, ["ed", "sw"], ["Energy distance", "Sliced Wasserstein-1"]
    ):
        for baseline, color in [
            ("baseline_generalising", "tab:blue"),
            ("baseline_memorising", "tab:orange"),
        ]:
            key = f"dist/{metric}_vs_{baseline}"
            ax.errorbar(
                frame["num_rollouts"],
                frame[f"{key}/mean"],
                yerr=frame[f"{key}/sd"],
                fmt="-o",
                markersize=3,
                color=color,
                label=baseline.replace("baseline_", ""),
            )
            ax.errorbar(
                frame["num_rollouts"],
                frame[f"floor/{key}/mean"],
                yerr=frame[f"floor/{key}/sd"],
                fmt=":",
                color=color,
                alpha=0.7,
                label=f"{baseline.replace('baseline_', '')} floor",
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("rollout count $R$")
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def _git_state(root: Path) -> dict[str, object]:
    """Commit and uncommitted files of the checkout; nulls outside git.

    M3 receives edits by rsync on top of a checkout, so the changed-file list
    is what distinguishes the code that ran from the recorded commit.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "changed_files": None}
    return {"commit": commit, "changed_files": changed}


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _installed_distributions() -> dict[str, str]:
    """Every installed distribution and its version, like ``pip freeze``."""
    return dict(
        sorted(
            (str(distribution.metadata["Name"]), distribution.version)
            for distribution in importlib.metadata.distributions()
        )
    )


def _hardware() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def file_sha256(path: str | Path) -> str | None:
    """SHA-256 of a file's bytes, or None when it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Environment variables that identify where and how a job ran.
RECORDED_ENVIRONMENT = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "PMC_SUBMISSION_DIR",
)


def write_run_record(
    output_dir: Path,
    config: dict[str, object],
    *,
    inputs: list[str | None] | None = None,
    **context: object,
) -> Path:
    """Save a run's configuration and provenance as ``run_record.json``.

    ``config`` is the run's CLI configuration as a dict; ``inputs`` names the
    files the run read (checkpoints, configs), each recorded with its SHA-256;
    ``context`` holds what the run established at load time, such as the model
    context and the size of the task pool. With the code snapshot that
    ``slurm/submit_pmc_diagnostics.sh`` saves per submission, this is what a
    rerun needs: the exact settings, inputs, code and software.
    """
    record = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "environment": {name: os.environ.get(name) for name in RECORDED_ENVIRONMENT},
        "python": sys.version.split()[0],
        "packages": _package_versions(("numpy", "pandas", "torch", "matplotlib")),
        "installed_distributions": _installed_distributions(),
        "hardware": _hardware(),
        "git": _git_state(REPOSITORY_ROOT),
        "inputs": {
            str(path): file_sha256(path) for path in (inputs or []) if path is not None
        },
        "config": config,
        "context": context,
    }
    path = output_dir / RUN_RECORD
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return path


def record_figure(
    output_dir: Path, stem: str, *, kind: str, title: str, **data: object
) -> None:
    """Register a figure in ``figures.json`` with what it was drawn from.

    ``data`` names the files, relative to ``output_dir``, that the figure's
    plotting function reads back: ``csv`` for every kind, plus ``paths`` for
    drift and ``sources`` for the dimension aggregate.
    """
    index_path = output_dir / FIGURE_INDEX
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[stem] = {"kind": kind, "title": title, **data}
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
