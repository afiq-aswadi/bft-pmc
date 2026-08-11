"""Path-stability diagnostic for Predictive Monte Carlo rollouts.

This reproduces the convergence diagnostic of Ng et al., *TabMGP: Martingale
Posterior with TabPFN* (Figure 3 and Appendix F): the scaled expected L1 norm
between the estimate on the observed data and the estimate after forward
sampling,

    E_{F_N} [ (1/p) || theta(F_n) - theta(F_N) ||_1 ],   N = n + T,

plotted against T. A trajectory that flattens at a non-zero constant is
evidence that theta(F_infinity) exists; one that drifts or spikes is not.

Every point on a curve is the family's own PMC estimator applied to a *prefix*
of a saved rollout, so the diagnostic reads the rollout bundles written next to
each sample bundle and needs no model, checkpoint, or regeneration:

- linear regression: least-squares weights from the (x, y) pairs;
- balls and urns: empirical token frequencies on the simplex;
- Markov: smoothed transition counts, matching ``markov.predictive_monte_carlo``.

Because theta(F_n) is evaluated as the estimator at prefix length n, every
trajectory starts at exactly zero.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

from analysis.rollouts import load_rollout_bundle
from plotting.paper_style import apply_paper_style


MARKOV_SMOOTHING = 1.0

# Which trace arrays identify each family, in the order they are looked up.
_FAMILY_TRACE_KEYS: dict[str, tuple[str, ...]] = {
    "lr": ("rollout_x", "rollout_y"),
    "bau": ("rollout_tokens",),
    "markov": ("rollout_states", "rollout_posterior_states"),
}


@dataclass(slots=True)
class RolloutTraces:
    """One rollout bundle, normalised to (prompts, rollouts, length[, dim])."""

    family: str
    prompt_len: int
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def n_prompts(self) -> int:
        return int(next(iter(self.arrays.values())).shape[0])

    @property
    def n_rollouts(self) -> int:
        return int(next(iter(self.arrays.values())).shape[1])

    @property
    def total_len(self) -> int:
        return int(next(iter(self.arrays.values())).shape[2])


def infer_family(archive: dict[str, np.ndarray]) -> str:
    """Identify the task family from the trace arrays present in a bundle."""
    for family, keys in _FAMILY_TRACE_KEYS.items():
        if any(key in archive for key in keys):
            return family
    raise ValueError(
        f"No known rollout traces in bundle with keys {sorted(archive)}; "
        f"expected one of {sorted(k for keys in _FAMILY_TRACE_KEYS.values() for k in keys)}."
    )


def _as_batched(array: np.ndarray, sample_ndim: int) -> np.ndarray:
    """Prepend a prompt axis to bundles that stored a single prompt."""
    if array.ndim == sample_ndim:
        return array[None]
    return array


def load_traces(path: str | Path) -> RolloutTraces:
    """Load one rollout bundle and normalise its shapes and metadata."""
    archive = load_rollout_bundle(path)
    family = infer_family(archive)

    if family == "lr":
        arrays = {
            "x": _as_batched(archive["rollout_x"], 3),
            "y": _as_batched(archive["rollout_y"], 2),
        }
    elif family == "bau":
        arrays = {"tokens": _as_batched(archive["rollout_tokens"], 2)}
    else:
        key = (
            "rollout_states"
            if "rollout_states" in archive
            else "rollout_posterior_states"
        )
        arrays = {"states": _as_batched(archive[key], 2)}

    metadata = {
        name: archive[name].tolist()
        for name in ("step", "prompt_source", "n_chains", "num_tasks")
        if name in archive
    }
    if "prompt_len" in archive:
        prompt_len = int(archive["prompt_len"])
    elif "prompt_tokens" in archive:
        prompt_len = int(np.asarray(archive["prompt_tokens"]).shape[-1])
    else:
        raise ValueError(
            f"Rollout bundle {path} records neither prompt_len nor prompt_tokens, "
            "so theta(F_n) is undefined."
        )
    return RolloutTraces(
        family=family,
        prompt_len=prompt_len,
        arrays=arrays,
        metadata=metadata,
    )


def prefix_grid(prompt_len: int, total_len: int, stride: int) -> np.ndarray:
    """Prefix lengths at which to evaluate theta, always starting at ``prompt_len``."""
    if stride < 1:
        raise ValueError("stride must be positive.")
    if prompt_len < 1:
        raise ValueError(
            "prompt_len must be positive; theta(F_n) is undefined for prior rollouts."
        )
    if total_len <= prompt_len:
        raise ValueError("total_len must exceed prompt_len.")
    points = np.arange(prompt_len, total_len + 1, stride)
    if points[-1] != total_len:
        points = np.append(points, total_len)
    return points


def _lr_theta_paths(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Least-squares weights on every prefix, accumulating the normal equations."""
    n_prompts, n_rollouts, _, dim = x.shape
    gram = np.zeros((n_prompts, n_rollouts, dim, dim))
    rhs = np.zeros((n_prompts, n_rollouts, dim))
    wanted = set(int(point) for point in grid)
    paths: list[np.ndarray] = []
    for length in range(1, int(grid[-1]) + 1):
        row = x[:, :, length - 1, :]
        gram += row[..., :, None] * row[..., None, :]
        rhs += row * y[:, :, length - 1][..., None]
        if length in wanted:
            # pinv rather than solve: short prefixes leave the Gram matrix singular.
            paths.append((np.linalg.pinv(gram) @ rhs[..., None])[..., 0])
    return np.stack(paths, axis=2)


def _bau_theta_paths(tokens: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Empirical token frequencies on every prefix."""
    n_prompts, n_rollouts, _ = tokens.shape
    vocab_size = int(tokens.max()) + 1
    counts = np.zeros((n_prompts, n_rollouts, vocab_size))
    prompt_index = np.arange(n_prompts)[:, None]
    rollout_index = np.arange(n_rollouts)[None, :]
    wanted = set(int(point) for point in grid)
    paths: list[np.ndarray] = []
    for length in range(1, int(grid[-1]) + 1):
        counts[prompt_index, rollout_index, tokens[:, :, length - 1]] += 1.0
        if length in wanted:
            paths.append(counts / length)
    return np.stack(paths, axis=2)


def _markov_theta_paths(
    states: np.ndarray,
    grid: np.ndarray,
    num_states: int,
) -> np.ndarray:
    """Smoothed transition matrices on every prefix, flattened row-major."""
    n_prompts, n_rollouts, _ = states.shape
    counts = np.full((n_prompts, n_rollouts, num_states, num_states), MARKOV_SMOOTHING)
    prompt_index = np.arange(n_prompts)[:, None]
    rollout_index = np.arange(n_rollouts)[None, :]
    wanted = set(int(point) for point in grid)
    paths: list[np.ndarray] = []
    for length in range(1, int(grid[-1]) + 1):
        if length >= 2:
            counts[
                prompt_index,
                rollout_index,
                states[:, :, length - 2],
                states[:, :, length - 1],
            ] += 1.0
        if length in wanted:
            normalised = counts / counts.sum(axis=-1, keepdims=True)
            paths.append(normalised.reshape(n_prompts, n_rollouts, -1))
    return np.stack(paths, axis=2)


def theta_paths(traces: RolloutTraces, grid: np.ndarray) -> np.ndarray:
    """Estimator evaluated on each prefix: (prompts, rollouts, grid, p)."""
    if traces.family == "lr":
        return _lr_theta_paths(traces.arrays["x"], traces.arrays["y"], grid)
    if traces.family == "bau":
        return _bau_theta_paths(traces.arrays["tokens"], grid)
    states = traces.arrays["states"]
    return _markov_theta_paths(states, grid, num_states=int(states.max()) + 1)


def scaled_l1_paths(paths: np.ndarray) -> np.ndarray:
    """Scaled L1 distance from theta(F_n), i.e. the estimate at the first grid point."""
    return np.abs(paths - paths[:, :, :1, :]).mean(axis=-1)


@dataclass(slots=True)
class SetupCurves:
    """Everything needed to draw and audit one setup's trajectories."""

    setup: str
    family: str
    prompt_len: int
    grid: np.ndarray
    per_prompt: np.ndarray  # (prompts, grid): expectation over rollouts
    expected: np.ndarray  # (grid,): expectation over rollouts and prompts
    theta_reference: np.ndarray  # (prompts, p): theta(F_n)
    theta_terminal: np.ndarray  # (prompts, rollouts, p): theta(F_N)
    metadata: dict[str, object]


def analyse_bundle(
    path: str | Path,
    *,
    setup: str,
    stride: int = 1,
) -> SetupCurves:
    """Compute the path-stability curves for one rollout bundle."""
    traces = load_traces(path)
    grid = prefix_grid(traces.prompt_len, traces.total_len, stride)
    paths = theta_paths(traces, grid)
    distances = scaled_l1_paths(paths)
    return SetupCurves(
        setup=setup,
        family=traces.family,
        prompt_len=traces.prompt_len,
        grid=grid,
        per_prompt=distances.mean(axis=1),
        expected=distances.mean(axis=(0, 1)),
        theta_reference=paths[:, 0, 0, :],
        theta_terminal=paths[:, :, -1, :],
        metadata=traces.metadata,
    )


def curves_to_frames(curves: list[SetupCurves]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert curves to tidy (aggregate, per-prompt) tables."""
    aggregate_rows: list[dict[str, object]] = []
    per_prompt_rows: list[dict[str, object]] = []
    for curve in curves:
        base = {
            "setup": curve.setup,
            "family": curve.family,
            "prompt_len": curve.prompt_len,
            "p": int(curve.theta_reference.shape[-1]),
            "n_prompts": int(curve.per_prompt.shape[0]),
            **curve.metadata,
        }
        for index, prefix in enumerate(curve.grid):
            aggregate_rows.append(
                {
                    **base,
                    "N": int(prefix),
                    "T": int(prefix) - curve.prompt_len,
                    "scaled_l1": float(curve.expected[index]),
                }
            )
            for prompt_index in range(curve.per_prompt.shape[0]):
                per_prompt_rows.append(
                    {
                        **base,
                        "prompt_idx": prompt_index,
                        "N": int(prefix),
                        "T": int(prefix) - curve.prompt_len,
                        "scaled_l1": float(curve.per_prompt[prompt_index, index]),
                    }
                )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(per_prompt_rows)


def plot_path_stability(
    frame: pd.DataFrame,
    save_path: str | Path,
    *,
    group_columns: tuple[str, ...] = ("setup",),
    fig_width_in: float = 3.3,
    print_frac: float = 0.6,
) -> Path:
    """Draw one grey trajectory per group, in the style of TabMGP Figure 3."""
    if frame.empty:
        raise ValueError("No trajectories to plot.")
    apply_paper_style(fig_width_in, print_frac)
    figure, axes = plt.subplots(figsize=(fig_width_in, fig_width_in * 0.62))
    for _, group in frame.groupby(list(group_columns)):
        ordered = group.sort_values("T")
        axes.plot(ordered["T"], ordered["scaled_l1"], color="0.25", linewidth=0.7)
    axes.set_xlabel("N - n")
    axes.set_ylabel("Scaled $L_1$")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.set_xlim(left=0)
    axes.set_ylim(bottom=0)
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return output_path


def discover_bundles(root: str | Path) -> list[Path]:
    """Every rollout bundle under a directory, or the file itself."""
    path = Path(root)
    if path.is_file():
        return [path]
    return sorted(path.rglob("*_rollouts.npz"))


def parse_input(entry: str) -> tuple[str, Path]:
    """Split a ``label=path`` CLI entry; the label defaults to the path name."""
    label, separator, raw_path = entry.partition("=")
    if not separator:
        return Path(entry).name, Path(entry)
    return label, Path(raw_path)


@dataclass(slots=True)
class PathStabilityConfig:
    """CLI arguments for the path-stability diagnostic."""

    inputs: tuple[str, ...] = ()
    output_dir: str = "outputs/path_stability"
    # Evaluate theta every `stride` observations; 1 reproduces the paper exactly.
    stride: int = 1
    # Draw one line per prompt (TabMGP Figure 8) instead of per setup (Figure 3).
    per_prompt_curves: bool = False

    def validate(self) -> None:
        if not self.inputs:
            raise ValueError(
                "Pass at least one --inputs entry: a rollout bundle, a directory "
                "of bundles, or label=path."
            )
        if self.stride < 1:
            raise ValueError("stride must be positive.")


def collect_curves(config: PathStabilityConfig) -> list[SetupCurves]:
    """Analyse every posterior rollout bundle reachable from the inputs."""
    curves: list[SetupCurves] = []
    for entry in config.inputs:
        label, root = parse_input(entry)
        bundles = discover_bundles(root)
        if not bundles:
            raise FileNotFoundError(f"No *_rollouts.npz bundles under {root}.")
        for bundle in bundles:
            traces = load_traces(bundle)
            if traces.prompt_len < 1:
                # Prior rollouts have no observed data, so theta(F_n) is undefined.
                continue
            curves.append(
                analyse_bundle(
                    bundle,
                    setup=f"{label}/{bundle.name.removesuffix('_rollouts.npz')}",
                    stride=config.stride,
                )
            )
    if not curves:
        raise RuntimeError(
            "Every bundle was prior-mode; the diagnostic needs posterior rollouts "
            "with a non-empty prompt."
        )
    return curves


def main(config: PathStabilityConfig) -> None:
    """Compute, save, and plot the path-stability diagnostic."""
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    curves = collect_curves(config)
    aggregate, per_prompt = curves_to_frames(curves)
    aggregate.to_csv(output_dir / "path_stability.csv", index=False)
    per_prompt.to_csv(output_dir / "path_stability_per_prompt.csv", index=False)

    references: dict[str, np.ndarray] = {}
    for curve in curves:
        references[f"{curve.setup}::theta_n"] = curve.theta_reference
        references[f"{curve.setup}::theta_N"] = curve.theta_terminal
        references[f"{curve.setup}::grid"] = curve.grid
    np.savez_compressed(output_dir / "theta_reference.npz", **references)

    frame = per_prompt if config.per_prompt_curves else aggregate
    groups = ("setup", "prompt_idx") if config.per_prompt_curves else ("setup",)
    plot_path_stability(
        frame,
        output_dir / "path_stability.png",
        group_columns=groups,
    )


if __name__ == "__main__":
    main(tyro.cli(PathStabilityConfig))
