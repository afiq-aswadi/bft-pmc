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
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from zipfile import BadZipFile

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
    traces: RolloutTraces | None = None,
) -> SetupCurves:
    """Compute the path-stability curves for one rollout bundle.

    Pass ``traces`` when the caller has already loaded the bundle; the file is
    then opened exactly once, which matters when another job is writing to the
    same sweep directory.
    """
    if traces is None:
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
    """Convert curves to tidy (aggregate, per-prompt) tables.

    Built column-wise rather than row-wise: a full sweep reaches millions of
    per-prompt rows (setups x prefixes x prompts), and materialising that as a
    list of dicts costs tens of gigabytes before pandas even sees it. Label
    columns are categorical so they cost one code per row, not one Python
    string object.
    """
    aggregate_frames: list[pd.DataFrame] = []
    per_prompt_frames: list[pd.DataFrame] = []
    for curve in curves:
        grid = np.asarray(curve.grid, dtype=np.int64)
        prefixes = grid.size
        n_prompts = int(curve.per_prompt.shape[0])
        # "lr-rope/T64_gaussian_L32" -> input label "lr-rope", encoding "rope".
        # Markov carries no encoding, so it labels itself.
        input_label = curve.setup.split("/", 1)[0]
        _, _, encoding = input_label.partition("-")
        labels = {
            "setup": curve.setup,
            "family": curve.family,
            "input_label": input_label,
            "pos_encoding": encoding or input_label,
            **{key: value for key, value in curve.metadata.items()},
        }
        numbers = {
            "prompt_len": curve.prompt_len,
            "p": int(curve.theta_reference.shape[-1]),
            "n_prompts": n_prompts,
        }

        aggregate = pd.DataFrame(
            {
                **{
                    name: pd.Categorical([value] * prefixes)
                    for name, value in labels.items()
                },
                **{name: np.full(prefixes, value) for name, value in numbers.items()},
                "N": grid,
                "T": grid - curve.prompt_len,
                "scaled_l1": np.asarray(curve.expected, dtype=np.float64),
            }
        )
        aggregate_frames.append(aggregate)

        # prefix-major ordering, matching per_prompt[prompt, prefix]
        total = prefixes * n_prompts
        per_prompt_frames.append(
            pd.DataFrame(
                {
                    **{
                        name: pd.Categorical([value] * total)
                        for name, value in labels.items()
                    },
                    **{name: np.full(total, value) for name, value in numbers.items()},
                    "prompt_idx": np.tile(np.arange(n_prompts), prefixes),
                    "N": np.repeat(grid, n_prompts),
                    "T": np.repeat(grid - curve.prompt_len, n_prompts),
                    "scaled_l1": np.asarray(curve.per_prompt, dtype=np.float64)
                    .T.reshape(-1),
                }
            )
        )

    return (
        pd.concat(aggregate_frames, ignore_index=True),
        pd.concat(per_prompt_frames, ignore_index=True),
    )


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


# Positional encodings keep one colour across every figure in the paper.
ENCODING_COLOURS: dict[str, str] = {
    "learned": "#1f77b4",
    "rope": "#d62728",
    "none": "#2ca02c",
    "markov": "0.25",
}
# Markov has no positional-encoding variants, so it stays out of that legend.
LEGEND_ENCODINGS = ("learned", "rope", "none")


def plot_path_stability_by_family(
    frame: pd.DataFrame,
    save_path: str | Path,
    *,
    group_columns: tuple[str, ...] = ("setup",),
    fig_width_in: float = 6.6,
    print_frac: float = 0.6,
    ylims: dict[str, tuple[float, float]] | None = None,
    colour: str | None = None,
) -> Path:
    """One panel per family, trajectories coloured by positional encoding.

    Pass ``colour`` to override that and draw every curve in one shade, which
    is what the per-encoding figures want: each file holds a single encoding,
    so hue carries no information there.

    The single-axes version above reproduces TabMGP Figure 3, which draws one
    dataset per panel. Overlaying three families and three encodings there
    leaves every curve the same grey and nothing attributable, so this is the
    figure to read when comparing setups.
    """
    if frame.empty:
        raise ValueError("No trajectories to plot.")
    families = [str(name) for name in pd.unique(frame["family"])]
    apply_paper_style(fig_width_in, print_frac)
    # Not sharey: BAU peaks near 0.08 and Markov near 0.05, so a shared axis
    # scaled to LR's 1.0 flattens both into the baseline.
    figure, axes_grid = plt.subplots(
        1,
        len(families),
        figsize=(fig_width_in, fig_width_in * 0.34),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    seen: dict[str, str] = {}
    for axes, family in zip(axes_grid[0], families):
        subset = frame[frame["family"] == family]
        for _, group in subset.groupby(list(group_columns), observed=True):
            encoding = str(group["pos_encoding"].iloc[0])
            line_colour = colour or ENCODING_COLOURS.get(encoding, "0.25")
            seen[encoding] = line_colour
            ordered = group.sort_values("T")
            axes.plot(
                ordered["T"],
                ordered["scaled_l1"],
                color=line_colour,
                linewidth=0.7,
                alpha=0.75,
            )
        axes.set_title(family)
        axes.set_xlabel("N - n")
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.set_xlim(left=0)
        if ylims is not None and family in ylims:
            axes.set_ylim(*ylims[family])
        else:
            axes.set_ylim(bottom=0)
    axes_grid[0][0].set_ylabel("Scaled $L_1$")
    handles = [
        plt.Line2D([], [], color=seen[name], linewidth=1.2, label=name)
        for name in LEGEND_ENCODINGS
        if name in seen
    ]
    figure.tight_layout()
    if handles:
        # Anchored to the figure's bottom edge and drawn downwards, so it can
        # never land on the tick labels or the x-axis titles.
        figure.legend(
            handles=handles,
            loc="upper center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return output_path


def family_ylims(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Per-family y-limits taken across every encoding in the frame."""
    limits: dict[str, tuple[float, float]] = {}
    for family, subset in frame.groupby("family", observed=True):
        top = float(subset["scaled_l1"].max()) * 1.05
        limits[str(family)] = (0.0, top if top > 0 else 1.0)
    return limits


def plot_path_stability_per_encoding(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    group_columns: tuple[str, ...] = ("setup",),
    stem: str = "path_stability",
    fig_width_in: float = 6.6,
    print_frac: float = 0.6,
    colour: str = "black",
) -> list[Path]:
    """A standalone figure per positional encoding, one panel per family.

    The y-limit for each family is computed once across *all* encodings and
    reused in every file. Letting each figure autoscale would make `rope` and
    `none` look alike no matter how far apart their drift actually is.
    """
    if frame.empty:
        raise ValueError("No trajectories to plot.")
    output_dir = Path(output_dir)
    limits = family_ylims(frame)
    written: list[Path] = []
    for encoding in [str(name) for name in pd.unique(frame["pos_encoding"])]:
        written.append(
            plot_path_stability_by_family(
                frame[frame["pos_encoding"] == encoding],
                output_dir / f"{stem}_{encoding}.png",
                group_columns=group_columns,
                fig_width_in=fig_width_in,
                print_frac=print_frac,
                ylims=limits,
                colour=colour,
            )
        )
    return written


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
    # How many single-dataset trajectories to draw per setup. TabMGP computes the
    # diagnostic from one realisation of z_{1:n} (Figure 3) and from 20 of them
    # (Figure 8); 0 instead averages over every prompt, which is a different
    # estimand -- the mean over observed datasets rather than one analyst's view.
    realisations: int = 0

    def validate(self) -> None:
        if not self.inputs:
            raise ValueError(
                "Pass at least one --inputs entry: a rollout bundle, a directory "
                "of bundles, or label=path."
            )
        if self.stride < 1:
            raise ValueError("stride must be positive.")
        if self.realisations < 0:
            raise ValueError("realisations must not be negative.")


def collect_curves(config: PathStabilityConfig) -> list[SetupCurves]:
    """Analyse every posterior rollout bundle reachable from the inputs."""
    curves: list[SetupCurves] = []
    skipped: list[str] = []
    for entry in config.inputs:
        label, root = parse_input(entry)
        bundles = discover_bundles(root)
        if not bundles:
            raise FileNotFoundError(f"No *_rollouts.npz bundles under {root}.")
        for bundle in bundles:
            try:
                traces = load_traces(bundle)
            except (FileNotFoundError, BadZipFile, EOFError) as error:
                # Sweeps are routinely still being written, or cleaned up, while
                # this runs: paths are listed up front but opened much later. One
                # unreadable bundle is not worth discarding hours of analysis, so
                # record it loudly and carry on.
                skipped.append(f"{bundle}: {error}")
                print(f"[path-stability] skipping {skipped[-1]}", file=sys.stderr)
                continue
            if traces.prompt_len < 1:
                # Prior rollouts have no observed data, so theta(F_n) is undefined.
                continue
            curves.append(
                analyse_bundle(
                    bundle,
                    setup=f"{label}/{bundle.name.removesuffix('_rollouts.npz')}",
                    stride=config.stride,
                    traces=traces,
                )
            )
    if skipped:
        print(
            f"[path-stability] {len(skipped)} bundle(s) were unreadable and are "
            "missing from the figure.",
            file=sys.stderr,
        )
    if not curves:
        raise RuntimeError(
            "No usable bundles: every one was prior-mode or unreadable. The "
            "diagnostic needs posterior rollouts with a non-empty prompt."
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

    if config.realisations:
        frame = per_prompt[per_prompt["prompt_idx"] < config.realisations]
        groups = ("setup", "prompt_idx")
    else:
        frame = aggregate
        groups = ("setup",)
    plot_path_stability(
        frame,
        output_dir / "path_stability.png",
        group_columns=groups,
    )
    plot_path_stability_by_family(
        frame,
        output_dir / "path_stability_by_family.png",
        group_columns=groups,
    )
    plot_path_stability_per_encoding(frame, output_dir, group_columns=groups)


if __name__ == "__main__":
    main(tyro.cli(PathStabilityConfig))
