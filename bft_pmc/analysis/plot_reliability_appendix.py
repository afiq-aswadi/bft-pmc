"""Draw the PMC reliability figures of Appendix D and the supplement.

Every figure is redrawn from the CSVs that the rollout-diagnostics runs saved
under ``outputs/pmc_diagnostics/``; no checkpoint is loaded and nothing is
generated. Next to the figures the script writes
``reliability_figure_manifest.json``, which records for every figure its
SHA-256, the SHA-256 of each input CSV and of the run record beside it, and
the plotted values of every panel.

    uv run eval.py plot-reliability                           # -> outputs/reliability_figures/
    uv run eval.py plot-reliability --paper-dir ../icl2latex  # update the paper in place

Figures are written as ``<paper-dir>/figures/<name>.pdf``:

``reliability_{length,count}_{prior,posterior}``
    Recovery error against rollout length N and rollout count R, one row per
    family, at M = 32 (Appendix D.2 and D.3).
``reliability_dimension_{bau,markov}``, ``reliability_taskdiv_{lr,bau,markov}``
    Recovery error against latent dimension and task diversity (D.4).
``reliability_taskdiv_length_{lr,bau,markov}``
    Recovery error against N at each task diversity (D.5).
``reliability_path_stability_{lr,bau,markov}``
    Path-stability (jellyfish) curves to N = 500 (D.1).
``reliability_supp_*``
    The per-family and per-encoding length and count curves of the supplement.

Which diagnostics run feeds which figure is fixed below; the comments explain
the choices (rollout horizon, independent count pools, hardware).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from bft_pmc.paths import PMC_DIAGNOSTICS, repo_relative
from bft_pmc.plotting.paper_style import apply_paper_style

COLORS = {"memorising": "#2ca02c", "generalising": "#1f77b4"}
"""Reference posterior -> line colour, shared with every other paper figure."""

FIGURE_WIDTH_IN = 5.5
CREATOR = "plot_reliability_appendix.py"
MANIFEST_NAME = "reliability_figure_manifest.json"

Panel = dict[str, Any]
Spec = tuple[str, str]
"""(row label, diagnostics run directory name)."""


def read_rows(path: Path) -> list[dict[str, str]]:
    """Every row of a CSV file as a dict of strings."""
    with path.open() as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(path: Path) -> dict[str, str]:
    """Path and SHA-256 of one input, plus the run record beside it if any."""
    record = path.parent / "run_record.json"
    result = {"path": repo_relative(path), "sha256": sha256(path)}
    if record.exists():
        result["run_record"] = repo_relative(record)
        result["run_record_sha256"] = sha256(record)
    return result


class ReliabilityFigures:
    """Draws figures into ``destination/figures`` and collects their manifest.

    ``root`` is the directory holding one subdirectory per diagnostics run,
    normally ``outputs/pmc_diagnostics``.
    """

    def __init__(self, root: Path, destination: Path) -> None:
        self.root = root
        self.destination = destination
        self.figures: list[dict[str, Any]] = []
        self.supplement: list[dict[str, Any]] = []

    # -- layout --------------------------------------------------------------

    @staticmethod
    def make_axes(nrows: int) -> tuple[Figure, np.ndarray]:
        """A ``nrows`` x 2 grid: energy distance left, sliced Wasserstein right."""
        apply_paper_style(FIGURE_WIDTH_IN)
        fig, axes = plt.subplots(
            nrows, 2, figsize=(FIGURE_WIDTH_IN, 0.65 + 1.35 * nrows), squeeze=False
        )
        fig.subplots_adjust(
            left=0.15,
            right=0.985,
            bottom=0.15 if nrows == 2 else 0.10,
            top=0.91,
            hspace=0.48,
            wspace=0.36,
        )
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.14, lw=0.5)
            ax.tick_params(length=2.5, width=0.6)
        return fig, axes

    @staticmethod
    def legend(fig: Figure, kind: str, mark_context: bool = True) -> None:
        """Shared legend: the two references, plus the N = 223 rule if drawn."""
        handles = [
            Line2D(
                [],
                [],
                color=color,
                marker="o",
                ms=3,
                lw=1.15,
                label=name.capitalize().replace("ising", "izing"),
            )
            for name, color in COLORS.items()
        ]
        if kind == "length" and mark_context:
            handles += [Line2D([], [], color=".4", ls="--", lw=0.8, label="$N=223$")]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.55, 1),
            ncol=len(handles),
            frameon=False,
            columnspacing=1.2,
            handlelength=1.7,
        )

    def finish(
        self, fig: Figure, name: str, inputs: Sequence[Path], panels: list[Panel]
    ) -> None:
        """Save ``fig`` as ``figures/<name>`` and record it in the manifest."""
        out = self.destination / "figures" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, metadata={"Creator": CREATOR})
        plt.close(fig)
        self.figures.append(
            {
                "figure": "figures/" + name,
                "sha256": sha256(out),
                "inputs": [provenance(path) for path in dict.fromkeys(inputs)],
                "panels": panels,
            }
        )

    # -- figure kinds --------------------------------------------------------

    def curves(
        self,
        specs: Sequence[Spec],
        source: str,
        kind: str,
        name: str,
        mark_context: bool = True,
    ) -> None:
        """Recovery error against rollout length or count, one row per run.

        ``kind`` is ``"length"`` or ``"count"``. ``mark_context`` draws the
        N = 223 training-context rule on length plots; runs that stop at
        N = 223 pass False, since the rule would sit on their last point.
        """
        fig, axes = self.make_axes(len(specs))
        panels: list[Panel] = []
        inputs: list[Path] = []
        for i, (label, run) in enumerate(specs):
            path = self.root / run / f"{source}_recovery_vs_{kind}.csv"
            data = read_rows(path)
            inputs.append(path)
            xkey = "rollout_length" if kind == "length" else "num_rollouts"
            x = np.array([float(row[xkey]) for row in data])
            for j, metric in enumerate(["ed", "sw"]):
                ax = axes[i, j]
                panel: Panel = {
                    "label": label,
                    "source": source,
                    "metric": metric,
                    "x": x.tolist(),
                    "series": {},
                }
                for ref, color in COLORS.items():
                    key = f"dist/{metric}_vs_baseline_{ref}"
                    if kind == "count":
                        y = np.array([float(row[key + "/mean"]) for row in data])
                        sd = np.array([float(row[key + "/sd"]) for row in data])
                        # The full-sample endpoint is a single draw: no error bar.
                        ax.errorbar(
                            x[:-1],
                            y[:-1],
                            yerr=sd[:-1],
                            fmt="none",
                            ecolor=color,
                            lw=0.65,
                            capsize=1.4,
                        )
                        panel["series"][ref + "_sd"] = sd.tolist()
                    else:
                        y = np.array([float(row[key]) for row in data])
                    ax.plot(x, y, color=color, marker="o", ms=2.5, lw=1.05)
                    panel["series"][ref] = y.tolist()
                ax.set_xscale("log", base=2)
                if kind == "length":
                    ticks = [8, 32, 128, 500] if max(x) > 223 else [8, 32, 128, 223]
                else:
                    ticks = [16, 64, 256, 1024]
                ax.set_xticks(ticks, [str(v) for v in ticks])
                ax.set_xlabel(
                    "Rollout length $N$" if kind == "length" else "Rollout count $R$"
                )
                ax.set_ylabel(
                    (label + "\n" if j == 0 else "")
                    + ("Energy distance" if metric == "ed" else "Sliced Wasserstein-1"),
                    linespacing=2.0,
                )
                if kind == "length":
                    if mark_context:
                        ax.axvline(223, color=".4", ls="--", lw=0.8)
                    values = [v for vals in panel["series"].values() for v in vals]
                    if min(values) > 0 and max(values) / min(values) > 30:
                        ax.set_yscale("log")
                else:
                    # Unbiased energy distance can dip below zero; show where.
                    ax.axhline(0, color=".65", lw=0.5, zorder=0)
                    ax.ticklabel_format(
                        axis="y", style="sci", scilimits=(-3, 3), useMathText=True
                    )
                panels.append(panel)
        self.legend(fig, kind, mark_context)
        self.finish(fig, name, inputs, panels)

    def aggregate(self, run: str, name: str, xkey: str) -> None:
        """Recovery error against latent dimension or task diversity.

        ``xkey`` is ``"latent_dim"`` (reads ``recovery_vs_dimension.csv``) or
        ``"num_tasks"`` (reads ``recovery_vs_task_diversity.csv``), both
        written by ``eval.py recovery-vs-dimension``.
        """
        filename = (
            "recovery_vs_dimension.csv"
            if xkey == "latent_dim"
            else "recovery_vs_task_diversity.csv"
        )
        path = self.root / run / filename
        data = read_rows(path)
        fig, axes = self.make_axes(2)
        panels: list[Panel] = []
        for i, source in enumerate(["prior", "posterior"]):
            subset = sorted(
                (row for row in data if row["source"] == source),
                key=lambda row: float(row[xkey]),
            )
            x = [float(row[xkey]) for row in subset]
            for j, metric in enumerate(["ed", "sw"]):
                ax = axes[i, j]
                panel: Panel = {
                    "source": source,
                    "metric": metric,
                    "x": x,
                    "series": {},
                }
                for ref, color in COLORS.items():
                    y = [float(row[f"{metric}_{ref}"]) for row in subset]
                    ax.plot(x, y, color=color, marker="o", ms=3, lw=1.2)
                    panel["series"][ref] = y
                ax.set_xscale("log", base=2)
                ax.set_xticks(x, [str(int(v)) for v in x])
                ax.set_xlabel(
                    "Free latent parameters"
                    if xkey == "latent_dim"
                    else "Task diversity $M$"
                )
                ax.set_ylabel(
                    (source.capitalize() + "\n" if j == 0 else "")
                    + ("Energy distance" if metric == "ed" else "Sliced Wasserstein-1"),
                    linespacing=2.0,
                )
                ax.ticklabel_format(
                    axis="y", style="sci", scilimits=(-3, 3), useMathText=True
                )
                panels.append(panel)
        self.legend(fig, "aggregate")
        self.finish(fig, name, [path], panels)

    def jellyfish(self) -> None:
        """Path-stability curves, one standalone figure per family.

        No positional encoding for LR and balls-and-urns, matching every other
        figure in the appendix. All three roll out to N = 500. LR and BAU read
        no position, so the longer rollout uses the trained weights unchanged;
        Markov is RoPE and reaches 500 only by rescaling its positions into the
        trained range (rotary position interpolation, scale 256/544). That
        makes the Markov panel a different model setting from the Markov
        results elsewhere in the paper, which the caption states: its recovery
        error is worse than the in-context run's, so this panel shows that the
        paths settle, not that the estimate is good.
        """
        extended = self.root / "jellyfish_m32_n500/path_stability_per_prompt.csv"
        encodings = {"lr": "none", "bau": "none", "markov": "markov"}
        data = read_rows(extended)
        apply_paper_style(FIGURE_WIDTH_IN)
        for family, encoding in encodings.items():
            fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH_IN, 2.1))
            fig.subplots_adjust(left=0.13, right=0.985, bottom=0.21, top=0.96)
            groups: dict[tuple[str, str], list[dict[str, str]]] = {}
            for row in data:
                if row["family"] == family and row["pos_encoding"] == encoding:
                    # One curve per (setup, prompt realisation).
                    groups.setdefault((row["setup"], row["prompt_idx"]), []).append(row)
            for curve in groups.values():
                curve.sort(key=lambda row: float(row["T"]))
                ax.plot(
                    [float(row["T"]) for row in curve],
                    [float(row["scaled_l1"]) for row in curve],
                    color="black",
                    lw=0.55,
                )
            ax.set_xlabel("Generated observations $N$")
            ax.set_ylabel("Scaled $L_1$")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.14, lw=0.5)
            self.finish(fig, f"reliability_path_stability_{family}.pdf", [extended], [])

    # -- manifest ------------------------------------------------------------

    def write_manifest(self) -> Path:
        """Write ``reliability_figure_manifest.json`` beside ``figures/``."""
        manifest = {
            "generator": "bft_pmc/analysis/plot_reliability_appendix.py",
            "generator_sha256": sha256(Path(__file__)),
            "figures": self.figures,
            "supplement": self.supplement,
        }
        path = self.destination / MANIFEST_NAME
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return path


def markov_n500_run(m: int) -> str:
    """The Markov N = 500 run at task diversity ``m``.

    M = 32 is the ``pi544`` run; the other task diversities were submitted
    later under the plain ``_n500`` name. Both use interpolation scale 256/544.
    """
    return "markov_m32_n500_pi544" if m == 32 else f"markov_m{m}_n500"


def draw_all(root: Path, destination: Path) -> ReliabilityFigures:
    """Draw every appendix and supplement figure and write the manifest."""
    figures = ReliabilityFigures(root, destination)

    # D.2 and D.3: all three families at N = 500. Markov gets there by rotary
    # position interpolation (scale 256/544), the same run as the
    # path-stability figure, so every Markov panel in D.1-D.3 is one model
    # setting. Its posterior sits far from the memorizing reference at every
    # length, which the in-context run does not: that is a property of the
    # rescaled model, and the text says so rather than reading it as a PMC
    # result.
    #
    # The `_pools` runs carry --independent-count-pools: the count curve draws
    # a fresh pool at every count and every repeat instead of subsampling one
    # saved pool of 1024, so D.3's R = 1024 endpoint has an error bar rather
    # than a single point. Length and count both come from the same run, so
    # each family shows one prompt across D.2 and D.3; do not mix a length
    # curve from a non-pools run with a count curve from a pools run.
    #
    # Their posterior numbers do NOT match the non-pools runs and are not
    # meant to. In pools mode analyze_source never calls rng.choice, so the
    # numpy stream sits elsewhere when the posterior source draws its prompt,
    # and balls-and-urns picks a different urn while linear regression picks a
    # different task. 1-Markov is unaffected: its prompt comes from the dataset.
    #
    # Linear regression uses the H100 run. The A100 twin (same arguments, same
    # seed) agrees to a relative 3e-4 and is kept as a hardware cross-check;
    # the earlier T4 runs differ by the same amount, which is architecture-
    # specific GEMM summation order compounded along 500 autoregressive steps.
    primary = [
        ("Linear regression", "lr_m32_none_n500_pools_h100"),
        ("Balls-and-urns", "bau_m32_none_n500_pools"),
        ("1-Markov", "markov_m32_n500_pi544_pools"),
    ]
    for source in ["prior", "posterior"]:
        for kind in ["length", "count"]:
            # No N = 223 rule on these: the cutoff is stated in the text, and
            # the line crowded curves that already carry it.
            figures.curves(
                primary, source, kind, f"reliability_{kind}_{source}.pdf", False
            )

    # D.4: recovery error against latent dimension and task diversity.
    for run, name, xkey in [
        ("bau_dimension_sweep_n500", "reliability_dimension_bau.pdf", "latent_dim"),
        (
            "markov_dimension_sweep_n500",
            "reliability_dimension_markov.pdf",
            "latent_dim",
        ),
        ("lr_task_diversity_none_n500", "reliability_taskdiv_lr.pdf", "num_tasks"),
        ("bau_task_diversity_none_n500", "reliability_taskdiv_bau.pdf", "num_tasks"),
        ("markov_task_diversity_n500", "reliability_taskdiv_markov.pdf", "num_tasks"),
    ]:
        figures.aggregate(run, name, xkey)

    # D.5 repeats the rollout-length analysis at each task diversity, on the
    # same runs as the aggregate figures above. LR and BAU read no position,
    # so their N = 500 runs reuse the trained weights.
    for family, ms in [
        ("lr", [32, 128, 256]),
        ("bau", [32, 256, 512]),
        ("markov", [16, 32, 64, 128]),
    ]:
        specs = [
            (
                f"$M={m}$",
                markov_n500_run(m)
                if family == "markov"
                else f"{family}_m{m}_none_n500",
            )
            for m in ms
        ]
        figures.curves(
            specs,
            "posterior",
            "length",
            f"reliability_taskdiv_length_{family}.pdf",
            False,
        )

    # D.1: path stability.
    figures.jellyfish()

    # Supplement: every family at both horizons (Markov in-context only) ...
    for family, ms, horizons in [
        ("lr", [32, 128, 256], [223, 500]),
        ("bau", [32, 256, 512], [223, 500]),
        ("markov", [16, 32, 64, 128], [223]),
    ]:
        for horizon in horizons:
            suffix = "_n500" if horizon == 500 else ""
            specs = [
                (
                    f"$M={m}$",
                    f"markov_m{m}_ctx256"
                    if family == "markov"
                    else f"{family}_m{m}_none{suffix}",
                )
                for m in ms
            ]
            for source in ["prior", "posterior"]:
                for kind in ["length", "count"]:
                    name = f"reliability_supp_{family}_n{horizon}_{source}_{kind}.pdf"
                    figures.curves(specs, source, kind, name)
                    figures.supplement.append(
                        {
                            "file": name,
                            "family": family,
                            "horizon": horizon,
                            "source": source,
                            "kind": kind,
                            "ms": ms,
                        }
                    )

    # ... and the learned and rotary encodings at M = 32.
    for source in ["prior", "posterior"]:
        for kind, horizon in [("length", 500), ("count", 223), ("count", 500)]:
            suffix = "_n500" if horizon == 500 else ""
            specs = [
                (
                    f"{'LR' if family == 'lr' else 'BAU'} / "
                    f"{'learned' if enc == 'learned' else 'RoPE'}",
                    f"{family}_m32_{enc}{suffix}",
                )
                for family in ["lr", "bau"]
                for enc in ["learned", "rope"]
            ]
            name = f"reliability_supp_encodings_n{horizon}_{source}_{kind}.pdf"
            figures.curves(specs, source, kind, name)
            figures.supplement.append(
                {
                    "file": name,
                    "family": "encodings",
                    "horizon": horizon,
                    "source": source,
                    "kind": kind,
                    "ms": [32],
                }
            )

    figures.write_manifest()
    return figures


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("outputs/reliability_figures"),
        help="where to write figures/<name>.pdf and the manifest; pass the paper "
        "checkout (e.g. ../icl2latex) to update its figures in place",
    )
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=PMC_DIAGNOSTICS,
        help="directory holding one subdirectory per diagnostics run",
    )
    args = parser.parse_args(argv)
    figures = draw_all(args.diagnostics_root, args.paper_dir)
    print(
        f"Generated {len(figures.figures)} figures and provenance manifest "
        f"in {args.paper_dir}"
    )


if __name__ == "__main__":
    main()
