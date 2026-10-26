#!/usr/bin/env python3
"""Build the data layer of the PMC supplementary explorer.

Reads from two sibling repositories that this folder deliberately sits outside
of -- the code repo (`pfn-memorising-generalising`) and the writing repo
(`icl2latex`) -- plus the flat `regen/` figure dump, and writes everything the
static site needs into `figures/` and `data/`:

    figures/<family>/*.png   web-sized copies of the paper's own renders
    data/figures.json        the (family, diversity, source, view) index
    data/series.json         sweep / dynamics / prior-discrimination curves

Nothing here re-runs a model: the marginal grids come from sample bundles that
live only on M3 (see docs/figure_provenance.md in the code repo), so the site
ships the rendered figures for those and computes curves in-browser only where
the numeric data is committed to the repo.

Usage:
    python3 build_site.py           # rebuild everything
    python3 build_site.py --no-figures   # data only, skip the image copy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
CODE = ROOT / "pfn-memorising-generalising"
PAPER = ROOT / "icl2latex" / "figures"
REGEN = ROOT / "regen"

# Sweep snapshots. LR and BAU use the no-positional-encoding runs: the
# FIGURE_PROVENANCE correction of 2026-09-20 supersedes paper_data/{lr,bau}/sweep.
LR_SWEEP = CODE / "outputs/lr/sweep_analysis_none/sweep_20260821_221637/metrics.csv"
BAU_SWEEP = CODE / "outputs/bau/sweep_analysis_none/20260819_194316/metrics.csv"
MARKOV_SWEEP = CODE / "paper_data/markov/sweep/metrics.csv"

MAX_PX = 1600  # longest edge of a web-sized figure
# WebP at q90 is ~3x lighter than optimised PNG on these plots and, checked
# side by side at 2x zoom on the densest Markov matrix, visually identical --
# axis numerals and the 1pt curves stay crisp.
WEBP_QUALITY = 90


# --------------------------------------------------------------------------
# family definitions
# --------------------------------------------------------------------------
# `sources` are ordered prior-first; `key` is what the UI sends back to us.
FAMILIES = {
    "lr": {
        "label": "Linear regression",
        "blurb": "A 0-Markov family: each task is a weight vector in ℝ⁸, "
        "observations are noisy linear responses. PMC recovers a distribution "
        "over the weights themselves.",
        "latent": "Regression coefficient",
        "diversity_label": "M",
        "diversity": [2**k for k in range(17)],  # 1 .. 65536
        "prompt_length": 32,
        "sources": [
            {"key": "prior", "label": "Prior", "kind": "prior"},
            {
                "key": "discrete",
                "label": "Posterior · in-distribution",
                "kind": "posterior",
            },
            {
                "key": "gaussian",
                "label": "Posterior · out-of-distribution",
                "kind": "posterior",
            },
            {"key": "random", "label": "Posterior · random", "kind": "posterior"},
        ],
    },
    "bau": {
        "label": "Balls-and-urns",
        "blurb": "A 0-Markov family over the simplex: each task is an urn "
        "composition, observations are draws from it. The population prior is "
        "Dirichlet, so every marginal is a Beta.",
        "latent": "Color probability",
        "diversity_label": "M",
        "diversity": [2**k for k in range(13)],  # 1 .. 4096
        "prompt_length": 32,
        "sources": [
            {"key": "prior", "label": "Prior", "kind": "prior"},
            {
                "key": "data_memorising",
                "label": "Posterior · in-distribution",
                "kind": "posterior",
            },
            {
                "key": "data_generalising",
                "label": "Posterior · out-of-distribution",
                "kind": "posterior",
            },
        ],
    },
    "markov": {
        "label": "Markov chains",
        "blurb": "A 1-Markov family: each task is a K=10 transition matrix. "
        "PMC returns a distribution over every row of that matrix, shown as a "
        "matrix of marginals.",
        "latent": "Transition probability",
        "diversity_label": "M",
        "diversity": [2**k for k in range(2, 12)],  # 4 .. 2048
        "prompt_length": 32,
        "sources": [
            {"key": "prior", "label": "Prior", "kind": "prior"},
            {
                "key": "posterior_in_distribution",
                "label": "Posterior · in-distribution",
                "kind": "posterior",
            },
            {
                "key": "posterior_out_of_distribution",
                "label": "Posterior · out-of-distribution",
                "kind": "posterior",
            },
        ],
    },
}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def web_name(src: Path, family: str) -> Path:
    return SITE / "figures" / family / (src.stem + ".webp")


def copy_figure(src: Path, dst: Path) -> dict | None:
    """Downscale `src` to web size and write it to `dst`; return its metadata."""
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, MAX_PX / max(w, h))
        if scale < 1.0:
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        im.save(dst, "WEBP", quality=WEBP_QUALITY, method=6)
    return {
        "src": f"figures/{dst.relative_to(SITE / 'figures')}",
        "w": im.size[0],
        "h": im.size[1],
    }


def _ink_bands(path: Path, gap: float = 0.008) -> list[tuple[float, float]]:
    """Horizontal bands of a figure that carry ink, as [start, end] fractions.

    Bands closer together than `gap` are merged, which fuses an axes box with
    the tick labels printed just under it. The gap sits between the ~0.002 that
    separates an axes from its own tick labels and the ~0.015 that separates
    one row of the grid from the next.
    """
    with Image.open(path) as im:
        rows = (np.asarray(im.convert("L")) < 245).mean(axis=1)
    on = rows > 0.002
    segs: list[list[float]] = []
    start = None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > 3:
                segs.append([start / len(on), i / len(on)])
            start = None
    if start is not None:
        segs.append([start / len(on), 1.0])
    merged: list[list[float]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(a, b) for a, b in merged]


def detect_slices(samples: list[Path]) -> dict | None:
    """Locate the legend/titles header, the density row, and the CDF row.

    LR and BAU draw both rows into one figure, so focusing on just the density
    or just the CDF means cropping. All grids from `plot_marginals` share one
    layout (legend, column titles, density axes, CDF axes), so one measurement
    per family serves every M and every prompt source -- but the measurement is
    taken over several figures and reduced to the widest reading of each row,
    so an unusually tall y-label in one cell can never clip another.
    """
    readings = [b for b in (_ink_bands(s) for s in samples) if len(b) == 4]
    if not readings:
        return None
    legend_top = min(r[0][0] for r in readings)
    head_end = min((r[1][1] + r[2][0]) / 2 for r in readings)
    row_split = min((r[2][1] + r[3][0]) / 2 for r in readings)
    cdf_end = max(r[3][1] for r in readings)
    return {
        "header": [max(0.0, legend_top - 0.02), head_end],
        "density": [head_end, max(row_split, head_end + 0.05)],
        "cdf": [row_split, min(1.0, cdf_end + 0.02)],
    }


def marginal_sources(family: str, m: int, source: str) -> dict[str, Path]:
    """Map one (family, M, source) cell to the figure file(s) that show it.

    LR and BAU draw density and CDF as the two rows of a single grid figure;
    Markov writes them as two separate matrices.
    """
    if family == "lr":
        stem = "prior" if source == "prior" else f"{source}_L32"
        return {"grid": REGEN / f"grid_marginal_4dim_T{m}_{stem}.png"}
    if family == "bau":
        return {"grid": REGEN / f"grid_marginal_4dim_M{m}__source_{source}.png"}
    if family == "markov":
        base = f"markov_full_M{m}_{source}_marginals_matrix"
        return {
            "density": PAPER / f"{base}.png",
            "cdf": PAPER / f"{base}_cdf.png",
        }
    return {}


# Cross-M overviews and the prediction-space figures, per family.
CONTEXT_FIGURES = {
    "lr": [
        ("stitched-density", "Density across every M", REGEN / "stitched_density_{src}.png"),
        ("stitched-cdf", "CDF across every M", REGEN / "stitched_cdf_{src}.png"),
    ],
    "bau": [
        ("stitched-density", "Density across every M", REGEN / "stitched_bau_density_{src}.png"),
        ("stitched-cdf", "CDF across every M", REGEN / "stitched_bau_cdf_{src}.png"),
    ],
    "markov": [],
}

OTHER_FIGURES = {
    "lr": [
        ("sweep", "Task-diversity sweep", PAPER / "lr_sweep_combined_2x2.png"),
        ("prior", "Prior comparisons", PAPER / "lr_sweep_prior_delta_ed_sw.png"),
        ("dynamics", "Training dynamics (M=32)", PAPER / "lr_dynamics_combined_logx_2x2.png"),
    ],
    "bau": [
        ("sweep", "Task-diversity sweep", PAPER / "bau_sweep_posterior_2x2.png"),
        ("prior", "Prior comparisons", PAPER / "bau_sweep_prior_kl_ed_sw.png"),
        ("dynamics", "Training dynamics (M=32)", PAPER / "bau_dynamics_combined_logx_2x2.png"),
    ],
    "markov": [
        ("sweep", "Task-diversity sweep", PAPER / "markov_sweep_combined_2x2.png"),
        ("prior", "Prior comparisons", PAPER / "markov_sweep_prior_kl_ed_sw.png"),
        ("dynamics", "Training dynamics (M=8)", PAPER / "markov_dynamics_M8_2x2.png"),
    ],
}

# The LR stitched figures are emitted per prompt source; the prior-only ones
# have no `_L32` suffix.
STITCH_SRC = {
    "lr": {"prior": "prior", "discrete": "discrete_L32", "gaussian": "gaussian_L32",
           "random": "random_L32"},
    "bau": {"prior": "prior", "data_memorising": "data_memorising",
            "data_generalising": "data_generalising"},
}


def build_figures(copy: bool) -> dict:
    index: dict = {"marginals": {}, "context": {}, "other": {}, "slices": {}, "missing": []}

    def place(src: Path, fam: str) -> dict | None:
        """Copy one source figure into the site, or reuse an existing copy.

        The full-resolution originals live only in the authors' workspace, so
        the published figures carry no link back to them.
        """
        out = web_name(src, fam)
        meta = None
        if copy:
            meta = copy_figure(src, out)
        elif out.exists():
            with Image.open(out) as im:
                meta = {"src": f"figures/{fam}/{out.name}", "w": im.size[0], "h": im.size[1]}
        return meta

    for fam, spec in FAMILIES.items():
        for m in spec["diversity"]:
            for source in spec["sources"]:
                entry: dict = {}
                for view, src in marginal_sources(fam, m, source["key"]).items():
                    meta = place(src, fam)
                    if meta is None:
                        index["missing"].append(str(src))
                    else:
                        entry[view] = meta
                if entry:
                    for meta in entry.values():
                        meta["dense"] = True
                    index["marginals"][f"{fam}|{m}|{source['key']}"] = entry

        for slug, label, tmpl in CONTEXT_FIGURES.get(fam, []):
            for source in spec["sources"]:
                src = Path(str(tmpl).replace("{src}", STITCH_SRC[fam][source["key"]]))
                meta = place(src, fam)
                if meta is None:
                    index["missing"].append(str(src))
                    continue
                index["context"].setdefault(f"{fam}|{source['key']}", []).append(
                    {"slug": slug, "label": label, "dense": True, **meta}
                )

        for slug, label, src in OTHER_FIGURES.get(fam, []):
            meta = place(src, fam)
            if meta is None:
                index["missing"].append(str(src))
                continue
            index["other"].setdefault(fam, []).append({"slug": slug, "label": label, **meta})

        # One layout measurement per family, taken from whichever grid we just
        # wrote; Markov ships density and CDF as separate files and needs none.
        grids = [
            e["grid"] for k, e in index["marginals"].items()
            if k.startswith(f"{fam}|") and "grid" in e
        ]
        if grids:
            step = max(1, len(grids) // 6)
            found = detect_slices([SITE / g["src"] for g in grids[::step]])
            if found:
                index["slices"][fam] = found

    return index


# --------------------------------------------------------------------------
# numeric series
# --------------------------------------------------------------------------
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def num(row: dict, key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, "", "N/A"):
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def sweep_series(rows, x_key, source_key, source_map, metrics) -> dict:
    """Collapse a sweep CSV into {source: {metric: [[x, y], ...]}}."""
    out: dict = {}
    for row in rows:
        src = source_map.get((row.get(source_key) or "").strip())
        if src is None:
            continue
        x = num(row, x_key)
        if x is None:
            continue
        bucket = out.setdefault(src, {})
        for label, col in metrics.items():
            y = num(row, col)
            if y is not None:
                bucket.setdefault(label, []).append([x, y])
    for bucket in out.values():
        for pts in bucket.values():
            pts.sort(key=lambda p: p[0])
    return out


def build_series() -> dict:
    series: dict = {"sweep": {}, "dynamics": {}, "prior": {}}

    # ---- task-diversity sweeps: latent-space distance vs M -----------------
    dist = {
        "ED vs memorising": "dist/ed_vs_baseline_memorising",
        "ED vs generalising": "dist/ed_vs_baseline_generalising",
        "SW vs memorising": "dist/sw_vs_baseline_memorising",
        "SW vs generalising": "dist/sw_vs_baseline_generalising",
    }
    series["sweep"]["lr"] = sweep_series(
        read_csv(LR_SWEEP), "num_tasks", "prompt_source",
        {"N/A": "prior", "memorising": "discrete", "generalising": "gaussian",
         "random": "random"},
        dist,
    )
    series["sweep"]["bau"] = sweep_series(
        read_csv(BAU_SWEEP), "num_tasks", "prompt_source",
        {"prior": "prior", "data_memorising": "data_memorising",
         "data_generalising": "data_generalising"},
        dist,
    )
    series["sweep"]["markov"] = sweep_series(
        read_csv(MARKOV_SWEEP), "n_chains", "prompt_source",
        {"N/A": "prior", "in_distribution": "posterior_in_distribution",
         "out_of_distribution": "posterior_out_of_distribution"},
        {
            "ED vs memorising": "dist/ed_vs_baseline_in_distribution",
            "ED vs generalising": "dist/ed_vs_baseline_out_of_distribution",
            "SW vs memorising": "dist/sw_vs_baseline_in_distribution",
            "SW vs generalising": "dist/sw_vs_baseline_out_of_distribution",
        },
    )

    # ---- training dynamics: distance vs optimiser step ---------------------
    def dynamics(path: Path, cols: dict) -> dict:
        rows = read_csv(path)
        out: dict = {}
        for row in rows:
            step = num(row, "step")
            if step is None:
                continue
            for label, col in cols.items():
                y = num(row, col)
                if y is not None:
                    out.setdefault(label, []).append([step, y])
        for pts in out.values():
            pts.sort(key=lambda p: p[0])
        return out

    series["dynamics"]["lr"] = dynamics(
        CODE / "paper_data/lr/dynamics/metrics.csv",
        {
            "ED vs memorising": "ed_vs_baseline_memorising_from_prompts_memorising",
            "ED vs generalising": "ed_vs_baseline_generalising_from_prompts_memorising",
            "SW vs memorising": "sw_vs_baseline_memorising_from_prompts_memorising",
            "SW vs generalising": "sw_vs_baseline_generalising_from_prompts_memorising",
        },
    )
    series["dynamics"]["bau"] = dynamics(
        CODE / "paper_data/bau/dynamics/metrics.csv",
        {
            "ED vs memorising": "ed_vs_baseline_memorising_from_prompts_memorising",
            "ED vs generalising": "ed_vs_baseline_generalising_from_prompts_memorising",
            "SW vs memorising": "sw_vs_baseline_memorising_from_prompts_memorising",
            "SW vs generalising": "sw_vs_baseline_generalising_from_prompts_memorising",
        },
    )
    # The saved monitor is not the paper's reference comparison. The available
    # latent-metric snapshot also conflicts with the paper's stated ED ranking.
    # Display the current paper figure directly until its numeric source is verified.
    series["dynamics"]["markov"] = {}

    # ---- prior discrimination: is the prior memorising or generalising? ----
    def prior_curve(path: Path, x_key: str, cols: dict) -> dict:
        out: dict = {}
        for row in read_csv(path):
            x = num(row, x_key)
            if x is None:
                continue
            for label, col in cols.items():
                y = num(row, col)
                if y is not None:
                    out.setdefault(label, []).append([x, y])
        for pts in out.values():
            pts.sort(key=lambda p: p[0])
        return out

    series["prior"]["lr"] = prior_curve(
        CODE / "paper_data/lr/sweep/prior_delta_mse.csv", "num_tasks",
        {"Δ vs memorising": "delta_vs_memorising",
         "Δ vs generalising": "delta_vs_generalising"},
    )
    series["prior"]["bau"] = prior_curve(
        CODE / "paper_data/bau/sweep/prior_predictive_kl.csv", "num_tasks",
        {"KL vs memorising": "kl_vs_memorising",
         "KL vs generalising": "kl_vs_generalising"},
    )
    series["prior"]["markov"] = prior_curve(
        CODE / "paper_data/markov/sweep/prior_predictive_kl.csv", "n_chains",
        {"KL vs memorising": "kl_vs_memorising",
         "KL vs generalising": "kl_vs_generalising"},
    )
    # Keep CSV column names intact; standardize only reader-facing labels.
    def display_labels(value):
        if isinstance(value, dict):
            return {(key.replace("memorising", "memorizing").replace("generalising", "generalizing") if " vs " in key else key):
                    display_labels(item) for key, item in value.items()}
        return value
    return display_labels(series)



def stamp_assets() -> int:
    """Re-version every local asset URL in index.html by content hash.

    Without this a browser can pair a cached script with a newer page — which
    is how the maths vanished once: fresh HTML full of empty formula slots,
    stale JS with no code to fill them.
    """
    import hashlib

    page = SITE / "index.html"
    if not page.exists():
        return 0
    html = page.read_text(encoding="utf-8")

    def version(rel: str) -> str:
        f = SITE / rel
        return hashlib.sha1(f.read_bytes()).hexdigest()[:8] if f.exists() else "1"

    def repl(m: re.Match) -> str:
        attr, url = m.group(1), m.group(2)
        if url.startswith(("http:", "https:", "data:")):
            return m.group(0)
        return f'{attr}="{url}?v={version(url)}"'

    new = re.sub(r'(src|href)="((?:assets|data|vendor)/[^"?]+)(?:\?v=[^"]*)?"', repl, html)
    if new != html:
        page.write_text(new, encoding="utf-8")
    return len(re.findall(r'(?:src|href)="(?:assets|data|vendor)/[^"]+"', new))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true", help="skip the image copy")
    args = ap.parse_args()

    data = SITE / "data"
    data.mkdir(exist_ok=True)

    index = build_figures(copy=not args.no_figures)

    def emit(name: str, global_name: str, payload: dict, indent: int | None) -> None:
        """Write one payload twice: readable JSON, and a JS global.

        Browsers refuse to fetch() a local file over file://, so the page loads
        the .js copies with <script> and works by double-clicking index.html.
        The .json copies are the citable data export.
        """
        blob = json.dumps(payload, indent=indent, separators=None if indent else (",", ":"))
        (data / f"{name}.json").write_text(blob)
        (data / f"{name}.js").write_text(f"window.{global_name} = {blob};\n")

    emit("figures", "FIGURES", {"families": FAMILIES, **index}, 1)
    emit("series", "SERIES", build_series(), None)

    files = list((SITE / "figures").rglob("*.webp"))
    total = sum(f.stat().st_size for f in files)
    print(f"figures: {len(files)} files, {total / 1e6:.1f} MB")
    print(f"marginal cells indexed: {len(index['marginals'])}")
    print(f"assets stamped: {stamp_assets()}")
    if index["missing"]:
        print(f"MISSING ({len(index['missing'])}):")
        for m in index["missing"][:20]:
            print("  ", m)


if __name__ == "__main__":
    main()
