"""Render matrix-view marginals for prior, ID, and OOD Markov PMC samples.

Reads npz bundles from `outputs/markov/sweep_analysis/samples/`:
  - M{n}_prior.npz                       single empty-prompt prior
  - M{n}_in_distribution_L8.npz          16 ID prompts x 128 PMC samples
  - M{n}_out_of_distribution_L8.npz      16 OOD prompts x 128 PMC samples

For each M and each (prior, posterior_id, posterior_ood), writes one density and
one CDF figure. For posteriors we pick a single representative prompt index.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from markov.plotting import plot_pmc_distribution_matrix


def _render_one(
    samples: np.ndarray,
    training: np.ndarray,
    out_dir: Path,
    *,
    label: str,
    title: str,
    prompt_tokens: np.ndarray | None = None,
    panel_size: float = 1.4,
    max_classes: int | None = None,
    dpi: int = 400,
    print_frac: float = 0.95,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("density", "cdf"):
        suffix = "" if mode == "density" else "_cdf"
        plot_pmc_distribution_matrix(
            samples,
            training,
            out_dir / f"{label}_marginals_matrix{suffix}.png",
            title_prefix=title,
            mode=mode,
            panel_size=panel_size,
            prompt_tokens=prompt_tokens,
            max_classes=max_classes,
            dpi=dpi,
            print_frac=print_frac,
        )


def _process_run(
    samples_dir: Path,
    out_dir: Path,
    n_chains: int,
    *,
    prompt_index: int,
    panel_size: float,
    max_classes: int | None,
    dpi: int = 400,
    print_frac: float = 0.95,
) -> None:
    prior_path = samples_dir / f"M{n_chains}_prior.npz"
    id_path = samples_dir / f"M{n_chains}_in_distribution_L8.npz"
    ood_path = samples_dir / f"M{n_chains}_out_of_distribution_L8.npz"

    missing_paths = [
        path for path in (prior_path, id_path, ood_path) if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(f"missing Markov sample bundles: {missing_paths}")

    run_out = out_dir / f"M{n_chains}"

    with np.load(prior_path) as a:
        training = a["training_transition_matrices"]
        prior_samples = a["model_samples"][0]
        _render_one(
            prior_samples,
            training,
            run_out,
            label="prior",
            title=f"Prior (M={n_chains})",
            panel_size=panel_size,
            max_classes=max_classes,
            dpi=dpi,
            print_frac=print_frac,
        )

    with np.load(id_path) as a:
        training = a["training_transition_matrices"]
        id_samples = a["model_samples"][prompt_index]
        prompt = a["prompt_tokens"][prompt_index]
        _render_one(
            id_samples,
            training,
            run_out,
            label="posterior_in_distribution",
            title=f"ID Posterior (M={n_chains}, prompt #{prompt_index})",
            prompt_tokens=prompt,
            panel_size=panel_size,
            max_classes=max_classes,
            dpi=dpi,
            print_frac=print_frac,
        )

    with np.load(ood_path) as a:
        training = a["training_transition_matrices"]
        ood_samples = a["model_samples"][prompt_index]
        prompt = a["prompt_tokens"][prompt_index]
        _render_one(
            ood_samples,
            training,
            run_out,
            label="posterior_out_of_distribution",
            title=f"OOD Posterior (M={n_chains}, prompt #{prompt_index})",
            prompt_tokens=prompt,
            panel_size=panel_size,
            max_classes=max_classes,
            dpi=dpi,
            print_frac=print_frac,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("outputs/markov/sweep_analysis/samples"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir. Default: outputs/markov/matrix_marginals (full K) or "
        "outputs/markov/matrix_marginals_K{max_classes} when --max-classes is set.",
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        nargs="+",
        default=None,
        help="Specific n_chains values; default = all M*_prior.npz found.",
    )
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--panel-size", type=float, default=1.4)
    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Restrict the rendered grid to the first N rows/cols. "
        "Underlying posterior still uses full K.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Raster DPI for the saved PNGs. 200 keeps the full 10x10 grids ~4x "
        "smaller than 400 with no visible loss at print size.",
    )
    parser.add_argument(
        "--print-frac",
        type=float,
        default=0.95,
        help="Fraction of the paper's text width the figure occupies in the "
        "LaTeX source; used to scale fonts (0.45 for the K4 main figures).",
    )
    args = parser.parse_args()

    if args.n_chains is None:
        priors = sorted(args.samples_dir.glob("M*_prior.npz"))
        n_chains_list = sorted(int(p.stem.split("_")[0][1:]) for p in priors)
    else:
        n_chains_list = sorted(args.n_chains)

    assert n_chains_list, f"No M*_prior.npz under {args.samples_dir}"

    out_dir = args.out_dir
    if out_dir is None:
        suffix = f"_K{args.max_classes}" if args.max_classes is not None else ""
        out_dir = Path(f"outputs/markov/matrix_marginals{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in n_chains_list:
        _process_run(
            args.samples_dir,
            out_dir,
            n,
            prompt_index=args.prompt_index,
            panel_size=args.panel_size,
            max_classes=args.max_classes,
            dpi=args.dpi,
            print_frac=args.print_frac,
        )


if __name__ == "__main__":
    main()
