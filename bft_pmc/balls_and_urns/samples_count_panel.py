"""Rollout-count sensitivity panel from a saved BAU PMC sample bundle.

Runs the recovery-error-vs-R subsampling analysis of rollout_diagnostics on
the paper's persisted samples/*.npz files, so the panel for an existing run
needs no model and no GPU. The fresh-draw Monte Carlo floor is reconstructed
from the stored Dirichlet concentrations, theta pool, and posterior pool
weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from bft_pmc.analysis.pmc_diagnostics import count_curves
from bft_pmc.analysis.pmc_diagnostics import plot_counts as _plot_counts


@dataclass
class CountPanelConfig:
    """Recovery error against rollout count R from a saved balls-and-urns sample bundle."""

    samples_path: str
    """A samples/*.npz bundle written by eval.py bau-sweep."""
    output_dir: str = "outputs/bau_samples_count_panel"
    """Directory for the CSV and figure."""
    prompt_index: int = 0
    """Prompt index into the bundle (0 for a prior bundle)."""
    count_grid: list[int] = field(
        default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024]
    )
    """Rollout counts R. Counts above the bundle's sample size are dropped,
    and the full sample is always included."""
    n_resamples: int = 50
    """Subsets drawn at each count below the full set."""
    n_projections: int = 100
    """Random projections for sliced Wasserstein."""
    seed: int = 42
    """Seed for the subsets and the reference draws."""


def main(config: CountPanelConfig) -> None:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    bundle = np.load(config.samples_path, allow_pickle=True)

    index = config.prompt_index
    model_samples = bundle["model_samples"][index]
    baselines = {
        "baseline_generalising": bundle["baseline_generalising"][index],
        "baseline_memorising": bundle["baseline_memorising"][index],
    }
    theta_pool = torch.from_numpy(bundle["theta_pool"]).float()
    prompt_source = str(bundle["prompt_source"])

    if prompt_source == "prior":
        alpha = torch.from_numpy(bundle["prior_dirichlet_alpha"]).float()
        pool_weights = torch.full((theta_pool.shape[0],), 1.0 / theta_pool.shape[0])
    else:
        alpha = torch.from_numpy(bundle["posterior_dirichlet_alpha"][index]).float()
        pool_weights = torch.from_numpy(bundle["posterior_pool_weights"][index]).float()

    def fresh_baseline_sampler(r: int) -> dict[str, np.ndarray]:
        generalising = torch.distributions.Dirichlet(alpha).sample((r,)).numpy()
        pool_index = torch.multinomial(pool_weights, r, replacement=True)
        return {
            "baseline_generalising": generalising,
            "baseline_memorising": theta_pool[pool_index].numpy(),
        }

    total = model_samples.shape[0]
    grid = sorted(r for r in config.count_grid if r <= total)
    if grid[-1] != total:
        grid.append(total)

    frame = count_curves(
        model_samples,
        baselines,
        fresh_baseline_sampler,
        grid,
        config.n_resamples,
        config.n_projections,
        rng,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(config.samples_path).stem}_recovery_vs_count"
    frame.to_csv(output_dir / f"{stem}.csv", index=False)
    _plot_counts(
        frame,
        output_dir / stem,
        f"{prompt_source}: recovery error vs rollout count "
        f"(paper samples, R_max={total})",
    )
    print(frame.to_string(index=False))
    print(f"written to {output_dir / stem}.csv")


if __name__ == "__main__":
    main(tyro.cli(CountPanelConfig))
