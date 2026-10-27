"""Recovery error against rollout count, redrawn from a saved balls-and-urns bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bft_pmc.balls_and_urns.samples_count_panel import CountPanelConfig, main

VOCAB, POOL, SAMPLES = 3, 4, 20


def _bundle(path: Path, source: str) -> Path:
    """A sample bundle in the layout eval.py bau-sweep writes."""
    rng = np.random.default_rng(0)
    prompts = 1 if source == "prior" else 2
    theta_pool = rng.dirichlet(np.ones(VOCAB), size=POOL)
    arrays: dict[str, np.ndarray] = {
        "model_samples": rng.dirichlet(np.ones(VOCAB), size=(prompts, SAMPLES)),
        "baseline_generalising": rng.dirichlet(np.ones(VOCAB), size=(prompts, SAMPLES)),
        "baseline_memorising": theta_pool[rng.integers(POOL, size=(prompts, SAMPLES))],
        "theta_pool": theta_pool,
        "prompt_source": np.asarray(source),
    }
    if source == "prior":
        arrays["prior_dirichlet_alpha"] = np.ones(VOCAB)
    else:
        arrays["posterior_dirichlet_alpha"] = np.full((prompts, VOCAB), 2.0)
        arrays["posterior_pool_weights"] = np.full((prompts, POOL), 1.0 / POOL)
    np.savez(path, **arrays)
    return path


@pytest.mark.parametrize(
    ("source", "prompt_index"), [("prior", 0), ("data_generalising", 1)]
)
def test_count_panel_reads_a_bundle_and_adds_the_full_sample(
    source: str, prompt_index: int, tmp_path: Path, saved_figures: list[Path]
) -> None:
    bundle = _bundle(tmp_path / f"run__source_{source}.npz", source)
    output_dir = tmp_path / "out"
    main(
        CountPanelConfig(
            samples_path=str(bundle),
            output_dir=str(output_dir),
            prompt_index=prompt_index,
            count_grid=[4, 8, 64],
            n_resamples=3,
            n_projections=4,
        )
    )
    frame = pd.read_csv(output_dir / f"{bundle.stem}_recovery_vs_count.csv")
    # 64 exceeds the 20 saved samples and is dropped; the full sample is added.
    assert frame["num_rollouts"].tolist() == [4, 8, SAMPLES]
    # The full sample is a single replicate, so it has no spread.
    assert frame["dist/ed_vs_baseline_generalising/sd"].iloc[-1] == 0.0
    assert frame["dist/ed_vs_baseline_generalising/sd"].iloc[0] > 0.0
    assert {path.suffix for path in saved_figures} == {".png", ".pdf"}
