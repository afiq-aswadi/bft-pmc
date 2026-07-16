from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from linear_regression.analysis.config import SweepConfig
from linear_regression.analysis.plotting import (
    plot_marginal_distributions,
    plot_per_prompt_marginals,
    plot_results,
)


def _analysis_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for num_tasks in [2, 4]:
        for source, prompt_length in [("N/A", 0), ("discrete", 2), ("gaussian", 2)]:
            row: dict[str, float | int | str] = {
                "run_id": f"run-{num_tasks}",
                "num_tasks": num_tasks,
                "checkpoint_step": 10,
                "prompt_source": source,
                "prompt_length": prompt_length,
                "n_samples": 2 if prompt_length else 0,
                "n_samples_prior": 2 if prompt_length == 0 else 0,
                "n_prompts": 1 if prompt_length else 0,
                "dist/ed_vs_baseline_memorising": 0.1,
                "dist/ed_vs_baseline_generalising": 0.2,
                "dist/sw_vs_baseline_memorising": 0.3,
                "dist/sw_vs_baseline_generalising": 0.4,
            }
            for distribution in ["memorising", "generalising", "random"]:
                for metric in [
                    "model_mse",
                    "baseline_memorising_mse",
                    "baseline_generalising_mse",
                    "delta_vs_baseline_memorising",
                    "delta_vs_baseline_generalising",
                ]:
                    row[f"data_{distribution}/{metric}"] = 0.1
            rows.append(row)
    return pd.DataFrame(rows)


def _write_analysis_samples(
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True)
    rng = np.random.default_rng(8)
    theta_pool = rng.normal(size=(2, 2))
    prior = {
        "pt": rng.normal(size=(24, 2)),
        "theta_pool": theta_pool,
        "dmmse_weights": np.full(2, 0.5),
    }
    posterior = {
        "pt": rng.normal(size=(1, 24, 2)),
        "theta_pool": theta_pool,
        "dmmse_weights": np.full((1, 2), 0.5),
        "baseline_generalising_posterior_means": np.zeros((1, 2)),
        "baseline_generalising_posterior_covs": np.eye(2)[None],
    }
    np.savez(samples_dir / "T2_prior.npz", **prior)
    np.savez(samples_dir / "T2_discrete_L2.npz", **posterior)
    np.savez(samples_dir / "ignored.npz", unrelated=np.ones(1))
    return prior, posterior


def test_lr_analysis_plotting_complete_pipeline(
    tmp_path: Path,
    saved_figures: list[Path],
) -> None:
    frame = _analysis_frame()
    prior, posterior = _write_analysis_samples(tmp_path)
    config = SweepConfig(eval_position=1, plot_memorising_marginals=True)
    plot_results(frame, tmp_path, config)
    assert len(saved_figures) == 10

    plot_marginal_distributions(
        prior,
        tmp_path / "prior_without_memorising.png",
        title="Prior",
        plot_memorising=False,
    )
    plot_per_prompt_marginals(
        posterior,
        tmp_path / "posterior_without_memorising.png",
        title="Posterior",
        plot_memorising=False,
    )
    assert len(saved_figures) == 12


def test_lr_analysis_plotting_without_distribution_or_samples(
    tmp_path: Path,
    saved_figures: list[Path],
) -> None:
    frame = _analysis_frame().drop(
        columns=[
            "dist/ed_vs_baseline_memorising",
            "dist/ed_vs_baseline_generalising",
            "dist/sw_vs_baseline_memorising",
            "dist/sw_vs_baseline_generalising",
        ]
    )
    frame = frame.drop(columns=[column for column in frame if "data_random" in column])
    plot_results(
        frame,
        tmp_path,
        SweepConfig(compute_distribution_metrics=False, eval_position=None),
    )
    assert len(saved_figures) == 4
