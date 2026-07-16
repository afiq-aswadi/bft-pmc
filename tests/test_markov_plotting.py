from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from markov.plotting import (
    HeatmapThresholdLocation,
    _default_transitions,
    _empirical_cdf,
    _format_chain_label,
    _nearest_positions,
    _pmc_figsize,
    _sample_reference_values,
    _stationary_distribution,
    _style_axis,
    _to_numpy,
    _weighted_atomic_cdf,
    load_task_diversity_history,
    plot_distribution_distance_dynamics,
    plot_distribution_distance_sweep,
    plot_pmc_distribution_matrix,
    plot_pmc_distributions,
    plot_pmc_matrix_summary,
    plot_task_diversity,
    plot_task_diversity_heatmap,
    plot_transient,
    resolve_task_diversity_heatmap_location,
    training_posterior_weights,
    transition_count_matrix,
)


def _write_task_diversity_summary(
    root: Path,
    steps: list[int],
    history_gaps: list[float],
) -> Path:
    rows: list[dict[str, int | float | str]] = []
    for index, n_chains in enumerate([4, 8]):
        history_path = root / f"history_{n_chains}.csv"
        id_kl = np.linspace(0.2, 0.1, len(steps))
        gaps = np.asarray(history_gaps) + index * 0.01
        pd.DataFrame(
            {
                "step": steps,
                "id_kl": id_kl,
                "ood_kl": id_kl + gaps,
            }
        ).to_csv(history_path, index=False)
        rows.append(
            {
                "n_chains": n_chains,
                "final_id_kl": id_kl[-1],
                "final_ood_kl": id_kl[-1] + gaps[-1],
                "generalization_gap": gaps[-1],
                "csv_path": str(history_path),
            }
        )
    summary_path = root / f"summary_{len(steps)}.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


def _random_markov_matrices(
    rng: np.random.Generator,
    count: int,
    k: int = 2,
) -> np.ndarray:
    return rng.dirichlet(np.ones(k), size=(count, k))


def test_markov_plotting_style_and_numeric_helpers() -> None:
    fig, axes = plt.subplots(1, 4)
    axes[0].plot([1, 2], [1, 2], label="line")
    _style_axis(axes[0], xlabel="x", ylabel="y", log_x_base2=True, log_y=True)
    _style_axis(axes[1], xlabel="x", log_x=True, symlog_y=True)
    _style_axis(axes[2], xlabel="x", show_legend=False, grid=False)
    _style_axis(axes[3], xlabel="x")
    assert axes[0].get_xscale() == "log"
    assert axes[1].get_yscale() == "symlog"
    plt.close(fig)

    assert _format_chain_label(8) == r"$2^{3}$"
    assert _format_chain_label(3) == "3"
    assert _nearest_positions(np.array([1, 2, 4]), np.array([1.1, 1.2, 4])) == [0, 2]
    assert _default_transitions(2, 5) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert _pmc_figsize(1) == (10.0, 4.2)
    np.testing.assert_array_equal(_to_numpy(torch.tensor([1.0])), np.array([1.0]))
    np.testing.assert_array_equal(_to_numpy([1.0]), np.array([1.0]))

    matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    stationary = _stationary_distribution(matrix)
    np.testing.assert_allclose(stationary @ matrix, stationary)
    empty_weights = training_posterior_weights(matrix[None], np.array([], dtype=int))
    np.testing.assert_array_equal(empty_weights, np.ones(1))
    posterior_weights = training_posterior_weights(
        np.stack([matrix, matrix[::-1, ::-1]]),
        np.array([0, 1, 1]),
    )
    assert posterior_weights.sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(
        transition_count_matrix(np.array([0]), 2), np.zeros((2, 2))
    )
    np.testing.assert_array_equal(
        transition_count_matrix(np.array([0, 1, 1]), 2),
        np.array([[0, 1], [0, 1]]),
    )

    values = np.array([2.0, 1.0, 2.0])
    weights = np.array([0.2, 0.5, 0.3])
    np.testing.assert_allclose(
        _weighted_atomic_cdf(values, weights)[1], np.array([0.5, 0.7, 1.0])
    )
    np.testing.assert_allclose(
        _empirical_cdf(np.array([2.0, 1.0]))[1], np.array([0.5, 1.0])
    )

    rng = np.random.default_rng(1)
    sampled = _sample_reference_values(values, sample_size=4, rng=rng)
    weighted = _sample_reference_values(values, sample_size=4, rng=rng, weights=weights)
    assert sampled.shape == weighted.shape == (4,)


def test_task_diversity_plots_and_threshold_resolution(
    tmp_path: Path,
    saved_figures: list[Path],
) -> None:
    summary = _write_task_diversity_summary(tmp_path, [1, 10, 100], [0.2, 0.1, 0.01])
    summary_df, combined = load_task_diversity_history(summary)
    location = resolve_task_diversity_heatmap_location(
        summary_df,
        combined,
        gap_tolerance=0.05,
    )
    assert location == HeatmapThresholdLocation(4, 100)
    assert resolve_task_diversity_heatmap_location(
        summary_df,
        combined,
        gap_tolerance=0.001,
    ) == HeatmapThresholdLocation(None, None)

    threshold_without_step = summary_df.copy()
    threshold_without_step.loc[0, "generalization_gap"] = 0.0
    high_gap_history = combined.copy()
    high_gap_history["generalization_gap"] = 1.0
    assert resolve_task_diversity_heatmap_location(
        threshold_without_step,
        high_gap_history,
        gap_tolerance=0.05,
    ) == HeatmapThresholdLocation(4, None)

    plot_task_diversity(
        summary, tmp_path / "diversity.png", max_steps=100, context_len=8
    )
    plot_task_diversity_heatmap(summary, tmp_path / "heatmap.png")
    for steps in [[1], [1, 2]]:
        short_root = tmp_path / f"short_{len(steps)}"
        short_root.mkdir()
        short_summary = _write_task_diversity_summary(
            short_root,
            steps,
            [0.01] * len(steps),
        )
        plot_task_diversity_heatmap(short_summary, short_root / "heatmap.png")
    assert len(saved_figures) == 4

    empty_summary = tmp_path / "empty.csv"
    pd.DataFrame(columns=["n_chains"]).to_csv(empty_summary, index=False)
    with pytest.raises(ValueError, match="No rows"):
        load_task_diversity_history(empty_summary)


def test_markov_pmc_summary_and_distribution_plots(
    tmp_path: Path,
    saved_figures: list[Path],
) -> None:
    rng = np.random.default_rng(2)
    samples = _random_markov_matrices(rng, 32)
    training = _random_markov_matrices(rng, 4)
    target = training[0]

    plot_pmc_matrix_summary(
        samples, training, tmp_path / "summary.png", title="Summary"
    )
    plot_pmc_matrix_summary(
        samples[0],
        training[0],
        tmp_path / "summary_target.png",
        title="Target",
        target_matrix=torch.from_numpy(target),
    )
    plot_pmc_distributions(
        samples,
        training,
        tmp_path / "prior.png",
        title_prefix="",
        transitions_to_plot=[(0, 0), (0, 1)],
    )
    plot_pmc_distributions(
        np.full((2, 2), 0.5),
        training[0],
        tmp_path / "posterior.png",
        prompt_tokens=torch.tensor([0, 1]),
        target_matrix=target,
        transitions_to_plot=[(0, 0)],
    )
    assert len(saved_figures) == 4

    with pytest.raises(ValueError, match="Expected"):
        plot_pmc_distributions(np.ones(2), training, tmp_path / "bad.png")


def test_markov_pmc_matrix_plot_paths(
    tmp_path: Path,
    saved_figures: list[Path],
) -> None:
    rng = np.random.default_rng(7)
    samples = _random_markov_matrices(rng, 32)
    training = _random_markov_matrices(rng, 4)
    plot_pmc_distribution_matrix(
        samples,
        training,
        tmp_path / "density.png",
        mode="density",
        share_axes=True,
    )
    plot_pmc_distribution_matrix(
        np.full((16, 2, 2), 0.5),
        _random_markov_matrices(rng, 9),
        tmp_path / "posterior_density.png",
        title_prefix="",
        prompt_tokens=np.array([0, 1, 1]),
        mode="density",
        share_axes=False,
        max_classes=1,
    )
    plot_pmc_distribution_matrix(
        samples,
        training,
        tmp_path / "cdf_shared.png",
        mode="cdf",
        share_axes=True,
    )
    plot_pmc_distribution_matrix(
        np.full((2, 2), 0.5),
        np.full((2, 2), 0.5),
        tmp_path / "single_matrix.png",
        mode="cdf",
        max_classes=1,
    )
    plot_pmc_distribution_matrix(
        samples,
        training,
        tmp_path / "cdf_independent.png",
        prompt_tokens=np.array([0, 1]),
        mode="cdf",
        share_axes=False,
    )
    assert len(saved_figures) == 10

    with pytest.raises(ValueError, match="mode"):
        plot_pmc_distribution_matrix(
            samples, training, tmp_path / "bad.png", mode="bad"
        )
    with pytest.raises(ValueError, match="Expected"):
        plot_pmc_distribution_matrix(np.ones(2), training, tmp_path / "bad.png")


def _distance_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int | str | None]] = []
    for prompt_length, sources in [
        (0, [None]),
        (2, ["in_distribution", "out_of_distribution"]),
    ]:
        for source in sources:
            rows.append(
                {
                    "n_chains": 4,
                    "prompt_length": prompt_length,
                    "prompt_source": source,
                    "dist/ed_vs_baseline_in_distribution": 0.1,
                    "dist/ed_vs_baseline_out_of_distribution": 0.2,
                    "dist/sw_vs_baseline_in_distribution": 0.3,
                    "dist/sw_vs_baseline_out_of_distribution": 0.4,
                }
            )
    return pd.DataFrame(rows)


def _dynamics_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "step": [1, 2],
            "ed_vs_baseline_in_distribution": [0.1, 0.2],
            "ed_vs_baseline_out_of_distribution": [0.2, 0.3],
            "sw_vs_baseline_in_distribution": [0.3, 0.4],
            "sw_vs_baseline_out_of_distribution": [0.4, 0.5],
        }
    )
    for metric in ["ed", "sw"]:
        for baseline in ["in_distribution", "out_of_distribution"]:
            for source in ["in_distribution", "out_of_distribution"]:
                frame[f"{metric}_vs_baseline_{baseline}_from_prompts_{source}"] = [
                    0.1,
                    0.2,
                ]
    return frame


def test_markov_distance_and_transient_plots(
    tmp_path: Path,
    saved_figures: list[Path],
) -> None:
    sweep = _distance_frame()
    sweep_path = tmp_path / "sweep.csv"
    sweep.to_csv(sweep_path, index=False)
    plot_distribution_distance_sweep(
        sweep, tmp_path / "sweep_posterior.png", prompt_length=2
    )
    plot_distribution_distance_sweep(
        sweep_path, tmp_path / "sweep_prior.png", prompt_length=0
    )

    dynamics = _dynamics_frame()
    dynamics_path = tmp_path / "dynamics.csv"
    dynamics.to_csv(dynamics_path, index=False)
    plot_distribution_distance_dynamics(
        dynamics,
        tmp_path / "dynamics_posterior.png",
        mode="posterior",
        log_xscale=True,
    )
    plot_distribution_distance_dynamics(
        dynamics_path,
        tmp_path / "dynamics_prior.png",
        mode="prior",
    )

    transient_path = tmp_path / "transient.csv"
    pd.DataFrame(
        {
            "n_chains": [4, 4],
            "step": [2, 1],
            "id_kl": [0.1, 0.2],
            "ood_kl": [0.2, 0.3],
        }
    ).to_csv(transient_path, index=False)
    plot_transient(transient_path, tmp_path / "transient.png", n_chains=4)
    assert len(saved_figures) == 5

    with pytest.raises(ValueError, match="No rows"):
        plot_transient(transient_path, tmp_path / "missing.png", n_chains=8)
    with pytest.raises(ValueError, match="per-source"):
        plot_distribution_distance_dynamics(
            dynamics[["step"]],
            tmp_path / "bad.png",
            mode="posterior",
        )

    plot_pmc_distributions(
        np.full((4, 2, 2), 0.5),
        np.full((3, 2, 2), 0.5),
        tmp_path / "degenerate.png",
        transitions_to_plot=[(0, 0)],
    )
    with pytest.raises(ValueError, match="Unsupported"):
        plot_distribution_distance_dynamics(dynamics, tmp_path / "bad.png", mode="bad")
