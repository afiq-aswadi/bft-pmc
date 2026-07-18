from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from balls_and_urns import plot_dynamics_combined as bau_dynamics
from balls_and_urns import plot_sweep_combined as bau_sweep
from balls_and_urns import plot_sweep_prior as bau_prior
from linear_regression import plot_dynamics_combined as lr_dynamics
from linear_regression import plot_stitched_sweep_dynamics as lr_stitched
from linear_regression import plot_sweep_combined as lr_sweep
from linear_regression import plot_sweep_prior as lr_prior
from markov import plot_dynamics_combined as markov_dynamics
from markov import plot_sweep_combined as markov_sweep
from markov import plot_sweep_prior as markov_prior


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DATA = REPO_ROOT / "paper_data"


def test_lr_aggregate_figure_entry_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_figures: list[Path],
) -> None:
    sweep_csv = PAPER_DATA / "lr/sweep/metrics.csv"
    dynamics_csv = PAPER_DATA / "lr/dynamics/metrics.csv"
    lr_sweep.main(lr_sweep.PlotConfig(str(sweep_csv), str(tmp_path), prompt_length=8))
    lr_sweep.main(lr_sweep.PlotConfig(str(sweep_csv), str(tmp_path), prompt_length=0))
    lr_dynamics.main(lr_dynamics.PlotConfig(str(dynamics_csv), str(tmp_path)))

    prior_df = pd.DataFrame(
        {
            "step": [1, 2],
            "ed_vs_baseline_memorising": [0.2, 0.1],
            "ed_vs_baseline_generalising": [0.3, 0.2],
            "sw_vs_baseline_memorising": [0.4, 0.3],
            "sw_vs_baseline_generalising": [0.5, 0.4],
        }
    )
    lr_dynamics._plot_dynamics(prior_df, tmp_path / "prior.png")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_sweep_prior.py",
            "--metrics-csv",
            str(sweep_csv),
            "--delta-csv",
            str(PAPER_DATA / "lr/sweep/prior_delta_mse.csv"),
            "--out-path",
            str(tmp_path / "lr_prior.png"),
        ],
    )
    lr_prior.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_stitched_sweep_dynamics.py",
            "--sweep-csv",
            str(sweep_csv),
            "--dynamics-csv",
            str(dynamics_csv),
            "--out-path",
            str(tmp_path / "lr_stitched.png"),
        ],
    )
    lr_stitched.main()
    # 10 aggregate figures plus the three intro-preview single-panel figures
    # (sweep_ed_id_1x1 and the two dynamics ed_id_1x1 variants).
    assert len(saved_figures) == 13


def test_lr_plot_helpers_validate_series_and_scales() -> None:
    fig, axes = plt.subplots(1, 3)
    frame = pd.DataFrame(
        {
            "x": [1, 2],
            "num_tasks": [1, 2],
            "step": [1, 2],
            "present": [1.0, 0.5],
            "second": [0.8, 0.4],
            "all_nan": [np.nan, np.nan],
        }
    )
    lr_sweep.plot_series(axes[0], frame["x"], ["present", "second"], frame)
    lr_dynamics.plot_series(
        axes[1],
        frame["x"],
        ["present", "second"],
        frame,
        lr_dynamics.SERIES,
    )
    for plotter, series in [
        (lr_sweep.plot_series, lr_sweep.SERIES),
        (lr_dynamics.plot_series, lr_dynamics.SERIES),
    ]:
        with pytest.raises(ValueError, match="empty"):
            plotter(axes[0], pd.Series(dtype=float), ["a", "b"], pd.DataFrame(), series)
        with pytest.raises(KeyError, match="missing plot columns"):
            plotter(axes[0], frame["x"], ["present", "missing"], frame, series)
        with pytest.raises(ValueError, match="missing values"):
            plotter(axes[0], frame["x"], ["present", "all_nan"], frame, series)
    with pytest.raises(KeyError, match="missing plot columns"):
        lr_prior._plot_panel(axes[0], frame["x"], frame, ["present", "missing"])
    with pytest.raises(ValueError, match="missing values"):
        lr_prior._plot_panel(axes[0], frame["x"], frame, ["present", "all_nan"])
    for plotter, axis_name in [
        (lr_stitched._plot_sweep_panel, "sweep"),
        (lr_stitched._plot_dyn_panel, "dynamics"),
    ]:
        with pytest.raises(ValueError, match="empty"):
            plotter(axes[0], pd.DataFrame(), ["a", "b"])
        with pytest.raises(KeyError, match=f"missing {axis_name} columns"):
            plotter(axes[0], frame, ["present", "missing"])
        with pytest.raises(ValueError, match="missing values"):
            plotter(axes[0], frame, ["present", "all_nan"])
    lr_prior._style_ax(axes[2], symlog_thresh=1e-3)
    lr_stitched._style(
        axes[0],
        xlabel="x",
        ylabel=None,
        xscale="linear",
        label_fs=8,
        tick_fs=7,
    )
    assert axes[2].get_yscale() == "symlog"
    plt.close(fig)


def test_bau_aggregate_figure_entry_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_figures: list[Path],
) -> None:
    sweep_csv = PAPER_DATA / "bau/sweep/metrics.csv"
    dynamics_csv = PAPER_DATA / "bau/dynamics/metrics.csv"
    bau_sweep.main(bau_sweep.PlotConfig(str(sweep_csv), str(tmp_path)))
    bau_dynamics.main(bau_dynamics.PlotConfig(str(dynamics_csv), str(tmp_path)))

    prior_df = pd.DataFrame(
        {
            "step": [1, 2],
            "ed_vs_baseline_memorising": [0.2, 0.1],
            "ed_vs_baseline_generalising": [0.3, 0.2],
            "sw_vs_baseline_memorising": [0.4, 0.3],
            "sw_vs_baseline_generalising": [0.5, 0.4],
        }
    )
    bau_dynamics._plot_dynamics(prior_df, tmp_path / "bau_prior_dynamics.png")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_sweep_prior.py",
            "--metrics-csv",
            str(PAPER_DATA / "bau/sweep/metrics_prior_n1024.csv"),
            "--kl-csv",
            str(PAPER_DATA / "bau/sweep/prior_predictive_kl.csv"),
            "--out-path",
            str(tmp_path / "bau_prior.png"),
        ],
    )
    bau_prior.main()
    assert len(saved_figures) == 9


def test_bau_plot_helpers_validate_series_and_scales() -> None:
    fig, axes = plt.subplots(1, 3)
    frame = pd.DataFrame(
        {
            "x": [1, 2],
            "present": [1.0, 0.5],
            "second": [0.8, 0.4],
            "all_nan": [np.nan, np.nan],
        }
    )
    bau_sweep.plot_series(
        axes[0], frame["x"], ["present", "second"], frame, bau_sweep.SERIES
    )
    bau_dynamics.plot_series(
        axes[1],
        frame["x"],
        ["present", "second"],
        frame,
        bau_dynamics.SERIES,
    )
    for plotter, series in [
        (bau_sweep.plot_series, bau_sweep.SERIES),
        (bau_dynamics.plot_series, bau_dynamics.SERIES),
    ]:
        with pytest.raises(ValueError, match="empty"):
            plotter(axes[0], pd.Series(dtype=float), ["a", "b"], pd.DataFrame(), series)
        with pytest.raises(KeyError, match="missing plot columns"):
            plotter(axes[0], frame["x"], ["present", "missing"], frame, series)
        with pytest.raises(ValueError, match="missing values"):
            plotter(axes[0], frame["x"], ["present", "all_nan"], frame, series)
    with pytest.raises(KeyError, match="missing plot columns"):
        bau_prior._plot_panel(axes[0], frame["x"], frame, ["present", "missing"])
    with pytest.raises(ValueError, match="missing values"):
        bau_prior._plot_panel(axes[0], frame["x"], frame, ["present", "all_nan"])
    bau_prior._style_ax(axes[2], symlog_thresh=1e-3)
    assert axes[2].get_yscale() == "symlog"
    plt.close(fig)


def test_lr_and_bau_sweep_plotters_reject_incomplete_prompt_sources(
    tmp_path: Path,
) -> None:
    base = {
        "run_id": ["run"],
        "num_tasks": [2],
        "checkpoint_step": [1],
        "prompt_length": [8],
    }
    lr_csv = tmp_path / "lr.csv"
    pd.DataFrame({**base, "prompt_source": ["memorising"]}).to_csv(
        lr_csv,
        index=False,
    )
    with pytest.raises(ValueError, match="complete posterior"):
        lr_sweep.main(lr_sweep.PlotConfig(str(lr_csv), str(tmp_path), prompt_length=8))

    pd.DataFrame({**base, "prompt_length": [0], "prompt_source": ["other"]}).to_csv(
        lr_csv,
        index=False,
    )
    with pytest.raises(ValueError, match="no prior rows"):
        lr_sweep.main(lr_sweep.PlotConfig(str(lr_csv), str(tmp_path), prompt_length=0))

    bau_csv = tmp_path / "bau.csv"
    pd.DataFrame({**base, "prompt_source": ["data_memorising"]}).to_csv(
        bau_csv,
        index=False,
    )
    with pytest.raises(ValueError, match="both prompt sources"):
        bau_sweep.main(bau_sweep.PlotConfig(str(bau_csv), str(tmp_path)))

    pd.DataFrame({**base, "prompt_source": ["none"]}).to_csv(bau_csv, index=False)
    with pytest.raises(ValueError, match="no posterior or prior"):
        bau_sweep.main(bau_sweep.PlotConfig(str(bau_csv), str(tmp_path)))


def test_markov_aggregate_figure_entry_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_figures: list[Path],
) -> None:
    sweep_root = PAPER_DATA / "markov/sweep"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_sweep_combined.py",
            "--runs-dir",
            str(sweep_root / "runs"),
            "--metrics-csv",
            str(sweep_root / "metrics.csv"),
            "--out-path",
            str(tmp_path / "markov_sweep.png"),
        ],
    )
    markov_sweep.main()

    for dynamics_root in [
        PAPER_DATA / "markov/dynamics_m32",
        PAPER_DATA / "markov/dynamics_m8",
    ]:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "plot_dynamics_combined.py",
                "--runs-dir",
                str(dynamics_root / "runs"),
                "--out-dir",
                str(tmp_path),
            ],
        )
        markov_dynamics.main()

    prior_args = [
        "plot_sweep_prior.py",
        "--metrics-csv",
        str(sweep_root / "metrics_prior_n1024.csv"),
        "--out-path",
        str(tmp_path / "markov_prior.png"),
    ]
    monkeypatch.setattr(sys, "argv", prior_args)
    markov_prior.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *prior_args,
            "--kl-csv",
            str(sweep_root / "prior_predictive_kl.csv"),
        ],
    )
    markov_prior.main()
    assert len(saved_figures) == 8


def test_markov_aggregate_plot_edge_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_figures: list[Path],
) -> None:
    metrics = pd.DataFrame(
        {
            "n_chains": [2, 2],
            "prompt_source": ["in_distribution", "out_of_distribution"],
            "dist/ed_vs_baseline_in_distribution": [0.1, 0.2],
            "dist/ed_vs_baseline_out_of_distribution": [0.2, 0.3],
            "dist/sw_vs_baseline_in_distribution": [0.1, 0.2],
            "dist/sw_vs_baseline_out_of_distribution": [0.2, 0.3],
        }
    )
    metrics_csv = tmp_path / "metrics.csv"
    metrics.to_csv(metrics_csv, index=False)

    figure, axis = plt.subplots()
    with pytest.raises(KeyError, match="missing plot column"):
        markov_dynamics._plot_metric(axis, metrics, metric="ed")
    nan_metrics = pd.DataFrame(
        {
            "step": [1],
            "ed_vs_baseline_in_distribution": [np.nan],
            "ed_vs_baseline_out_of_distribution": [0.1],
        }
    )
    with pytest.raises(ValueError, match="missing values"):
        markov_dynamics._plot_metric(axis, nan_metrics, metric="ed")
    with pytest.raises(KeyError, match="missing prompt-source"):
        markov_dynamics._filter_prompt_source(
            metrics.assign(step=[1, 2]), "in_distribution"
        )
    with pytest.raises(ValueError, match="empty Markov sweep"):
        markov_sweep._plot_panel(
            axis, [], mem_attr="ed_vs_memorising", gen_attr="ed_vs_generalising"
        )
    plt.close(figure)

    empty_root = tmp_path / "empty_runs"
    empty_root.mkdir()
    with pytest.raises(ValueError, match="no complete rows"):
        markov_sweep._build_rows(empty_root, metrics_csv)

    malformed_root = tmp_path / "malformed"
    malformed_root.mkdir()
    (malformed_root / "unparseable").mkdir()
    metrics.to_csv(malformed_root / "unparseable/metrics.csv", index=False)
    with pytest.raises(ValueError, match="cannot parse n_chains"):
        markov_sweep._build_rows(malformed_root, metrics_csv)

    missing_root = tmp_path / "missing_history"
    missing_root.mkdir()
    (missing_root / "run_chains2").mkdir()
    with pytest.raises(FileNotFoundError):
        markov_sweep._build_rows(missing_root, metrics_csv)

    run_dir = tmp_path / "run_chains2"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        markov_dynamics._process_run(run_dir, tmp_path, n_chains=2)
    metrics["step"] = [1, 2]
    for source in ("in_distribution", "out_of_distribution"):
        for metric in ("ed", "sw"):
            for baseline in ("in_distribution", "out_of_distribution"):
                metrics[f"{metric}_vs_baseline_{baseline}_from_prompts_{source}"] = [
                    0.2,
                    0.1,
                ]
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    with pytest.raises(FileNotFoundError):
        markov_dynamics._process_run(run_dir, tmp_path, n_chains=2)
    pd.DataFrame(
        {
            "step": [1, 2],
            "kl/id/wellspec/memorising": [0.2, 0.1],
            "kl/id/wellspec/generalising": [0.3, 0.2],
            "kl/ood/wellspec/memorising": [0.4, 0.3],
            "kl/ood/wellspec/generalising": [0.5, 0.4],
        }
    ).to_csv(run_dir / "wandb_kl_history.csv", index=False)
    markov_dynamics._process_run(run_dir, tmp_path, n_chains=2)
    assert len(saved_figures) == 2

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_dynamics_combined.py",
            "--runs-dir",
            str(malformed_root),
            "--out-dir",
            str(tmp_path / "figures"),
        ],
    )
    with pytest.raises(ValueError, match="cannot parse n_chains"):
        markov_dynamics.main()


def test_intro_identifiability_figure_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.plot_intro_identifiability as intro

    out_path = tmp_path / "intro_identifiability_spike_slab.png"
    monkeypatch.setattr(sys, "argv", ["plot_intro_identifiability.py", "--out-path", str(out_path)])
    intro.main()
    assert out_path.is_file()
