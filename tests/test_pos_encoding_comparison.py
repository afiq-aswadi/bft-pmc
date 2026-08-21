from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis.pos_encoding_comparison as comparison


def _write_training_log(path: Path, *, losses: list[float]) -> Path:
    records = [
        {
            "step": (index + 1) * 100,
            "train/loss": value + 0.01,
            "eval/loss": value,
            "eval/accuracy": 0.5,
        }
        for index, value in enumerate(losses)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _write_metrics(path: Path, *, offset: float, task_column: str = "num_tasks") -> Path:
    frame = pd.DataFrame(
        {
            task_column: [1, 2, 1, 2],
            "prompt_source": ["prior", "prior", "memorising", "memorising"],
            "dist/ed_vs_baseline_memorising": [0.1 + offset, 0.2 + offset, 0.3, 0.4],
            "dist/sw_vs_baseline_memorising": [0.5, 0.6, 0.7, 0.8],
            "run_id": ["M1", "M2", "M1", "M2"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def test_num_tasks_parsed_from_trainer_log_filenames(tmp_path: Path) -> None:
    assert comparison._num_tasks_from_name(tmp_path / "num_tasks_32.json") == 32
    assert comparison._num_tasks_from_name(tmp_path / "other.json") is None
    assert comparison._num_tasks_from_name(tmp_path / "num_tasks_all.json") is None


def test_load_training_logs_accepts_files_and_directories(tmp_path: Path) -> None:
    single = _write_training_log(tmp_path / "learned.json", losses=[1.0, 0.9])
    directory = tmp_path / "rope" / "logs"
    _write_training_log(directory / "num_tasks_32.json", losses=[1.1, 0.8])

    frame = comparison.load_training_logs(
        (f"learned={single}", f"rope={tmp_path / 'rope'}")
    )
    assert set(frame["pos_encoding"]) == {"learned", "rope"}
    assert frame.loc[frame["pos_encoding"] == "rope", "num_tasks"].iloc[0] == 32
    assert frame["num_tasks"].isna().any()

    with pytest.raises(FileNotFoundError, match="No trainer log files"):
        comparison.load_training_logs((f"empty={tmp_path / 'nothing'}",))


def test_load_sweep_metrics_aligns_the_task_axis(tmp_path: Path) -> None:
    learned = _write_metrics(tmp_path / "learned" / "metrics.csv", offset=0.0)
    _write_metrics(
        tmp_path / "markov" / "metrics.csv", offset=0.5, task_column="n_chains"
    )

    frame = comparison.load_sweep_metrics(
        (f"learned={learned}", f"markov={tmp_path / 'markov'}")
    )
    assert "num_tasks" in frame.columns
    assert "n_chains" not in frame.columns
    assert set(frame["pos_encoding"]) == {"learned", "markov"}

    with pytest.raises(FileNotFoundError, match="No metrics.csv"):
        comparison.load_sweep_metrics((f"none={tmp_path / 'missing'}",))


def test_final_training_losses_takes_the_last_evaluation(tmp_path: Path) -> None:
    log = _write_training_log(tmp_path / "learned.json", losses=[1.0, 0.7])
    frame = comparison.load_training_logs((f"learned={log}",))
    final = comparison.final_training_losses(frame)
    assert final["eval/loss"].item() == pytest.approx(0.7)
    assert final["step"].item() == 200

    with pytest.raises(ValueError, match="no eval/loss"):
        comparison.final_training_losses(frame.drop(columns=["eval/loss"]))


def test_compare_metrics_reports_the_spread_across_encodings(tmp_path: Path) -> None:
    learned = _write_metrics(tmp_path / "learned" / "metrics.csv", offset=0.0)
    rope = _write_metrics(tmp_path / "rope" / "metrics.csv", offset=0.5)
    frame = comparison.load_sweep_metrics((f"learned={learned}", f"rope={rope}"))

    wide = comparison.compare_metrics(frame)
    assert {"learned", "rope", "spread", "metric"} <= set(wide.columns)
    ed_rows = wide[wide["metric"] == "dist/ed_vs_baseline_memorising"]
    assert ed_rows["spread"].max() == pytest.approx(0.5)
    # sliced Wasserstein is identical across encodings here
    sw_rows = wide[wide["metric"] == "dist/sw_vs_baseline_memorising"]
    assert sw_rows["spread"].max() == pytest.approx(0.0)

    with pytest.raises(ValueError, match="No namespaced metric"):
        comparison.compare_metrics(pd.DataFrame({"pos_encoding": ["a"], "x": [1]}))


def test_plots_are_written(tmp_path: Path) -> None:
    log = _write_training_log(tmp_path / "learned.json", losses=[1.0, 0.7])
    training = comparison.load_training_logs((f"learned={log}",))
    assert comparison.plot_training_curves(training, tmp_path / "curves.png").is_file()
    with pytest.raises(ValueError, match="No eval/loss rows"):
        comparison.plot_training_curves(training.iloc[:0], tmp_path / "empty.png")

    metrics = comparison.load_sweep_metrics(
        (f"learned={_write_metrics(tmp_path / 'l' / 'metrics.csv', offset=0.0)}",)
    )
    assert comparison.plot_metric_panels(metrics, tmp_path / "panels.png").is_file()
    with pytest.raises(ValueError, match="No namespaced metric"):
        comparison.plot_metric_panels(
            pd.DataFrame({"pos_encoding": ["a"], "num_tasks": [1]}),
            tmp_path / "bad.png",
        )


def test_encoding_colors_fall_back_for_unknown_labels() -> None:
    assert comparison._encoding_color("rope") == comparison.ENCODING_COLORS["rope"]
    assert comparison._encoding_color("mystery") == comparison._FALLBACK_COLOR


def test_render_summary_handles_either_input() -> None:
    assert "Positional-encoding comparison" in comparison.render_summary(None, None)
    losses = pd.DataFrame(
        {
            "pos_encoding": ["learned", "rope"],
            "num_tasks": [32, 32],
            "step": [100, 100],
            "eval/loss": [1.0, 0.9],
            "train/loss": [1.1, 1.0],
        }
    )
    text = comparison.render_summary(losses, None)
    assert "lowest eval loss: rope" in text

    wide = pd.DataFrame({"metric": ["m"], "learned": [1.0], "rope": [2.0], "spread": [1.0]})
    assert "Largest disagreements" in comparison.render_summary(None, wide)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="label=path"):
        comparison.ComparisonConfig().validate()
    comparison.ComparisonConfig(training_logs=("a=b",)).validate()


def test_main_writes_tables_figures_and_summary(tmp_path: Path) -> None:
    learned_log = _write_training_log(tmp_path / "learned.json", losses=[1.0, 0.7])
    rope_log = _write_training_log(tmp_path / "rope.json", losses=[1.0, 0.6])
    learned_metrics = _write_metrics(tmp_path / "learned" / "metrics.csv", offset=0.0)
    rope_metrics = _write_metrics(tmp_path / "rope" / "metrics.csv", offset=0.5)

    out_dir = tmp_path / "out"
    comparison.main(
        comparison.ComparisonConfig(
            training_logs=(f"learned={learned_log}", f"rope={rope_log}"),
            sweep_metrics=(f"learned={learned_metrics}", f"rope={rope_metrics}"),
            output_dir=str(out_dir),
        )
    )
    for name in (
        "training_curves.csv",
        "final_training_loss.csv",
        "training_curves.png",
        "sweep_metrics.csv",
        "metric_comparison.csv",
        "metric_panels.png",
        "summary.md",
        "config.json",
    ):
        assert (out_dir / name).is_file(), name
    assert "rope" in (out_dir / "summary.md").read_text(encoding="utf-8")

    # training logs alone are enough
    training_only = tmp_path / "training_only"
    comparison.main(
        comparison.ComparisonConfig(
            training_logs=(f"learned={learned_log}",),
            output_dir=str(training_only),
        )
    )
    assert (training_only / "final_training_loss.csv").is_file()
    assert not (training_only / "metric_comparison.csv").exists()
