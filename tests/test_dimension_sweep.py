from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bft_pmc.analysis import dimension_sweep
from bft_pmc.analysis.dimension_sweep import (
    TITLE_WIDTH,
    DimensionSweepConfig,
    collect,
    common_length,
    load_source_frames,
    main,
    parse_inputs,
    plot,
)

LENGTHS = [8, 64, 500]


def _write_run(directory: Path, source: str, scale: float) -> None:
    """One recovery-vs-length CSV in the layout the diagnostics scripts write."""
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "rollout_length": LENGTHS,
            "dist/ed_vs_baseline_generalising": [scale * 3, scale * 2, scale],
            "dist/sw_vs_baseline_generalising": [scale * 6, scale * 4, scale * 2],
            "dist/ed_vs_baseline_memorising": [scale * 9, scale * 6, scale * 3],
            "dist/sw_vs_baseline_memorising": [scale * 12, scale * 8, scale * 4],
        }
    )
    frame.to_csv(directory / f"{source}_recovery_vs_length.csv", index=False)


@pytest.fixture
def runs(tmp_path: Path) -> dict[int, Path]:
    """Two runs at latent dimensions 5 and 11, both sources present."""
    made: dict[int, Path] = {}
    for dimension, scale in [(5, 0.01), (11, 0.02)]:
        directory = tmp_path / f"run_dim{dimension}"
        for source in ("prior", "posterior"):
            _write_run(directory, source, scale)
        made[dimension] = directory
    return made


def test_parse_inputs_orders_by_dimension() -> None:
    parsed = parse_inputs(["11=b", "5=a"])
    assert list(parsed) == [5, 11]
    assert parsed[5] == Path("a")


@pytest.mark.parametrize(
    "entries, message",
    [
        (["5"], "not of the form"),
        (["x=a"], "integer latent dimension"),
        (["0=a"], "must be positive"),
        (["5=a", "5=b"], "duplicate latent dimension"),
    ],
)
def test_parse_inputs_rejects_malformed_entries(
    entries: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_inputs(entries)


def test_parse_inputs_names_task_diversity_in_its_errors() -> None:
    with pytest.raises(ValueError, match="integer task diversity"):
        parse_inputs(["x=a"], "task-diversity")
    with pytest.raises(ValueError, match="<num_tasks>=<results_dir>"):
        parse_inputs(["32"], "task-diversity")


def test_load_source_frames_reads_every_directory(runs: dict[int, Path]) -> None:
    frames = load_source_frames(runs, "prior")
    assert set(frames) == {5, 11}
    assert list(frames[5]["rollout_length"]) == LENGTHS


def test_load_source_frames_reports_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prior_recovery_vs_length.csv"):
        load_source_frames({5: tmp_path / "absent"}, "prior")


def test_common_length_defaults_to_the_largest_shared_length(
    runs: dict[int, Path],
) -> None:
    frames = load_source_frames(runs, "prior")
    assert common_length(frames, None) == 500


def test_common_length_honours_an_explicit_request(runs: dict[int, Path]) -> None:
    frames = load_source_frames(runs, "prior")
    assert common_length(frames, 64) == 64


def test_common_length_rejects_a_length_missing_from_one_input(
    runs: dict[int, Path],
) -> None:
    frames = load_source_frames(runs, "prior")
    with pytest.raises(ValueError, match="not present in every input"):
        common_length(frames, 999)


def test_common_length_rejects_disjoint_grids(tmp_path: Path) -> None:
    frames = {
        5: pd.DataFrame({"rollout_length": [8]}),
        11: pd.DataFrame({"rollout_length": [16]}),
    }
    with pytest.raises(ValueError, match="share no common rollout length"):
        common_length(frames, None)


def test_collect_returns_one_row_per_source_and_dimension(
    runs: dict[int, Path],
) -> None:
    frame, length = collect(runs, ["prior", "posterior"], None)
    assert length == 500
    assert len(frame) == 4
    prior_five = frame[(frame["source"] == "prior") & (frame["latent_dim"] == 5)]
    assert prior_five["ed_generalising"].iloc[0] == pytest.approx(0.01)
    assert prior_five["sw_memorising"].iloc[0] == pytest.approx(0.04)


def test_collect_reads_the_requested_length(runs: dict[int, Path]) -> None:
    frame, length = collect(runs, ["prior"], 8)
    assert length == 8
    assert frame["ed_generalising"].iloc[0] == pytest.approx(0.03)


def test_collect_reports_a_missing_metric_column(tmp_path: Path) -> None:
    directory = tmp_path / "short"
    directory.mkdir()
    pd.DataFrame({"rollout_length": [8]}).to_csv(
        directory / "prior_recovery_vs_length.csv", index=False
    )
    other = tmp_path / "short2"
    other.mkdir()
    pd.DataFrame({"rollout_length": [8]}).to_csv(
        other / "prior_recovery_vs_length.csv", index=False
    )
    with pytest.raises(KeyError, match="ed_vs_baseline_memorising"):
        collect({5: directory, 11: other}, ["prior"], None)


def test_plot_writes_both_formats(runs: dict[int, Path], tmp_path: Path) -> None:
    frame, _ = collect(runs, ["prior", "posterior"], None)
    stem = tmp_path / "figure"
    plot(frame, ["prior", "posterior"], stem, "title")
    assert stem.with_suffix(".png").exists()
    assert stem.with_suffix(".pdf").exists()


def test_main_writes_the_csv_and_figure(
    runs: dict[int, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "aggregated"
    main(
        DimensionSweepConfig(
            inputs=[f"{dimension}={path}" for dimension, path in runs.items()],
            output_dir=str(output_dir),
        )
    )
    written = pd.read_csv(output_dir / "recovery_vs_dimension.csv")
    assert list(written["latent_dim"]) == [5, 11, 5, 11]
    assert (output_dir / "recovery_vs_dimension.png").exists()
    assert "rollout length 500" in capsys.readouterr().out

    record = json.loads((output_dir / "run_record.json").read_text())
    assert record["context"]["rollout_length"] == 500
    assert sorted(record["inputs"]) == sorted(
        str(path / f"{source}_recovery_vs_length.csv")
        for path in runs.values()
        for source in ("prior", "posterior")
    )
    assert all(len(digest) == 64 for digest in record["inputs"].values())
    assert record["context"]["input_directories"] == {
        str(dimension): str(path) for dimension, path in runs.items()
    }
    figure = json.loads((output_dir / "figures.json").read_text())[
        "recovery_vs_dimension"
    ]
    assert figure == {
        "kind": "dimension",
        "title": "Recovery error vs latent dimension (rollout length N=500)",
        "csv": "recovery_vs_dimension.csv",
        "sources": ["prior", "posterior"],
    }


def test_plot_wraps_a_long_title(
    runs: dict[int, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A title longer than the two-panel figure is wide wraps instead of clipping."""
    frame, _ = collect(runs, ["prior"], None)
    kept = []
    monkeypatch.setattr(dimension_sweep.plt, "close", kept.append)
    title = "Balls and urns (no positional encoding, M=32, context extended to 1024)"
    plot(frame, ["prior"], tmp_path / "figure", title)
    lines = kept[0].get_suptitle().splitlines()
    assert len(lines) == 2
    assert all(len(line) <= TITLE_WIDTH for line in lines)
    assert " ".join(lines) == title


def test_main_against_task_diversity_writes_its_own_files(
    runs: dict[int, Path], tmp_path: Path
) -> None:
    """The same runs keyed by task diversity M: its own column, stem and kind."""
    output_dir = tmp_path / "task_diversity"
    main(
        DimensionSweepConfig(
            inputs=[f"{m}={path}" for m, path in zip((128, 32), runs.values())],
            output_dir=str(output_dir),
            x_axis="task-diversity",
        )
    )
    written = pd.read_csv(output_dir / "recovery_vs_task_diversity.csv")
    assert list(written["num_tasks"]) == [32, 128, 32, 128]
    assert "latent_dim" not in written
    # M=32 is the second run, whose values are twice the first's.
    prior_32 = written[(written["source"] == "prior") & (written["num_tasks"] == 32)]
    assert prior_32["ed_generalising"].iloc[0] == pytest.approx(0.02)
    assert (output_dir / "recovery_vs_task_diversity.png").exists()
    assert not (output_dir / "recovery_vs_dimension.csv").exists()

    record = json.loads((output_dir / "run_record.json").read_text())
    assert record["context"]["family"] == "task_diversity"
    assert set(record["context"]["input_directories"]) == {"32", "128"}
    figure = json.loads((output_dir / "figures.json").read_text())[
        "recovery_vs_task_diversity"
    ]
    assert figure == {
        "kind": "task_diversity",
        "title": "Recovery error vs task diversity (rollout length N=500)",
        "csv": "recovery_vs_task_diversity.csv",
        "sources": ["prior", "posterior"],
    }


def test_main_keeps_an_explicit_title(runs: dict[int, Path], tmp_path: Path) -> None:
    output_dir = tmp_path / "titled"
    main(
        DimensionSweepConfig(
            inputs=[f"{dimension}={path}" for dimension, path in runs.items()],
            output_dir=str(output_dir),
            title="Markov",
        )
    )
    figure = json.loads((output_dir / "figures.json").read_text())
    assert figure["recovery_vs_dimension"]["title"] == "Markov (rollout length N=500)"


def test_validate_requires_at_least_two_inputs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        DimensionSweepConfig(inputs=["5=a"]).validate()


def test_validate_rejects_an_unknown_x_axis() -> None:
    config = DimensionSweepConfig(inputs=["5=a", "11=b"], x_axis="width")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown x axis 'width'"):
        config.validate()


def test_validate_rejects_an_unknown_source() -> None:
    config = DimensionSweepConfig(inputs=["5=a", "11=b"], sources=["prior", "middle"])
    with pytest.raises(ValueError, match="unknown source"):
        config.validate()
