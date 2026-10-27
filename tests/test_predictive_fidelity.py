"""Next-token fidelity against the exact predictive, for balls-and-urns and 1-Markov."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bft_pmc.analysis import predictive_fidelity as fidelity_module
from bft_pmc.analysis.predictive_fidelity import fidelity, main, scored_positions
from pfn_transformerlens.model.PFN import UnsupervisedPFN


def test_fidelity_scores_only_the_positions_both_predictives_define() -> None:
    exact = torch.softmax(torch.randn(3, 5, 4), dim=-1)
    model = torch.cat([exact, torch.softmax(torch.randn(3, 2, 4), dim=-1)], dim=1)
    assert scored_positions(exact, model) == 5
    value, per_position = fidelity(exact, model)
    assert value == pytest.approx(0.0, abs=1e-6)
    assert per_position.shape == (5,)


def test_self_test_passes_and_reports_both_families(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--self-test"])
    out = capsys.readouterr().out
    assert "BAU" in out and "Markov" in out
    assert "self-test passed" in out


def test_a_family_and_checkpoints_are_required() -> None:
    with pytest.raises(SystemExit, match="--family and --checkpoint-paths"):
        main([])


def _read_outputs(output_dir: Path, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(output_dir / f"fidelity_{family}.csv")
    per_position = pd.read_csv(output_dir / f"fidelity_{family}_per_position.csv")
    settings = json.loads((output_dir / f"fidelity_{family}_settings.json").read_text())
    assert settings["family"] == family
    return summary, per_position


@pytest.mark.parametrize("with_cache", [False, True])
def test_balls_and_urns_fidelity_writes_one_row_per_checkpoint(
    with_cache: bool,
    nope_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if with_cache:
        # predict_on_prompt returns (prediction, cache) when a cache is kept.
        real = UnsupervisedPFN.predict_on_prompt
        monkeypatch.setattr(
            UnsupervisedPFN,
            "predict_on_prompt",
            lambda self, *args, **kwargs: (real(self, *args, **kwargs), None),
        )
    output_dir = tmp_path / "bau"
    args = ["--family", "bau", "--checkpoint-paths", str(nope_checkpoint)]
    args += ["--seq-len", "20", "--n-sequences", "6", "--batch-size", "4"]
    main([*args, "--device", "cpu", "--output-dir", str(output_dir)])

    summary, per_position = _read_outputs(output_dir, "bau")
    assert summary["checkpoint"].tolist() == [nope_checkpoint.parent.name]
    assert summary["seq_len"].tolist() == [20]
    for law in ("in_distribution", "population"):
        assert np.isfinite(summary[f"sym_kl_{law}"]).all()
        assert (summary[f"sym_kl_{law}"] >= 0).all()
    label = nope_checkpoint.parent.name
    assert list(per_position.columns) == [
        "position",
        f"{label}_in_distribution",
        f"{label}_population",
    ]
    assert len(per_position) == 20


def test_markov_fidelity_uses_the_run_config_and_chain_pool(
    tiny_run: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "markov"
    args = ["--family", "markov", "--checkpoint-paths", str(tiny_run / "model.pt")]
    args += ["--config-path", str(tiny_run / "resolved_config.yaml")]
    args += ["--seq-len", "6", "--n-sequences", "5", "--batch-size", "2"]
    main([*args, "--output-dir", str(output_dir)])

    summary, per_position = _read_outputs(output_dir, "markov")
    assert summary["checkpoint"].tolist() == ["model"]
    assert np.isfinite(summary["sym_kl_population"]).all()
    # Both predictives give one next-state prediction per observed state.
    assert len(per_position) == 6


def test_runner_is_chosen_by_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake(path: str, args: object) -> dict[str, tuple[float, np.ndarray]]:
        calls.append(path)
        return {law: (0.5, np.zeros(3)) for law in ("in_distribution", "population")}

    monkeypatch.setattr(fidelity_module, "run_markov", fake)
    out = tmp_path / "out"
    main(
        [
            "--family",
            "markov",
            "--checkpoint-paths",
            "a/checkpoint_step_5.pt",
            "b.pt",
            "--output-dir",
            str(out),
        ]
    )
    assert calls == ["a/checkpoint_step_5.pt", "b.pt"]
    summary = pd.read_csv(out / "fidelity_markov.csv")
    # A checkpoint_step file is labelled by its run directory, anything else by its stem.
    assert summary["checkpoint"].tolist() == ["a", "b"]
