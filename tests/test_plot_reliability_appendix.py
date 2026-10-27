"""The reliability-appendix figures, drawn from synthetic diagnostics CSVs.

Every input CSV is written on demand, the first time the generator reads it, in
the layout the rollout-diagnostics runs save. The test therefore exercises the
real run names and file layout without duplicating the figure list.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bft_pmc.analysis import plot_reliability_appendix as appendix

REFS = ("memorising", "generalising")
METRICS = ("ed", "sw")
COUNTS = [16, 64, 256, 1024]


def _lengths(run: str) -> list[int]:
    if "n500" in run or "pi544" in run:
        return [8, 16, 32, 64, 128, 223, 256, 384, 500]
    return [8, 16, 32, 64, 128, 223]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthetic_rows(path: Path) -> list[dict[str, object]]:
    """Rows for one diagnostics CSV, chosen by its file name."""
    name, run = path.name, path.parent.name
    if name.endswith("_recovery_vs_length.csv"):
        # 1/N spans a factor of 62 to N = 500 (log axis) and 28 to N = 223.
        return [
            {
                "rollout_length": n,
                **{f"dist/{m}_vs_baseline_{r}": 1.0 / n for m in METRICS for r in REFS},
            }
            for n in _lengths(run)
        ]
    if name.endswith("_recovery_vs_count.csv"):
        rows = []
        for r_count in COUNTS:
            row: dict[str, object] = {"num_rollouts": r_count}
            for m in METRICS:
                for ref in REFS:
                    row[f"dist/{m}_vs_baseline_{ref}/mean"] = 1.0 / r_count
                    row[f"dist/{m}_vs_baseline_{ref}/sd"] = 0.1 / r_count
            rows.append(row)
        return rows
    if name in {"recovery_vs_dimension.csv", "recovery_vs_task_diversity.csv"}:
        xkey = "latent_dim" if name == "recovery_vs_dimension.csv" else "num_tasks"
        return [
            {
                "source": source,
                xkey: x,
                **{f"{m}_{ref}": 1.0 / x for m in METRICS for ref in REFS},
            }
            for source in ("prior", "posterior")
            for x in (32, 8, 128)  # unsorted on purpose
        ]
    if name == "path_stability_per_prompt.csv":
        return [
            {
                "family": family,
                "pos_encoding": encoding,
                "setup": f"{family}-{encoding}",
                "prompt_idx": prompt,
                "T": t,
                "scaled_l1": 0.01 * t * (prompt + 1),
            }
            for family, encoding in [
                ("lr", "none"),
                ("bau", "none"),
                ("markov", "markov"),
            ]
            for prompt in range(2)
            for t in (3, 1, 2)  # unsorted on purpose
        ]
    raise AssertionError(f"unexpected input {path}")


@pytest.fixture
def drawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Draw everything into tmp_path/paper from on-demand synthetic inputs."""
    root = tmp_path / "pmc_diagnostics"
    real_read_rows = appendix.read_rows

    def read_rows(path: Path) -> list[dict[str, str]]:
        if not path.exists():
            _write(path, synthetic_rows(path))
            if path.parent.name.startswith("lr_m32"):
                (path.parent / "run_record.json").write_text('{"config": {}}')
        return real_read_rows(path)

    monkeypatch.setattr(appendix, "read_rows", read_rows)
    paper = tmp_path / "paper"
    appendix.main(["--paper-dir", str(paper), "--diagnostics-root", str(root)])
    return root, paper


def test_every_appendix_and_supplement_figure_is_drawn(
    drawn: tuple[Path, Path],
) -> None:
    _, paper = drawn
    manifest = json.loads((paper / appendix.MANIFEST_NAME).read_text())
    figures = sorted(entry["figure"] for entry in manifest["figures"])
    assert len(figures) == 41
    assert len(manifest["supplement"]) == 26
    for figure in figures:
        assert (paper / figure).stat().st_size > 0
    expected = {
        "figures/reliability_length_prior.pdf",
        "figures/reliability_count_posterior.pdf",
        "figures/reliability_dimension_markov.pdf",
        "figures/reliability_taskdiv_length_bau.pdf",
        "figures/reliability_path_stability_markov.pdf",
        "figures/reliability_supp_encodings_n500_prior_count.pdf",
    }
    assert expected <= set(figures)
    assert manifest["generator"] == "bft_pmc/analysis/plot_reliability_appendix.py"


def test_manifest_records_the_plotted_values_and_input_hashes(
    drawn: tuple[Path, Path],
) -> None:
    root, paper = drawn
    manifest = json.loads((paper / appendix.MANIFEST_NAME).read_text())
    entries = {entry["figure"]: entry for entry in manifest["figures"]}

    length = entries["figures/reliability_length_prior.pdf"]
    first = length["panels"][0]
    assert first["label"] == "Linear regression"
    assert first["x"] == [8.0, 16.0, 32.0, 64.0, 128.0, 223.0, 256.0, 384.0, 500.0]
    assert first["series"]["memorising"][0] == pytest.approx(1 / 8)
    lr_input = length["inputs"][0]
    assert lr_input["path"] == str(
        root / "lr_m32_none_n500_pools_h100" / "prior_recovery_vs_length.csv"
    )
    assert lr_input["sha256"] == appendix.sha256(Path(lr_input["path"]))
    assert "run_record_sha256" in lr_input
    assert "run_record" not in length["inputs"][1]  # BAU has no run record here

    count = entries["figures/reliability_count_posterior.pdf"]
    assert count["panels"][0]["series"]["generalising_sd"][-1] == pytest.approx(
        0.1 / 1024
    )

    dimension = entries["figures/reliability_dimension_bau.pdf"]
    assert dimension["panels"][0]["x"] == [8.0, 32.0, 128.0]  # sorted

    jellyfish = entries["figures/reliability_path_stability_lr.pdf"]
    assert jellyfish["panels"] == []
    assert jellyfish["inputs"][0]["path"].endswith(
        "jellyfish_m32_n500/path_stability_per_prompt.csv"
    )


def test_markov_n500_runs_follow_their_submission_names() -> None:
    assert appendix.markov_n500_run(32) == "markov_m32_n500_pi544"
    assert appendix.markov_n500_run(64) == "markov_m64_n500"
