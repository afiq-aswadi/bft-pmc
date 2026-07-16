from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import tyro
import yaml

import balls_and_urns.sweep_analysis as bau_sweep
import linear_regression.sweep_analysis as lr_sweep
import markov.run_pmc as run_pmc
import markov.sweep_analysis as markov_sweep
import sweeps.run_yaml_sweep as yaml_sweep


ROOT = Path(__file__).resolve().parents[1]
PAPER_SWEEP_CONFIGS = {
    "bau_task_diversity_sweep.yaml",
    "lr_task_diversity_sweep.yaml",
    "markov_task_diversity_sweep.yaml",
}


def test_sweep_defaults_use_family_scoped_paths() -> None:
    assert lr_sweep.SweepConfig().checkpoint_root == "checkpoints/lr/task_diversity"
    assert lr_sweep.SweepConfig().output_dir == "outputs/lr/sweep_analysis"
    assert bau_sweep.SweepConfig().checkpoint_root == "checkpoints/bau/task_diversity"
    assert bau_sweep.SweepConfig().output_dir == "outputs/bau/sweep_analysis"
    assert (
        markov_sweep.SweepConfig().checkpoint_root
        == "checkpoints/markov/task_diversity"
    )
    assert markov_sweep.SweepConfig().output_dir == "outputs/markov/sweep_analysis"


@pytest.mark.parametrize(
    ("command", "expected_options"),
    [
        ("lr-sweep", ("--checkpoint-root", "--output-dir")),
        (
            "bau-sweep",
            ("--checkpoint-root", "--eval-dataset-dir", "--output-dir"),
        ),
        (
            "markov-sweep",
            ("--checkpoint-root", "--training-output-root", "--output-dir"),
        ),
        ("plot-lr-sweep", ("--metrics-csv", "--output-dir")),
        ("plot-lr-dynamics", ("--metrics-csv", "--output-dir")),
        ("plot-bau-sweep", ("--metrics-csv", "--output-dir")),
        ("plot-bau-dynamics", ("--metrics-csv", "--output-dir")),
    ],
)
def test_public_eval_cli_uses_top_level_options(
    command: str,
    expected_options: tuple[str, ...],
) -> None:
    result = subprocess.run(
        [sys.executable, "eval.py", command, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--config." not in result.stdout
    for option in expected_options:
        assert option in result.stdout


def test_lr_sweep_analysis_writes_tables_and_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = lr_sweep.SweepConfig(output_dir=str(tmp_path))
    metrics = pd.DataFrame({"step": [1], "mse": [0.5]})
    per_prompt = pd.DataFrame({"prompt": [0], "mse": [0.5]})
    plot_calls: list[Path] = []
    monkeypatch.setattr(
        lr_sweep,
        "run_analysis",
        lambda received, output_dir: (metrics, per_prompt),
    )
    monkeypatch.setattr(
        lr_sweep,
        "plot_results",
        lambda frame, output_dir, received: plot_calls.append(Path(output_dir)),
    )

    lr_sweep.main(config)

    output_dirs = list(tmp_path.glob("sweep_*"))
    assert len(output_dirs) == 1
    assert (output_dirs[0] / "config.json").exists()
    assert (output_dirs[0] / "metrics.csv").exists()
    assert (output_dirs[0] / "per_prompt_metrics.csv").exists()
    assert plot_calls == output_dirs

    monkeypatch.setattr(
        lr_sweep,
        "run_analysis",
        lambda received, output_dir: (metrics, pd.DataFrame()),
    )
    lr_sweep.main(replace(config, output_dir=str(tmp_path / "empty")))
    empty_output = next((tmp_path / "empty").glob("sweep_*"))
    assert not (empty_output / "per_prompt_metrics.csv").exists()


def test_pmc_entrypoint_builds_sampling_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = SimpleNamespace(model=object(), dataset=object())
    load_calls: list[tuple[Path, Path]] = []
    generation_calls: list[tuple[object, object, object, Path]] = []
    monkeypatch.setattr(
        run_pmc,
        "load_trained_markov_artifacts",
        lambda config, checkpoint: (
            load_calls.append((Path(config), Path(checkpoint))) or artifacts
        ),
    )
    monkeypatch.setattr(
        run_pmc,
        "generate_and_save_pmc_artifacts",
        lambda model, dataset, sampling, output_dir: generation_calls.append(
            (model, dataset, sampling, Path(output_dir))
        ),
    )
    config = run_pmc.PMCConfig(
        config_path=tmp_path / "config.yaml",
        checkpoint_path=tmp_path / "model.pt",
        output_dir=tmp_path / "pmc",
        num_samples=7,
        prompt_len=3,
        generation_length=10,
        seed=4,
    )

    run_pmc.main(config)

    assert load_calls == [(config.config_path, config.checkpoint_path)]
    assert generation_calls[0][0:2] == (artifacts.model, artifacts.dataset)
    assert generation_calls[0][2] == run_pmc.PMCSamplingConfig(7, 3, 10, 4)
    assert generation_calls[0][3] == config.output_dir


def test_yaml_sweep_helpers_and_device_queue(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert yaml_sweep.expand_grid({}) == [{}]
    assert yaml_sweep.expand_grid({"x": ["1", "2"], "y": ["a"]}) == [
        {"x": "1", "y": "a"},
        {"x": "2", "y": "a"},
    ]
    assert yaml_sweep.substitute_command("run ~X~", {"X": "3"}) == "run 3"
    yaml_sweep.check_no_unsubstituted("run 3")
    with pytest.raises(AssertionError, match="Unsubstituted"):
        yaml_sweep.check_no_unsubstituted("run ~MISSING~")

    assert yaml_sweep.run_device_queue("0", ["one", "two"], True) == (2, 0)
    return_codes = iter([0, 2, 0, 0])
    monkeypatch.setattr(
        yaml_sweep.subprocess,
        "run",
        lambda command, shell: SimpleNamespace(returncode=next(return_codes)),
    )
    assert yaml_sweep.run_device_queue("0", ["a", "b"], False) == (1, 1)
    assert yaml_sweep.run_device_queue(
        "0", ["c", "d"], False, concurrent_per_device=2
    ) == (2, 0)
    output = capsys.readouterr().out
    assert "Failed with return code 2" in output
    assert "Completed successfully" in output


def _write_sweep(path: Path, config: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(config))


def test_yaml_sweep_main_validation_and_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(AssertionError, match="not found"):
        yaml_sweep.main(str(missing))

    path = tmp_path / "sweep.yaml"
    base_config: dict[str, object] = {
        "commands": ["prepare ~SHARED~", "train ~DEVICE~ ~LOCAL~"],
        "concurrent_per_device": 2,
        "parameters": {
            "all": {"devices": ["0", "1"], "SHARED": ["x", "y"]},
            "gpu0": {"LOCAL": ["a"]},
            "gpu1": {"LOCAL": ["b"]},
        },
    }
    _write_sweep(path, base_config)
    yaml_sweep.main(str(path), dry_run=True)
    output = capsys.readouterr().out
    assert "DRY RUN: Launching 4 jobs across 2 devices" in output
    assert "Total: 4 succeeded, 0 failed" in output

    _write_sweep(path, {**base_config, "concurrent_per_device": 0})
    with pytest.raises(AssertionError, match="concurrent_per_device"):
        yaml_sweep.main(str(path), dry_run=True)

    no_devices = {
        **base_config,
        "parameters": {"all": {"devices": [], "SHARED": ["x"]}},
    }
    _write_sweep(path, no_devices)
    with pytest.raises(AssertionError, match="at least one device"):
        yaml_sweep.main(str(path), dry_run=True)

    _write_sweep(path, base_config)
    monkeypatch.setattr(
        yaml_sweep,
        "run_device_queue",
        lambda device_id, commands, dry_run, concurrent: (len(commands) - 1, 1),
    )
    with pytest.raises(SystemExit, match="1"):
        yaml_sweep.main(str(path))


def test_yaml_sweep_accepts_documented_positional_path(tmp_path: Path) -> None:
    path = tmp_path / "sweep.yaml"
    _write_sweep(
        path,
        {
            "commands": ["train ~DEVICE~"],
            "parameters": {"all": {"devices": ["0"]}},
        },
    )
    result = subprocess.run(
        [sys.executable, "sweeps/run_yaml_sweep.py", str(path), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "DRY RUN: Launching 1 jobs" in result.stdout


def test_only_paper_sweep_configs_are_published() -> None:
    config_names = {path.name for path in (ROOT / "sweeps/configs").glob("*.yaml")}
    assert config_names == PAPER_SWEEP_CONFIGS


def test_paper_sweep_configs_use_published_task_diversity_grids() -> None:
    expected_grids = {
        "lr_task_diversity_sweep.yaml": (
            "NUM_TASKS",
            {str(2**j) for j in range(17)},
        ),
        "bau_task_diversity_sweep.yaml": (
            "NUM_TASKS",
            {str(2**j) for j in range(13)},
        ),
        "markov_task_diversity_sweep.yaml": (
            "N_CHAINS",
            {str(2**j) for j in range(2, 12)},
        ),
    }

    for config_name, (parameter_name, expected_values) in expected_grids.items():
        config_path = ROOT / "sweeps/configs" / config_name
        config = yaml.safe_load(config_path.read_text())
        parameters = config["parameters"]
        actual_values = {
            value
            for group_name, group_parameters in parameters.items()
            if group_name != "all"
            for value in group_parameters.get(parameter_name, [])
        }
        assert actual_values == expected_values, config_path


@pytest.mark.parametrize(
    "config_path",
    sorted((ROOT / "sweeps/configs").glob("*.yaml")),
)
def test_checked_in_sweeps_have_non_overwriting_valid_jobs(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    yaml_sweep.main(str(config_path), dry_run=True)
    lines = [
        line.split("] ", 1)[1]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[GPU ")
    ]
    assert lines

    normalized = [
        re.sub(r"^CUDA_VISIBLE_DEVICES=\S+\s+", "", command) for command in lines
    ]
    assert len(normalized) == len(set(normalized)), config_path

    checkpoint_commands = [
        command for command in lines if "--save-checkpoint" in command
    ]
    checkpoint_roots = re.findall(
        r"--checkpoint-root\s+(\S+)", "\n".join(checkpoint_commands)
    )
    assert len(checkpoint_roots) == len(checkpoint_commands), config_path
    if len(checkpoint_roots) != len(set(checkpoint_roots)):
        assert all("--use-wandb" in command for command in checkpoint_commands)

    for command in checkpoint_commands:
        if "--save-every" in command:
            assert "--checkpoint-schedule linear" in command, command

    for command in lines:
        if "uv run train.py lr" not in command:
            continue
        d_model_match = re.findall(r"--d-model\s+(\d+)", command)
        n_heads_match = re.findall(r"--n-heads\s+(\d+)", command)
        d_head_match = re.findall(r"--d-head\s+(\d+)", command)
        assert len(d_model_match) == len(n_heads_match) == len(d_head_match) == 1
        d_model = int(d_model_match[0])
        n_heads = int(n_heads_match[0])
        d_head = int(d_head_match[0])
        assert d_model == n_heads * d_head, command


@pytest.mark.parametrize(
    ("path", "expected_option"),
    [
        ("scripts/generate_bau_eval_dataset.py", "--output-dir"),
        ("scripts/compute_bau_prior_predictive_kl.py", "--checkpoint-root"),
        ("scripts/compute_lr_prior_delta_mse.py", "--checkpoint-root"),
        ("scripts/compute_markov_prior_predictive_kl.py", "--manifest-csv"),
        ("balls_and_urns/train.py", "--num-tasks"),
        ("balls_and_urns/beta_bernoulli.py", "--prior-alpha"),
    ],
)
def test_direct_script_cli_imports_project(path: str, expected_option: str) -> None:
    result = subprocess.run(
        [sys.executable, path, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert expected_option in result.stdout


@pytest.mark.parametrize(
    "path",
    [
        "linear_regression/sweep_analysis.py",
        "markov/run_pmc.py",
        "sweeps/run_yaml_sweep.py",
        "balls_and_urns/beta_bernoulli.py",
    ],
)
def test_cli_guards_do_not_bypass_tyro(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if path in {
        "linear_regression/sweep_analysis.py",
        "markov/run_pmc.py",
        "balls_and_urns/beta_bernoulli.py",
    }:

        def stop_after_cli(target: object) -> None:
            raise SystemExit(0)

        monkeypatch.setattr(tyro, "cli", stop_after_cli)
        with pytest.raises(SystemExit, match="0"):
            runpy.run_path(path, run_name="__main__")
        return

    monkeypatch.setattr(tyro, "cli", lambda target: None)
    runpy.run_path(path, run_name="__main__")
