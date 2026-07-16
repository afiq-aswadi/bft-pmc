from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from markov.config import MarkovConfig, dump_config
import markov.task_diversity_threshold as threshold
import markov.train as markov_train


class FakeWandbRun:
    def __init__(self) -> None:
        self.logs: list[dict[str, object]] = []
        self.finished = False

    def log(self, data: dict[str, object]) -> None:
        self.logs.append(data)

    def finish(self) -> None:
        self.finished = True


def _training_config(max_steps: int = 3) -> MarkovConfig:
    return MarkovConfig(
        k=2,
        seq_len=4,
        n_chains=2,
        batch_size=2,
        learning_rate=0.001,
        eval_interval=3,
        max_steps=max_steps,
        d_model=4,
        num_layers=1,
        num_heads=2,
        expansion_factor=2,
        context_len=3,
        num_eval_trials=1,
        delta_eval_batch_size=1,
        seed=1,
    )


@pytest.mark.parametrize(
    "args",
    [
        markov_train.TrainConfig(pmc_num_samples=0),
        markov_train.TrainConfig(pmc_prompt_len=-1),
        markov_train.TrainConfig(pmc_generation_length=1),
        markov_train.TrainConfig(run_name_suffix="bad/name"),
    ],
)
def test_transient_cli_validation(args: markov_train.TrainConfig) -> None:
    with pytest.raises(ValueError):
        args.validate()


def test_markov_train_config_name_wandb_and_checkpoint_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    dump_config(_training_config(), config_path)
    args = markov_train.TrainConfig(
        config_path=str(config_path),
        n_chains=3,
        max_steps=2,
    )
    resolved = markov_train.resolve_config(args)
    assert resolved.n_chains == 3
    assert resolved.max_steps == 2

    monkeypatch.setattr(
        markov_train,
        "datetime",
        SimpleNamespace(
            now=lambda: SimpleNamespace(strftime=lambda pattern: "20260101_000000")
        ),
    )
    assert markov_train.build_run_name(config=_training_config()).endswith(
        "20260101_000000"
    )
    assert (
        markov_train.initialize_wandb(
            enabled=False,
            project="project",
            run_name="run",
            config=_training_config(),
        )
        is None
    )

    fake_run = FakeWandbRun()
    fake_module = SimpleNamespace(
        init=lambda **kwargs: fake_run, Image=lambda path: path
    )
    monkeypatch.setattr(markov_train, "import_module", lambda name: fake_module)
    assert (
        markov_train.initialize_wandb(
            enabled=True,
            project="project",
            run_name="run",
            config=_training_config(),
        )
        is fake_run
    )
    assert markov_train._wandb_image(tmp_path / "figure.png") == str(
        tmp_path / "figure.png"
    )

    model = markov_train.MarkovTransformer(3, 4, 4, 1, 2, 2, 10_000.0)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = markov_train.save_checkpoint(
        model=model,
        optimizer=optimizer,
        config=_training_config(),
        step=1,
        checkpoint_dir=tmp_path,
    )
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["step"] == 1


def test_markov_training_job_untracked_and_tracked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delta_values = {
        "wellspec_generalising": 0.1,
        "wellspec_memorising": 0.2,
        "misspec_generalising": 0.3,
        "misspec_memorising": 0.4,
    }
    monkeypatch.setattr(
        markov_train,
        "evaluate_kl",
        lambda **kwargs: 0.2 if kwargs["is_ood"] else 0.1,
    )
    monkeypatch.setattr(
        markov_train,
        "evaluate_baseline_deltas",
        lambda **kwargs: delta_values,
    )
    monkeypatch.setattr(
        markov_train,
        "resolve_pmc_sampling_config",
        lambda config, seq_len: replace(config, prompt_len=0, generation_length=2),
    )

    def save_pmc(*, output_dir: Path, **kwargs: object) -> None:
        del kwargs
        pd.DataFrame({"sample": [1]}).to_csv(
            output_dir / "pmc_samples.npz", index=False
        )

    def save_plot(*, save_path: Path, **kwargs: object) -> None:
        del kwargs
        save_path.touch()

    monkeypatch.setattr(markov_train, "generate_and_save_pmc_artifacts", save_pmc)
    monkeypatch.setattr(markov_train, "plot_transient", save_plot)
    monkeypatch.setattr(markov_train, "build_run_name", lambda config: "generated")

    untracked = markov_train.run_training_job(
        config=_training_config(),
        checkpoint_root=tmp_path / "checkpoints",
        output_root=tmp_path / "outputs",
        pmc_sampling=None,
        device="cpu",
    )
    assert untracked.run_name == "generated"
    assert untracked.latest_checkpoint_path is not None
    assert untracked.latest_checkpoint_path.name == "checkpoint_step_000003.pt"
    assert len(pd.read_csv(untracked.csv_path)) == 2
    assert untracked.final_ood_kl == pytest.approx(0.2)

    fake_run = FakeWandbRun()
    monkeypatch.setattr(markov_train, "initialize_wandb", lambda **kwargs: fake_run)
    monkeypatch.setattr(markov_train, "_wandb_image", lambda path: str(path))
    tracked = markov_train.run_training_job(
        config=_training_config(max_steps=1),
        checkpoint_root=tmp_path / "checkpoints",
        output_root=tmp_path / "outputs",
        run_name="tracked",
        run_name_suffix="retry",
        pmc_sampling=markov_train.PMCSamplingConfig(2, 0, 2, 0),
        use_wandb=True,
        device="cpu",
    )
    assert tracked.run_name == "tracked-retry"
    assert fake_run.finished
    assert len(fake_run.logs) == 2
    assert "kl/id/wellspec/generalising" in fake_run.logs[0]

    monkeypatch.setattr(markov_train, "plot_transient", lambda **kwargs: None)
    with pytest.raises(FileNotFoundError, match="Transient figure"):
        markov_train.run_training_job(
            config=_training_config(max_steps=1),
            checkpoint_root=tmp_path / "checkpoints",
            output_root=tmp_path / "outputs",
            run_name="missing-figure",
            pmc_sampling=markov_train.PMCSamplingConfig(2, 0, 2, 0),
            use_wandb=True,
            device="cpu",
        )


def test_markov_train_main_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _training_config(max_steps=1)
    monkeypatch.setattr(markov_train, "resolve_config", lambda args: config)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        markov_train, "run_training_job", lambda **kwargs: calls.append(kwargs)
    )
    args = markov_train.TrainConfig(
        run_name="run",
        pmc_num_samples=2,
        pmc_prompt_len=0,
        pmc_generation_length=2,
    )
    markov_train.main(args)
    assert calls[0]["config"] == config
    assert calls[0]["run_name"] == "run"


@pytest.mark.parametrize(
    "config",
    [
        threshold.ThresholdExperimentConfig([]),
        threshold.ThresholdExperimentConfig([0]),
        threshold.ThresholdExperimentConfig([1], max_steps=0),
        threshold.ThresholdExperimentConfig([1], eval_interval=0),
        threshold.ThresholdExperimentConfig([1], gap_tolerance=-1),
    ],
)
def test_threshold_config_validation(
    config: threshold.ThresholdExperimentConfig,
) -> None:
    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "args",
    [
        threshold.ThresholdConfig(experiment_name=""),
        threshold.ThresholdConfig(pmc_num_samples=0),
        threshold.ThresholdConfig(pmc_prompt_len=-1),
        threshold.ThresholdConfig(pmc_generation_length=1),
    ],
)
def test_threshold_cli_validation(args: threshold.ThresholdConfig) -> None:
    with pytest.raises(ValueError):
        args.validate()


def test_threshold_config_io_estimation_and_rows(tmp_path: Path) -> None:
    assert not threshold.ThresholdConfig().use_wandb
    config_path = tmp_path / "threshold.yaml"
    config_path.write_text(
        "n_chains_values: [8, 2]\nmax_steps: 3\ngap_tolerance: 0.1\n",
        encoding="utf-8",
    )
    config = threshold.load_threshold_config(config_path)
    assert config.n_chains_values == [8, 2]

    list_path = tmp_path / "list.yaml"
    list_path.write_text("- bad\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Expected a mapping"):
        threshold.load_threshold_config(list_path)
    unknown_path = tmp_path / "unknown.yaml"
    unknown_path.write_text("n_chains_values: [1]\nunknown: 1\n", encoding="utf-8")
    with pytest.raises(KeyError, match="Unknown"):
        threshold.load_threshold_config(unknown_path)

    rows = [
        {"n_chains": 2, "generalization_gap": 0.2},
        {"n_chains": 8, "generalization_gap": 0.05},
    ]
    assert threshold.estimate_threshold(rows, 0.1) == 8
    assert threshold.estimate_threshold(rows, 0.01) is None

    result = markov_train.TrainingRunResult(
        run_name="run",
        checkpoint_dir=tmp_path / "checkpoints",
        output_dir=tmp_path / "output",
        csv_path=tmp_path / "history.csv",
        figure_path=tmp_path / "figure.png",
        final_model_path=tmp_path / "model.pt",
        pmc_samples_path=None,
        latest_checkpoint_path=None,
        final_train_loss=1.0,
        final_id_kl=0.1,
        final_ood_kl=0.2,
    )
    row = threshold.row_from_result(2, result)
    assert row["latest_checkpoint_path"] == ""
    assert row["pmc_samples_path"] == ""
    summary_path = tmp_path / "summary.csv"
    threshold.write_summary_csv([row], summary_path)
    assert pd.read_csv(summary_path).iloc[0]["n_chains"] == 2


def test_threshold_main_threshold_found_and_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = _training_config(max_steps=1)
    threshold_config = threshold.ThresholdExperimentConfig(
        n_chains_values=[4, 2, 4],
        max_steps=1,
        eval_interval=1,
        gap_tolerance=0.1,
    )
    monkeypatch.setattr(threshold, "load_config", lambda path: base_config)
    monkeypatch.setattr(
        threshold,
        "load_threshold_config",
        lambda path: threshold_config,
    )

    def fake_training(
        *, config: MarkovConfig, run_name: str, **kwargs: object
    ) -> markov_train.TrainingRunResult:
        del kwargs
        output_dir = tmp_path / run_name
        output_dir.mkdir(exist_ok=True)
        history_path = output_dir / "history.csv"
        pd.DataFrame({"step": [1], "id_kl": [0.1], "ood_kl": [0.15]}).to_csv(
            history_path,
            index=False,
        )
        return markov_train.TrainingRunResult(
            run_name=run_name,
            checkpoint_dir=output_dir,
            output_dir=output_dir,
            csv_path=history_path,
            figure_path=output_dir / "figure.png",
            final_model_path=output_dir / "model.pt",
            pmc_samples_path=output_dir / "pmc.npz",
            latest_checkpoint_path=output_dir / "checkpoint.pt",
            final_train_loss=1.0,
            final_id_kl=0.1,
            final_ood_kl=0.15 if config.n_chains == 2 else 0.4,
        )

    monkeypatch.setattr(threshold, "run_training_job", fake_training)
    plot_calls: list[Path] = []
    monkeypatch.setattr(
        threshold,
        "plot_task_diversity",
        lambda csv_path, save_path, **kwargs: plot_calls.append(Path(save_path)),
    )
    monkeypatch.setattr(
        threshold,
        "plot_task_diversity_heatmap",
        lambda csv_path, save_path, **kwargs: plot_calls.append(Path(save_path)),
    )
    args = threshold.ThresholdConfig(
        output_root=str(tmp_path / "outputs"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        experiment_name="threshold",
        max_steps=1,
        eval_interval=1,
        pmc_num_samples=2,
        pmc_prompt_len=0,
        pmc_generation_length=2,
    )
    threshold.main(args)
    report = tmp_path / "outputs/threshold/threshold_report.txt"
    assert "Estimated threshold" in report.read_text(encoding="utf-8")
    assert len(plot_calls) == 2

    monkeypatch.setattr(
        threshold,
        "load_threshold_config",
        lambda path: replace(threshold_config, gap_tolerance=0.001),
    )
    missing_args = replace(
        args, experiment_name="missing", max_steps=None, eval_interval=None
    )
    threshold.main(missing_args)
    missing_report = tmp_path / "outputs/missing/threshold_report.txt"
    assert "No threshold found" in missing_report.read_text(encoding="utf-8")
