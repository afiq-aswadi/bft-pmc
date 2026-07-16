from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from linear_regression.analysis.config import SweepConfig
from linear_regression.analysis.data import (
    PromptData,
    save_prior_dataset,
    save_shared_dataset,
)
import linear_regression.analysis.runner as runner
from linear_regression.priors import DiscretePrior


class RunnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


def _shared_arrays() -> dict[str, np.ndarray]:
    return {
        "prompt_xs": np.ones((2, 2, 2), dtype=np.float32),
        "prompt_ys_generalising": np.ones((2, 2), dtype=np.float32),
        "prompt_ys_random": np.zeros((2, 2), dtype=np.float32),
        "generalising_xs": np.ones((2, 2, 2), dtype=np.float32),
        "generalising_ys": np.ones((2, 2), dtype=np.float32),
        "random_xs": np.zeros((2, 2, 2), dtype=np.float32),
        "random_ys": np.zeros((2, 2), dtype=np.float32),
    }


def _save_shared(path: Path) -> None:
    arrays = _shared_arrays()
    save_shared_dataset(
        path,
        arrays["prompt_xs"],
        arrays["prompt_ys_generalising"],
        arrays["prompt_ys_random"],
        arrays["generalising_xs"],
        arrays["generalising_ys"],
        arrays["random_xs"],
        arrays["random_ys"],
        task_size=2,
        noise_std=0.1,
        eval_batch_size=2,
        seq_len=2,
        max_prompt_length=2,
        max_n_prompts=2,
    )


def test_shared_eval_context_none_existing_and_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runner.load_or_create_shared_eval_context(SweepConfig(), []).data is None

    dataset_dir = tmp_path / "existing"
    _save_shared(dataset_dir / "shared.npz")
    existing_config = SweepConfig(
        eval_dataset_dir=str(dataset_dir),
        eval_batch_size=2,
        seq_len=2,
        noise_std=0.1,
        prompt_lengths=(0, 2),
        n_prompts=(2,),
    )
    existing = runner.load_or_create_shared_eval_context(
        existing_config,
        [],
    )
    assert existing.generalising_batch is not None
    assert existing.random_batch is not None
    invalid_cached_configs = [
        replace(existing_config, noise_std=0.2),
        replace(existing_config, eval_batch_size=3),
        replace(existing_config, seq_len=3),
        replace(existing_config, prompt_lengths=(0, 3)),
        replace(existing_config, n_prompts=(3,)),
    ]
    for invalid_config in invalid_cached_configs:
        with pytest.raises(ValueError, match="Cached shared dataset"):
            runner.load_or_create_shared_eval_context(invalid_config, [])

    created_dir = tmp_path / "created"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint_step_1.pt"
    checkpoint.touch()
    monkeypatch.setattr(runner, "find_latest_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(
        runner,
        "load_run_info",
        lambda path: {"task_size": 2},
    )
    monkeypatch.setattr(
        runner,
        "sample_batch",
        lambda generator, batch_size, seq_len: (
            torch.ones(batch_size, seq_len, 2),
            torch.ones(batch_size, seq_len),
        ),
    )
    created = runner.load_or_create_shared_eval_context(
        SweepConfig(
            eval_dataset_dir=str(created_dir),
            eval_batch_size=2,
            seq_len=2,
            prompt_lengths=(0, 2),
            n_prompts=(2,),
        ),
        [run_dir],
    )
    assert created.data is not None
    assert (created_dir / "shared.npz").exists()

    monkeypatch.setattr(runner, "find_latest_checkpoint", lambda path: None)
    with pytest.raises(AssertionError, match="No checkpoints"):
        runner.load_or_create_shared_eval_context(
            SweepConfig(eval_dataset_dir=str(tmp_path / "missing")),
            [run_dir],
        )


def test_build_run_eval_inputs_with_persisted_and_generated_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "datasets"
    shared_path = dataset_dir / "shared.npz"
    _save_shared(shared_path)
    shared_data = dict(np.load(shared_path))
    context = runner.SharedEvalContext(
        data=shared_data,
        generalising_batch=(torch.ones(2, 2, 2), torch.ones(2, 2)),
        random_batch=(torch.zeros(2, 2, 2), torch.zeros(2, 2)),
    )
    prior = DiscretePrior(2, tasks=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    run_info = {"task_size": 2, "num_tasks": 2}
    config = SweepConfig(
        eval_dataset_dir=str(dataset_dir),
        eval_batch_size=2,
        seq_len=2,
        prompt_lengths=(0, 2),
        n_prompts=(2,),
        compute_distribution_metrics=True,
        include_random_eval=True,
    )

    monkeypatch.setattr(
        runner,
        "sample_batch",
        lambda generator, batch_size, seq_len: (
            torch.ones(batch_size, seq_len, 2),
            torch.ones(batch_size, seq_len),
        ),
    )
    eval_data, prompt_data = runner.build_run_eval_inputs(
        config,
        "new-run",
        run_info,
        prior,
        context,
    )
    assert eval_data is not None and set(eval_data) == {
        "data_memorising",
        "data_generalising",
        "random",
    }
    assert prompt_data is not None
    assert (dataset_dir / "new-run.npz").exists()

    persisted_config = SweepConfig(
        eval_dataset_dir=str(dataset_dir),
        separate_eval_prompts=True,
        eval_n_prompts=1,
        eval_prompt_length=1,
        include_random_eval=False,
    )
    persisted_eval, persisted_prompts = runner.build_run_eval_inputs(
        persisted_config,
        "new-run",
        run_info,
        prior,
        context,
    )
    assert persisted_eval is not None
    assert set(persisted_eval) == {"data_memorising", "data_generalising"}
    assert persisted_prompts is not None

    random_eval, _ = runner.build_run_eval_inputs(
        replace(persisted_config, include_random_eval=True),
        "new-run",
        run_info,
        prior,
        context,
    )
    assert random_eval is not None and "random" in random_eval

    save_prior_dataset(
        dataset_dir / "bad-count.npz",
        np.ones((2, 2)),
        np.ones((2, 2, 2)),
        np.ones((2, 2)),
        3,
        "bad-count",
    )
    with pytest.raises(ValueError, match="task count"):
        runner.build_run_eval_inputs(
            config,
            "bad-count",
            run_info,
            prior,
            context,
        )
    save_prior_dataset(
        dataset_dir / "bad-id.npz",
        np.ones((2, 2)),
        np.ones((2, 2, 2)),
        np.ones((2, 2)),
        2,
        "other-run",
    )
    with pytest.raises(ValueError, match="belongs to"):
        runner.build_run_eval_inputs(
            config,
            "bad-id",
            run_info,
            prior,
            context,
        )

    save_prior_dataset(
        dataset_dir / "short.npz",
        np.ones((2, 2)),
        np.ones((2, 2, 2)),
        np.ones((2, 2)),
        2,
        "short",
    )
    with pytest.raises(AssertionError):
        runner.build_run_eval_inputs(
            SweepConfig(eval_dataset_dir=str(dataset_dir), eval_position=3),
            "short",
            run_info,
            prior,
            context,
        )


def test_build_run_eval_inputs_without_saved_datasets() -> None:
    prior = DiscretePrior(2, tasks=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    context = runner.SharedEvalContext(data=None)
    config = SweepConfig(
        compute_distribution_metrics=True,
        separate_eval_prompts=True,
        include_random_eval=True,
        prompt_lengths=(0, 2),
        n_prompts=(2,),
        eval_n_prompts=1,
        eval_prompt_length=1,
    )
    eval_data, prompt_data = runner.build_run_eval_inputs(
        config,
        "run",
        {"task_size": 2, "num_tasks": 2},
        prior,
        context,
    )
    assert eval_data is not None and "random" in eval_data
    assert prompt_data is not None

    no_data, no_prompts = runner.build_run_eval_inputs(
        SweepConfig(compute_distribution_metrics=False, separate_eval_prompts=False),
        "run",
        {"task_size": 2, "num_tasks": 2},
        prior,
        context,
    )
    assert no_data is None
    assert no_prompts is None


def test_sweep_runner_orchestration_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint root"):
        runner.run_analysis(
            SweepConfig(checkpoint_root=str(tmp_path / "missing")),
            tmp_path / "out",
        )

    checkpoint_root = tmp_path / "checkpoints"
    skipped_run = checkpoint_root / "skip"
    active_run = checkpoint_root / "active"
    skipped_run.mkdir(parents=True)
    active_run.mkdir()
    checkpoint = active_run / "checkpoint_step_7.pt"
    checkpoint.touch()
    model = RunnerModel()
    prompt_data = PromptData(
        xs=np.ones((1, 2, 2), dtype=np.float32),
        ys_gaussian=np.ones((1, 2), dtype=np.float32),
        ys_discrete=np.ones((1, 2), dtype=np.float32),
        ys_random=np.ones((1, 2), dtype=np.float32),
    )

    monkeypatch.setattr(runner, "SupervisedPFN", nn.Module)
    monkeypatch.setattr(
        runner,
        "find_latest_checkpoint",
        lambda path: None if path.name == "skip" else checkpoint,
    )
    monkeypatch.setattr(
        runner,
        "load_run_info",
        lambda path: {
            "model": model,
            "task_size": 2,
            "tasks": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "num_tasks": 2,
        },
    )
    monkeypatch.setattr(
        runner,
        "load_or_create_shared_eval_context",
        lambda config, run_dirs: runner.SharedEvalContext(data=None),
    )
    monkeypatch.setattr(
        runner,
        "build_run_eval_inputs",
        lambda *args: ({}, prompt_data),
    )
    monkeypatch.setattr(
        runner,
        "compute_all_predictive_metrics",
        lambda *args, **kwargs: {"model_mse": 0.1},
    )
    monkeypatch.setattr(
        runner,
        "prepare_model_for_long_rollout",
        lambda *args, **kwargs: 3,
    )

    def fake_distribution(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, float], dict[str, np.ndarray], list[dict[str, float]]]:
        del args, kwargs
        metrics = {"dist/ed_vs_baseline_memorising": 0.1}
        return metrics, {"pt": np.zeros((2, 2))}, [metrics]

    monkeypatch.setattr(
        runner, "compute_distribution_metrics_single", fake_distribution
    )
    config = SweepConfig(
        checkpoint_root=str(checkpoint_root),
        prompt_sources=("discrete", "gaussian", "random"),
        prompt_lengths=(0, 2),
        n_samples=(2,),
        n_samples_prior=(2,),
        n_prompts=(1,),
        predictive_steps=3,
    )
    metrics, per_prompt = runner.run_analysis(config, tmp_path / "out")
    assert len(metrics) == 4
    assert len(per_prompt) == 4
    assert set(metrics["prompt_source"]) == {
        "N/A",
        "memorising",
        "generalising",
        "random",
    }

    no_distribution, empty_per_prompt = runner.run_analysis(
        SweepConfig(
            checkpoint_root=str(checkpoint_root),
            compute_distribution_metrics=False,
        ),
        tmp_path / "out_no_distribution",
    )
    assert len(no_distribution) == 1
    assert empty_per_prompt.empty

    monkeypatch.setattr(runner, "find_latest_checkpoint", lambda path: None)
    with pytest.raises(RuntimeError, match="No results"):
        runner.run_analysis(config, tmp_path / "out_empty")
