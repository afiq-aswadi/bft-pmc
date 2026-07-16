from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import NoReturn

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import tyro

from balls_and_urns.dataset import (
    load_generalising_dataset,
    load_memorising_dataset,
)
from linear_regression.priors import DiscretePrior
from pfn_transformerlens.model.PFN import DistributionPrediction
import scripts.compute_bau_prior_predictive_kl as bau_prior
import scripts.compute_lr_prior_delta_mse as lr_prior
import scripts.compute_markov_prior_predictive_kl as markov_prior
import scripts.generate_bau_eval_dataset as bau_dataset


class LRPredictionModel(nn.Module):
    def __init__(self, tuple_result: bool = False) -> None:
        super().__init__()
        self.tuple_result = tuple_result

    def predict_on_prompt(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> DistributionPrediction | tuple[DistributionPrediction, None]:
        prediction = DistributionPrediction(
            probs=torch.full((*x.shape[:2], 2), 0.5),
            y_grid=torch.tensor([-1.0, 1.0]),
        )
        return (prediction, None) if self.tuple_result else prediction


class BAUPredictionModel(nn.Module):
    def __init__(self, tuple_result: bool = False) -> None:
        super().__init__()
        self.tuple_result = tuple_result

    def predict_on_prompt(
        self,
        tokens: torch.Tensor,
    ) -> DistributionPrediction | tuple[DistributionPrediction, None]:
        prediction = DistributionPrediction(
            probs=torch.full((*tokens.shape, 2), 0.5),
            y_grid=torch.arange(2).float(),
        )
        return (prediction, None) if self.tuple_result else prediction


@pytest.mark.parametrize("module", [lr_prior, bau_prior, markov_prior])
def test_sidecar_device_resolution(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve = module._resolve_device
    assert resolve("cpu") == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve(None) == torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    if module is markov_prior or module is lr_prior or module is bau_prior:
        assert resolve(None) == torch.device("mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve(None) == torch.device("cpu")


def test_lr_prior_delta_metric_and_config_validation(tmp_path: Path) -> None:
    prior = DiscretePrior(1, tasks=torch.tensor([[-1.0], [1.0]]))
    metrics = lr_prior._prior_delta_mse(
        model=LRPredictionModel(tuple_result=True),
        prior=prior,
        task_size=1,
        noise_variance=0.25,
        batch_size=4,
        device=torch.device("cpu"),
        seed=1,
    )
    assert metrics.keys() == {
        "model_mse_self",
        "baseline_memorising_mse_self",
        "delta_vs_memorising",
        "delta_vs_generalising",
    }

    with pytest.raises(FileNotFoundError):
        lr_prior.PriorDeltaConfig(checkpoint_root=tmp_path / "missing").validate()
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="noise_std"):
        lr_prior.PriorDeltaConfig(checkpoint_root=tmp_path, noise_std=-1).validate()
    with pytest.raises(ValueError, match="batch_size"):
        lr_prior.PriorDeltaConfig(checkpoint_root=tmp_path, batch_size=0).validate()


def test_lr_prior_delta_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    run_dir = checkpoint_root / "run"
    skip_dir = checkpoint_root / "skip"
    run_dir.mkdir(parents=True)
    skip_dir.mkdir()
    checkpoint = run_dir / "checkpoint_step_1.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        lr_prior,
        "find_latest_checkpoint",
        lambda path: checkpoint if path.name == "run" else None,
    )
    monkeypatch.setattr(
        lr_prior,
        "load_run_info",
        lambda path, device: {
            "model": LRPredictionModel(),
            "tasks": torch.tensor([[-1.0], [1.0]]),
            "task_size": 1,
            "num_tasks": 2,
        },
    )
    monkeypatch.setattr(lr_prior, "SupervisedPFN", nn.Module)
    monkeypatch.setattr(
        lr_prior,
        "_prior_delta_mse",
        lambda **kwargs: {
            "model_mse_self": 0.1,
            "baseline_memorising_mse_self": 0.2,
            "delta_vs_memorising": 0.3,
            "delta_vs_generalising": 0.4,
        },
    )
    config = lr_prior.PriorDeltaConfig(
        checkpoint_root=checkpoint_root,
        out_csv=tmp_path / "results/lr.csv",
        device="cpu",
    )
    lr_prior.main(config)
    assert pd.read_csv(config.out_csv)["num_tasks"].tolist() == [2]

    monkeypatch.setattr(lr_prior, "find_latest_checkpoint", lambda path: None)
    with pytest.raises(RuntimeError, match="No linear-regression"):
        lr_prior.main(replace(config, out_csv=tmp_path / "empty.csv"))


def test_bau_prior_kl_metric_config_and_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kl_mem, kl_gen = bau_prior._prior_predictive_kl(
        model=BAUPredictionModel(tuple_result=True),
        thetas=torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
        alpha=torch.ones(2),
        bos_token=2,
        device=torch.device("cpu"),
    )
    assert kl_mem == pytest.approx(0.0)
    assert kl_gen == pytest.approx(0.0)

    with pytest.raises(FileNotFoundError):
        bau_prior.PriorKLConfig(checkpoint_root=tmp_path / "missing").validate()
    checkpoint_root = tmp_path / "checkpoints"
    run_dir = checkpoint_root / "run"
    skip_dir = checkpoint_root / "skip"
    run_dir.mkdir(parents=True)
    skip_dir.mkdir()
    with pytest.raises(ValueError, match="alpha_value"):
        bau_prior.PriorKLConfig(
            checkpoint_root=checkpoint_root, alpha_value=0
        ).validate()

    checkpoint = run_dir / "checkpoint_step_1.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        bau_prior,
        "find_latest_checkpoint",
        lambda path: checkpoint if path.name == "run" else None,
    )
    monkeypatch.setattr(
        bau_prior,
        "load_run_info",
        lambda path, device: {
            "model": BAUPredictionModel(),
            "tasks": torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
            "num_tasks": 2,
        },
    )
    monkeypatch.setattr(bau_prior, "UnsupervisedPFN", nn.Module)
    config = bau_prior.PriorKLConfig(
        checkpoint_root=checkpoint_root,
        out_csv=tmp_path / "results/bau.csv",
        device="cpu",
    )
    bau_prior.main(config)
    assert pd.read_csv(config.out_csv)["num_tasks"].tolist() == [2]

    monkeypatch.setattr(bau_prior, "find_latest_checkpoint", lambda path: None)
    with pytest.raises(RuntimeError, match="No BAU"):
        bau_prior.main(replace(config, out_csv=tmp_path / "empty.csv"))


class UniformMarkovModel(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*tokens.shape, 3)


def test_markov_prior_path_and_matrix_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    explicit_config = tmp_path / "config.yaml"
    explicit_config.touch()
    explicit_checkpoint = tmp_path / "model.pt"
    explicit_checkpoint.touch()
    row = pd.Series(
        {
            "config_path": str(explicit_config),
            "checkpoint_path": str(explicit_checkpoint),
        }
    )
    assert markov_prior._resolve_config_path(row, output_dir) == explicit_config
    assert markov_prior._resolve_checkpoint_path(row, output_dir) == explicit_checkpoint

    empty_row = pd.Series(dtype=object)
    sibling = output_dir / "resolved_config.yaml"
    sibling.touch()
    assert markov_prior._resolve_config_path(empty_row, output_dir) == sibling
    sibling.unlink()

    monkeypatch.chdir(tmp_path)
    fallback_config = tmp_path / "outputs/markov/training/run/resolved_config.yaml"
    fallback_config.parent.mkdir(parents=True)
    fallback_config.touch()
    assert markov_prior._resolve_config_path(empty_row, Path("elsewhere/run")) == Path(
        "outputs/markov/training/run/resolved_config.yaml"
    )
    fallback_config.unlink()
    with pytest.raises(FileNotFoundError, match="resolved_config"):
        markov_prior._resolve_config_path(empty_row, Path("elsewhere/run"))

    fallback_checkpoint = (
        tmp_path / "checkpoints/markov/task_diversity/run/checkpoint_step_100000.pt"
    )
    fallback_checkpoint.parent.mkdir(parents=True)
    fallback_checkpoint.touch()
    assert markov_prior._resolve_checkpoint_path(
        empty_row, Path("elsewhere/run")
    ) == Path("checkpoints/markov/task_diversity/run/checkpoint_step_100000.pt")
    fallback_checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="final checkpoint"):
        markov_prior._resolve_checkpoint_path(empty_row, Path("elsewhere/run"))

    direct = output_dir / "transition_matrices.npy"
    matrices = np.full((2, 2, 2), 0.5, dtype=np.float32)
    np.save(direct, matrices)
    assert markov_prior._resolve_training_matrices(output_dir) == direct
    loaded = markov_prior._load_training_matrices(direct, torch.device("cpu"))
    assert loaded.shape == (2, 2, 2)
    direct.unlink()

    fallback_matrix = tmp_path / "outputs/markov/training/run/transition_matrices.npy"
    np.save(fallback_matrix, matrices)
    assert markov_prior._resolve_training_matrices(Path("elsewhere/run")) == Path(
        "outputs/markov/training/run/transition_matrices.npy"
    )
    fallback_matrix.unlink()

    bundle = output_dir / "pmc_eval_bundle.npz"
    np.savez(bundle, training_matrices=matrices)
    assert markov_prior._resolve_training_matrices(output_dir) == bundle
    assert markov_prior._load_training_matrices(bundle, torch.device("cpu")).shape == (
        2,
        2,
        2,
    )
    bundle.unlink()
    with pytest.raises(FileNotFoundError, match="training_matrices"):
        markov_prior._resolve_training_matrices(output_dir)


def test_markov_prior_metric_config_and_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrices = torch.full((2, 2, 2), 0.5)
    kl_mem, kl_gen = markov_prior._first_step_kl(
        model=UniformMarkovModel(),
        training_matrices=matrices,
        k=2,
        bos_token_id=2,
        device=torch.device("cpu"),
    )
    assert kl_mem == pytest.approx(0.0)
    assert kl_gen == pytest.approx(0.0)

    with pytest.raises(FileNotFoundError):
        markov_prior.PriorKLConfig(manifest_csv=tmp_path / "missing.csv").validate()

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = run_dir / "resolved_config.yaml"
    config_path.touch()
    checkpoint = run_dir / "model.pt"
    checkpoint.touch()
    np.save(run_dir / "transition_matrices.npy", np.full((2, 2, 2), 0.5))
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "run_name": "run",
                "n_chains": 2,
                "output_dir": str(run_dir),
                "config_path": str(config_path),
                "checkpoint_path": str(checkpoint),
            }
        ]
    ).to_csv(manifest_path, index=False)
    markov_config = SimpleNamespace(
        k=2,
        d_model=4,
        seq_len=4,
        num_layers=1,
        num_heads=2,
        expansion_factor=2,
        rope_theta=10_000.0,
    )
    monkeypatch.setattr(markov_prior, "load_config", lambda path: markov_config)
    monkeypatch.setattr(
        markov_prior, "MarkovTransformer", lambda **kwargs: UniformMarkovModel()
    )
    monkeypatch.setattr(markov_prior, "load_markov_state_dict", lambda path, device: {})
    monkeypatch.setattr(markov_prior, "_first_step_kl", lambda **kwargs: (0.1, 0.2))
    config = markov_prior.PriorKLConfig(
        manifest_csv=manifest_path,
        out_csv=tmp_path / "results/markov.csv",
        device="cpu",
    )
    markov_prior.main(config)
    assert pd.read_csv(config.out_csv)["n_chains"].tolist() == [2]

    np.save(run_dir / "transition_matrices.npy", np.full((1, 2, 2), 0.5))
    with pytest.raises(ValueError, match="manifest says"):
        markov_prior.main(replace(config, out_csv=tmp_path / "mismatch.csv"))

    pd.DataFrame(columns=["run_name", "n_chains", "output_dir"]).to_csv(
        manifest_path, index=False
    )
    with pytest.raises(RuntimeError, match="contains no runs"):
        markov_prior.main(replace(config, out_csv=tmp_path / "empty.csv"))


def _fake_run_info() -> dict[str, object]:
    return {
        "tasks": torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
        "num_tasks": 2,
    }


def _patch_bau_dataset_sources(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: Path,
) -> None:
    monkeypatch.setattr(bau_dataset, "find_latest_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(
        bau_dataset, "load_run_info", lambda path, device: _fake_run_info()
    )
    monkeypatch.setattr(bau_dataset, "make_bau_generator", lambda alpha: object())
    monkeypatch.setattr(bau_dataset, "make_generator_from_pool", lambda pool: object())
    monkeypatch.setattr(
        bau_dataset,
        "sample_batch",
        lambda generator, batch_size, seq_len: (
            None,
            torch.arange(batch_size * seq_len).reshape(batch_size, seq_len) % 2,
        ),
    )


def test_bau_dataset_validation_cache_and_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(FileNotFoundError):
        bau_dataset.DatasetConfig(tmp_path / "out", tmp_path / "missing").validate()
    checkpoint_root = tmp_path / "checkpoints"
    run_dir = checkpoint_root / "run"
    run_dir.mkdir(parents=True)
    for config in [
        bau_dataset.DatasetConfig(tmp_path / "out", checkpoint_root, alpha_value=0),
        bau_dataset.DatasetConfig(tmp_path / "out", checkpoint_root, seq_len=0),
        bau_dataset.DatasetConfig(tmp_path / "out", checkpoint_root, batch_size=0),
    ]:
        with pytest.raises(ValueError):
            config.validate()

    bau_dataset._check_cache_compat(
        {"seq_len": 2}, {"seq_len": 2}, tmp_path / "cache.npz", ("seq_len",)
    )
    with pytest.raises(AssertionError, match="different settings"):
        bau_dataset._check_cache_compat(
            {"seq_len": 1},
            {"seq_len": 2},
            tmp_path / "cache.npz",
            ("seq_len",),
        )

    checkpoint = run_dir / "checkpoint_step_1.pt"
    checkpoint.touch()
    _patch_bau_dataset_sources(monkeypatch, checkpoint)
    config = bau_dataset.DatasetConfig(
        output_dir=tmp_path / "data",
        checkpoint_root=checkpoint_root,
        seq_len=2,
        batch_size=3,
    )
    bau_dataset.main(config)
    shared = load_generalising_dataset(config.output_dir / "shared.npz")
    memorising = load_memorising_dataset(config.output_dir / "run.npz")
    assert shared["generalising_tokens"].shape == (3, 2)
    assert memorising["memorising_tokens"].shape == (3, 2)
    assert (config.output_dir / "metadata.json").exists()

    bau_dataset.main(config)
    with pytest.raises(AssertionError, match="alpha"):
        bau_dataset.main(replace(config, alpha_value=2.0))

    monkeypatch.setattr(
        bau_dataset,
        "load_memorising_dataset",
        lambda path: {"num_tasks": 3, "memorising_tokens": np.zeros((3, 2))},
    )
    with pytest.raises(AssertionError, match="num_tasks"):
        bau_dataset.main(config)
    monkeypatch.setattr(
        bau_dataset,
        "load_memorising_dataset",
        lambda path: {"num_tasks": 2, "memorising_tokens": np.zeros((1, 2))},
    )
    with pytest.raises(AssertionError, match="shape"):
        bau_dataset.main(config)

    bau_dataset.main(replace(config, overwrite=True))


def test_bau_dataset_requires_checkpoint_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    (checkpoint_root / "empty").mkdir(parents=True)
    monkeypatch.setattr(bau_dataset, "find_latest_checkpoint", lambda path: None)
    with pytest.raises(AssertionError, match="No checkpoint-containing"):
        bau_dataset.main(bau_dataset.DatasetConfig(tmp_path / "data", checkpoint_root))


def _stop(*args: object, **kwargs: object) -> NoReturn:
    raise SystemExit("coverage stop")


@pytest.mark.parametrize(
    "path",
    [
        "scripts/compute_lr_prior_delta_mse.py",
        "scripts/compute_bau_prior_predictive_kl.py",
        "scripts/compute_markov_prior_predictive_kl.py",
        "scripts/generate_bau_eval_dataset.py",
    ],
)
def test_public_data_script_guards(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tyro, "cli", _stop)
    with pytest.raises(SystemExit, match="coverage stop"):
        runpy.run_path(path, run_name="__main__")
