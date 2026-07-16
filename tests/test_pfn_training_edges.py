from __future__ import annotations

import json
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pfn_transformerlens.model.bucketizer import Bucketizer
from pfn_transformerlens.model.configs import SupervisedRegressionPFNConfig
from pfn_transformerlens.train import TrainingConfig

pfn_train = importlib.import_module("pfn_transformerlens.train")


def _model_config() -> SupervisedRegressionPFNConfig:
    return SupervisedRegressionPFNConfig(
        n_layers=1,
        d_model=8,
        d_head=4,
        n_heads=2,
        d_mlp=8,
        n_ctx=8,
        d_vocab=4,
        input_dim=1,
        prediction_type="point",
        act_fn="gelu",
        device="cpu",
    )


class TaskPrior:
    def __init__(self) -> None:
        self.tasks = torch.tensor([[1.0], [2.0]])
        self.task_size = 1
        self.num_tasks = 2


class TinyRegressionGenerator:
    def __init__(self) -> None:
        self.prior = TaskPrior()

    def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.linspace(-1.0, 1.0, seq_len).unsqueeze(-1)
        return x, x[:, 0]


class LossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            input_type="discrete",
            prediction_type="point",
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*y.shape, 1) + self.anchor


def test_training_device_and_loss_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    config = TrainingConfig(device="auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert config.get_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert config.get_device() == "cpu"

    bucketizer = Bucketizer(
        bucket_type="uniform",
        bucket_support="bounded",
        d_vocab=2,
        y_min=-1.0,
        y_max=1.0,
    )
    loss, metrics = pfn_train._compute_distributional_nll(
        torch.zeros(2, 2, 2),
        torch.zeros(2, 2),
        bucketizer,
        compute_mse=True,
    )
    assert loss.ndim == 0 and "mse" in metrics

    point_loss, point_metrics = pfn_train.compute_unsupervised_loss(
        LossModel(),
        torch.ones(2, 3),
    )
    assert point_loss.ndim == 0
    assert point_metrics["loss_type"] == "MSE"


def test_evaluate_model_accumulates_supervised_and_unsupervised_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = LossModel()
    batches = iter(
        [
            (torch.ones(2, 2, 1), torch.ones(2, 2)),
            (None, torch.ones(2, 2)),
        ]
    )
    monkeypatch.setattr(pfn_train, "sample_batch", lambda *args: next(batches))
    monkeypatch.setattr(
        pfn_train,
        "compute_loss",
        lambda model, x, y, **kwargs: (
            torch.tensor(1.0),
            {"loss": 2.0, "loss_type": "MSE"},
        ),
    )
    metrics = pfn_train.evaluate_model(
        model,
        TinyRegressionGenerator(),
        eval_batches=2,
        seq_len=2,
        batch_size=2,
        device="cpu",
    )
    assert metrics == {"loss": 2.0, "loss_type": "MSE"}
    assert model.training


def _training_config(tmp_path: Path, **overrides: object) -> TrainingConfig:
    values: dict[str, object] = {
        "batch_size": 2,
        "seq_len": 2,
        "num_steps": 1,
        "learning_rate": 1e-3,
        "use_warmup": False,
        "use_grad_clip": True,
        "log_every": 1,
        "save_checkpoint": True,
        "save_every": 1,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "eval_every": 1,
        "eval_batches": 1,
        "device": "cpu",
        "log_file": str(tmp_path / "metrics.json"),
        "seed": 1,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def test_training_checkpoint_evaluation_and_log_paths(tmp_path: Path) -> None:
    generator = TinyRegressionGenerator()
    config = _training_config(tmp_path)
    pfn_train.train(
        generator,
        _model_config(),
        config,
        eval_data_generator=TinyRegressionGenerator(),
    )
    checkpoint = tmp_path / "checkpoints/checkpoint_step_1.pt"
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["task_distribution"]["task_size"] == 1
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "eval/loss" in metrics[0]

    separate_log = tmp_path / "separate_metrics.json"
    pfn_train.train(
        generator,
        _model_config(),
        _training_config(
            tmp_path / "second",
            log_every=2,
            save_checkpoint=False,
            log_file=str(separate_log),
        ),
    )
    separate_metrics = json.loads(separate_log.read_text(encoding="utf-8"))
    assert separate_metrics[0]["step"] == 1


class FakeScaler:
    def __init__(self) -> None:
        self.scaled = False
        self.unscaled = False
        self.updated = False

    def is_enabled(self) -> bool:
        return True

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        self.scaled = True
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self.unscaled = True

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        self.updated = True


class FakeLogger:
    def __init__(self, *args: object) -> None:
        self.enabled = True
        self.run_id = "run-id"
        self.run_name = "run"
        self.run_url = "url"

    def log(self, metrics: dict[str, float], step: int) -> None:
        pass

    def log_checkpoint(self, path: Path, step: int, metadata: object) -> None:
        pass

    def finish(self) -> None:
        pass


def test_training_enabled_scaler_warmup_and_wandb_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaler = FakeScaler()
    seeded: list[int] = []
    monkeypatch.setattr(
        pfn_train.torch.amp, "GradScaler", lambda *args, **kwargs: scaler
    )
    monkeypatch.setattr(pfn_train, "WandbLogger", FakeLogger)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seeded.append)
    config = _training_config(
        tmp_path,
        use_warmup=True,
        warmup_steps=1,
        save_checkpoint=False,
        eval_every=None,
        log_file=None,
    )
    pfn_train.train(TinyRegressionGenerator(), _model_config(), config)
    assert scaler.scaled and scaler.unscaled and scaler.updated
    assert seeded == [1]
    assert Path(config.checkpoint_dir).name == "run-id"
