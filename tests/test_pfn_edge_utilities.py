from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from pfn_transformerlens import checkpointing
from pfn_transformerlens.checkpointing import CheckpointMetadata
from pfn_transformerlens.model.PFN import PFNModel
from pfn_transformerlens.model.bucketizer import Bucketizer, estimate_riemann_borders
from pfn_transformerlens.model.configs import (
    ClassificationPFNConfig,
    SupervisedRegressionPFNConfig,
    UnsupervisedPFNConfig,
)
from pfn_transformerlens.model.configs.base import validate_bucket_config
from pfn_transformerlens.sampler.data_generator import (
    DeterministicFunctionGenerator,
    SupervisedProbabilisticGenerator,
    UnsupervisedProbabilisticGenerator,
)
from pfn_transformerlens.sampler.dataloader import (
    _collate_batch,
    _is_picklable,
    build_dataloader,
)
from pfn_transformerlens.sampler.prior_likelihood import (
    DiscreteTaskDistribution,
    LikelihoodDistribution,
    PriorDistribution,
)
from pfn_transformerlens.sampler.sampler import Sampler
from pfn_transformerlens.wandb_logger import WandbLogger
import pfn_transformerlens.wandb_utils as wandb_utils


def _point_config() -> SupervisedRegressionPFNConfig:
    return SupervisedRegressionPFNConfig(
        n_layers=1,
        d_model=8,
        d_head=4,
        n_heads=2,
        d_mlp=8,
        n_ctx=8,
        d_vocab=4,
        input_dim=2,
        prediction_type="point",
        act_fn="gelu",
    )


def _metadata() -> CheckpointMetadata:
    return CheckpointMetadata("now", "id", "name", "url", "hash")


def test_checkpoint_environment_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        checkpointing.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="ignored"),
    )
    assert checkpointing._get_git_hash() is None
    monkeypatch.setattr(
        checkpointing.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="abc\n"),
    )
    assert checkpointing._get_git_hash() == "abc"

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("git", 1)

    monkeypatch.setattr(checkpointing.subprocess, "run", timeout)
    assert checkpointing._get_git_hash() is None

    monkeypatch.setattr(checkpointing.torch.cuda, "is_available", lambda: True)
    assert checkpointing._get_device("auto") == "cuda"
    monkeypatch.setattr(checkpointing.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(checkpointing.torch.backends.mps, "is_available", lambda: True)
    assert checkpointing._get_device("auto") == "mps"
    monkeypatch.setattr(checkpointing.torch.backends.mps, "is_available", lambda: False)
    assert checkpointing._get_device("auto") == "cpu"
    assert checkpointing._get_device("cpu") == "cpu"


def test_checkpoint_step_edges_and_optional_payload(tmp_path: Path) -> None:
    assert checkpointing.get_logarithmic_checkpoint_steps(0) == [0]
    assert checkpointing.get_logarithmic_checkpoint_steps(1, 3, 2) == [0, 1]
    assert checkpointing.get_logarithmic_checkpoint_steps(5, 1, 2) == [0, 2, 4, 5]

    config = _point_config()
    model = PFNModel(config)
    path = tmp_path / "checkpoint.pt"
    checkpointing.save_checkpoint(
        path,
        step=3,
        model_state=model.state_dict(),
        optimizer_state={"state": 1},
        model_config=config,
        training_config={"steps": 3},
        metadata=_metadata(),
        scheduler_state={"scheduler": 2},
        task_distribution={"tasks": torch.tensor([1.0])},
    )
    raw = torch.load(path, weights_only=False)
    assert raw["scheduler_state_dict"] == {"scheduler": 2}
    loaded, optimizer, metadata = checkpointing.load_checkpoint(
        path, device="cpu", load_optimizer=True
    )
    assert optimizer == {"state": 1}
    assert metadata.git_hash == "hash"
    assert hasattr(loaded, "task_distribution")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("uniform", None, 1.0, None, 2), "both"),
        (("uniform", 1.0, 1.0, None, 2), "less"),
        (("riemann", None, None, None, 2), "requires"),
        (("riemann", None, None, torch.ones(2, 2), 2), "one-dimensional"),
        (("riemann", None, None, torch.ones(2), 2), "length"),
        (("other", None, None, None, 2), "Unknown"),
    ],
)
def test_shared_bucket_config_validation(
    args: tuple[str, float | None, float | None, torch.Tensor | None, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_bucket_config(*args)


def test_model_config_validation_edges() -> None:
    common: dict[str, Any] = {
        "n_layers": 1,
        "d_model": 8,
        "d_head": 4,
        "n_heads": 2,
        "d_mlp": 8,
        "n_ctx": 8,
        "d_vocab": 4,
        "act_fn": "gelu",
    }
    with pytest.raises(ValueError, match="Point prediction"):
        SupervisedRegressionPFNConfig(
            **common,
            prediction_type="point",
            bucket_type="uniform",
            y_min=-1,
            y_max=1,
        )
    with pytest.raises(ValueError, match="requires bucket_type"):
        SupervisedRegressionPFNConfig(**common, prediction_type="distribution")
    with pytest.raises(ValueError, match="mask_type"):
        SupervisedRegressionPFNConfig(
            **common, prediction_type="point", mask_type="bad"
        )
    with pytest.raises(ValueError, match="prediction_type"):
        SupervisedRegressionPFNConfig(**common, prediction_type="bad")

    with pytest.raises(ValueError, match="mask_type"):
        UnsupervisedPFNConfig(**common, mask_type="bad")
    with pytest.raises(ValueError, match="positive"):
        UnsupervisedPFNConfig(**{**common, "d_vocab": 0})
    with pytest.raises(ValueError, match="input_type"):
        UnsupervisedPFNConfig(**common, input_type="bad")
    with pytest.raises(ValueError, match="prediction_type"):
        UnsupervisedPFNConfig(**common, prediction_type="bad")
    with pytest.raises(ValueError, match="require bucket_type"):
        UnsupervisedPFNConfig(
            **common,
            input_type="continuous",
            prediction_type="distribution",
        )
    with pytest.raises(ValueError, match="only be set"):
        UnsupervisedPFNConfig(**common, bucket_type="uniform", y_min=-1, y_max=1)
    point = UnsupervisedPFNConfig(
        **common,
        input_type="continuous",
        prediction_type="point",
    )
    assert point.d_vocab_out == 1

    with pytest.raises(ValueError, match="num_classes"):
        ClassificationPFNConfig(**common, num_classes=0)
    with pytest.raises(ValueError, match="y_type"):
        ClassificationPFNConfig(**common, y_type="bad")
    with pytest.raises(ValueError, match="mask_type"):
        ClassificationPFNConfig(**common, mask_type="bad")


def test_bucketizer_validation_and_riemann_estimation() -> None:
    with pytest.raises(ValueError, match="positive"):
        estimate_riemann_borders(torch.ones(3), num_buckets=0)
    with pytest.raises(ValueError, match="at least one"):
        estimate_riemann_borders(torch.empty(0), num_buckets=2)
    with pytest.raises(ValueError, match="more samples"):
        estimate_riemann_borders(torch.ones(2), num_buckets=2)
    borders = estimate_riemann_borders(
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
        num_buckets=2,
        widen_borders_factor=2,
    )
    assert torch.equal(borders, torch.tensor([-4.0, -1.0, 2.0]))
    with pytest.warns(UserWarning, match="not unique"):
        with pytest.raises(ValueError, match="requested number"):
            estimate_riemann_borders(torch.ones(4), num_buckets=2)

    base = {
        "bucket_type": "uniform",
        "bucket_support": "bounded",
        "d_vocab": 2,
        "y_min": -1.0,
        "y_max": 1.0,
    }
    for overrides in [
        {"d_vocab": 1},
        {"y_min": None},
        {"y_min": 1.0, "y_max": 1.0},
        {"bucket_type": "riemann", "y_min": None, "y_max": None},
        {"bucket_type": "other"},
    ]:
        with pytest.raises(ValueError):
            Bucketizer(**{**base, **overrides})
    with pytest.raises(AssertionError, match="1D"):
        Bucketizer(
            bucket_type="riemann",
            bucket_support="bounded",
            d_vocab=2,
            y_min=None,
            y_max=None,
            borders=torch.ones(3, 1),
        )
    with pytest.raises(AssertionError, match="Expected"):
        Bucketizer(
            bucket_type="riemann",
            bucket_support="bounded",
            d_vocab=2,
            y_min=None,
            y_max=None,
            borders=torch.ones(2),
        )
    with pytest.raises(AssertionError, match="increasing"):
        Bucketizer(
            bucket_type="riemann",
            bucket_support="bounded",
            d_vocab=2,
            y_min=None,
            y_max=None,
            borders=torch.tensor([0.0, 1.0, 1.0]),
        )
    bucketizer = Bucketizer(**base)
    assert (
        bucketizer.decode(torch.tensor([0, 1]), dtype=torch.float64).dtype
        == torch.float64
    )
    with pytest.raises(ValueError, match="temperature"):
        bucketizer.sample(torch.zeros(2), temperature=0)


class PicklableGenerator:
    device = "cpu"

    def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(seq_len, 1), torch.ones(seq_len)


def _loader_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "seq_len": 3,
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
        "prefetch_factor": 2,
        "persistent_workers": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dataloader_collation_and_worker_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.ones(2, 1)
    y = torch.ones(2)
    stacked_x, stacked_y = _collate_batch([(x, y), (x, y)])
    assert stacked_x is not None and stacked_x.shape == (2, 2, 1)
    assert stacked_y.shape == (2, 2)
    none_x, stacked_y = _collate_batch([(None, y), (None, y)])
    assert none_x is None and stacked_y.shape == (2, 2)
    assert _is_picklable(PicklableGenerator())
    assert not _is_picklable(lambda: None)

    unpicklable = PicklableGenerator()
    unpicklable.callback = lambda: None
    with pytest.warns(UserWarning, match="not picklable"):
        loader = build_dataloader(unpicklable, _loader_config(num_workers=1))
    assert loader.num_workers == 0

    cuda_generator = PicklableGenerator()
    cuda_generator.device = "cuda"
    with pytest.warns(UserWarning, match="disabling pin_memory"):
        loader = build_dataloader(cuda_generator, _loader_config(pin_memory=True))
    assert not loader.pin_memory

    loader = build_dataloader(
        PicklableGenerator(),
        _loader_config(num_workers=1, persistent_workers=True),
    )
    assert loader.prefetch_factor == 2
    assert loader.persistent_workers


def _normal_parameterizer(
    theta: torch.Tensor, x: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {
        "loc": torch.zeros(x.shape[0]) + theta,
        "scale": torch.ones(x.shape[0]),
    }


class StringDevicePrior:
    device = "cpu"

    def sample(self) -> torch.Tensor:
        return torch.tensor(2.0)


def test_generator_parameter_round_trips_and_device_conversion() -> None:
    prior = PriorDistribution(torch.distributions.Normal(0.0, 1.0))
    likelihood = LikelihoodDistribution(
        torch.distributions.Normal(0.0, 1.0),
        _normal_parameterizer,
        input_dim=1,
    )
    supervised = SupervisedProbabilisticGenerator(prior, likelihood)
    (x, y), params = supervised.generate_with_params(3)
    assert x.shape == (3, 1) and y.shape == (3,)
    assert "theta" in params

    unsupervised = UnsupervisedProbabilisticGenerator(prior, likelihood)
    y, params = unsupervised.generate_with_params(3)
    assert y.shape == (3,) and "theta" in params

    deterministic = DeterministicFunctionGenerator(
        StringDevicePrior(),
        lambda inputs, theta: inputs[:, 0] * theta,
        input_dim=1,
        noise_std=0.1,
    )
    x, y = deterministic.generate(3)
    assert x.device.type == "cpu" and y.shape == (3,)
    (x, y), params = deterministic.generate_with_params(3)
    assert x.device.type == "cpu" and y.shape == (3,)
    assert "params" in params


def test_prior_likelihood_edge_behavior() -> None:
    scalar = DiscreteTaskDistribution(torch.tensor([1.0, 2.0]))
    assert scalar.event_shape == torch.Size([])
    scalar_log_prob = scalar.log_prob(torch.tensor([1.0, 3.0]))
    assert torch.isfinite(scalar_log_prob[0]) and torch.isneginf(scalar_log_prob[1])
    assert scalar_log_prob.device == scalar.tasks.device

    vector = DiscreteTaskDistribution(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    vector_log_prob = vector.log_prob(torch.tensor([[1.0, 2.0], [0.0, 0.0]]))
    assert torch.isfinite(vector_log_prob[0]) and torch.isneginf(vector_log_prob[1])
    with pytest.raises(ValueError, match="tasks must have shape"):
        DiscreteTaskDistribution(torch.empty(0))
    with pytest.raises(ValueError, match="finite floating-point"):
        DiscreteTaskDistribution(torch.tensor([1, 2]))

    likelihood = LikelihoodDistribution(
        torch.distributions.Normal(0.0, 1.0),
        _normal_parameterizer,
        input_dim=1,
    )
    with pytest.raises(ValueError, match="Unsupported"):
        likelihood._create_distribution_from_params([])
    with pytest.raises(RuntimeError, match="unconditioned"):
        likelihood.log_prob(torch.tensor(0.0))
    assert "unconditioned" in repr(likelihood)
    conditioned = likelihood.condition_on_prior_and_input(
        torch.tensor(1.0), torch.ones(2, 1)
    )
    assert "loc=(2,)" in repr(conditioned)


def test_sampler_rejects_ambiguous_sources() -> None:
    generator = PicklableGenerator()
    with pytest.raises(ValueError, match="either"):
        Sampler(2, config=object(), data_generator=generator)
    with pytest.raises(ValueError, match="Must provide"):
        Sampler(2)


@dataclass
class Template:
    alpha: int = 1


class PlainTemplate:
    def __init__(self) -> None:
        self.visible = 1
        self._hidden = 2


def test_wandb_run_name_all_input_forms() -> None:
    scheme = wandb_utils.RunNameScheme.from_templates(
        model=Template(),
        training={"steps": None},
        data=PlainTemplate(),
    )
    assert scheme.model_fields == ("alpha",)
    assert scheme.training_fields == ("steps",)
    assert scheme.data_fields == ("visible",)
    assert wandb_utils.RunNameScheme.from_templates().model_fields == ()
    assert wandb_utils._extract_fields_from_template(1) == []

    with pytest.raises(ValueError, match="non-empty"):
        wandb_utils.create_run_name(base="")
    name = wandb_utils.create_run_name(
        base="My experiment",
        model_config=Template(),
        training_config={"steps": 10, "ignored": 2},
        data_config=PlainTemplate(),
        scheme=scheme,
        extra={"enabled": True, "rate": 1.0, "punctuation": "!!!"},
    )
    assert name.startswith("my-experiment-model-alpha1")
    assert "enabledtrue" in name
    assert "rate1" in name
    assert name.endswith("punctuation")
    assert wandb_utils._to_mapping(None) == {}
    assert wandb_utils._to_mapping(Template()) == {"alpha": 1}
    assert wandb_utils._to_mapping({"a": 1}) == {"a": 1}
    assert wandb_utils._to_mapping(PlainTemplate()) == {"visible": 1}
    assert wandb_utils._to_mapping(1) == {}
    assert wandb_utils._slugify("!!!") == "run"
    assert wandb_utils.create_run_name(base="only") == "only"
    assert (
        wandb_utils.create_run_name(
            base="empty",
            model_config=1,
            include_fields={"model": ("missing",)},
        )
        == "empty"
    )
    partial = wandb_utils.create_run_name(
        base="partial",
        model_config={"present": 1},
        include_fields={"model": ("missing", "present")},
    )
    assert partial == "partial-model-present1"


@dataclass
class LoggerDataConfig:
    num_tasks: int = 2


@dataclass
class LoggerTrainingConfig:
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = "run"
    wandb_tags: tuple[str, ...] = ()
    wandb_notes: str | None = None
    wandb_log_model: bool = True


@dataclass
class FakeRun:
    id: str = "run-id"
    name: str = "run-name"
    url: str = "https://example.test/run"


class FakeArtifact:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class FakeWandb:
    def __init__(self, existing_run: bool = False, create_run: bool = True) -> None:
        self.run = FakeRun() if existing_run else None
        self.create_run = create_run
        self.config_updates: list[tuple[dict[str, object], bool]] = []
        self.config = SimpleNamespace(update=self._update_config)
        self.init_calls: list[dict[str, object]] = []
        self.log_calls: list[tuple[dict[str, float], int]] = []
        self.artifacts: list[FakeArtifact] = []
        self.finished = False

    def _update_config(self, config: dict[str, object], allow_val_change: bool) -> None:
        self.config_updates.append((config, allow_val_change))

    def init(self, **kwargs: object) -> None:
        self.init_calls.append(kwargs)
        if self.create_run:
            self.run = FakeRun()

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.log_calls.append((metrics, step))

    def Artifact(self, **kwargs: object) -> FakeArtifact:
        artifact = FakeArtifact(**kwargs)
        self.artifacts.append(artifact)
        return artifact

    def log_artifact(self, artifact: FakeArtifact) -> None:
        assert artifact in self.artifacts

    def finish(self) -> None:
        self.finished = True


def test_wandb_logger_disabled_and_input_validation() -> None:
    disabled = WandbLogger(LoggerTrainingConfig(), _point_config())
    disabled.log({"loss": 1.0}, 1)
    disabled.log_checkpoint("unused.pt", 1)
    disabled.finish()
    with pytest.raises(TypeError, match="dataclass instance"):
        WandbLogger(LoggerTrainingConfig(), _point_config(), data_config={"bad": 1})
    with pytest.raises(TypeError, match="dataclass instance"):
        WandbLogger(
            LoggerTrainingConfig(), _point_config(), data_config=LoggerDataConfig
        )


def test_wandb_logger_enabled_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    training = LoggerTrainingConfig(use_wandb=True)
    logger = WandbLogger(training, _point_config(), LoggerDataConfig())
    assert logger.run_id == "run-id"
    init_config = fake.init_calls[0]["config"]
    assert isinstance(init_config, dict)
    assert init_config["data"] == {"num_tasks": 2}
    logger.log({"loss": 0.5}, 3)
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    logger.log_checkpoint(checkpoint, 3, _metadata())
    metadata = fake.artifacts[0].kwargs["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["git_hash"] == "hash"
    assert metadata["data_config"] == {"num_tasks": 2}
    assert fake.artifacts[0].files == [str(checkpoint)]
    logger.finish()
    assert fake.finished

    existing = FakeWandb(existing_run=True)
    monkeypatch.setitem(sys.modules, "wandb", existing)
    WandbLogger(training, _point_config())
    assert len(existing.config_updates) == 1

    no_run = FakeWandb(create_run=False)
    monkeypatch.setitem(sys.modules, "wandb", no_run)
    no_run_logger = WandbLogger(training, _point_config())
    with pytest.raises(AssertionError, match="initialized"):
        no_run_logger.log_checkpoint(checkpoint, 1)


def test_wandb_logger_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match="required"):
        WandbLogger(LoggerTrainingConfig(use_wandb=True), _point_config())
