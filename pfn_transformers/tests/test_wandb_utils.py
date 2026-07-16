from dataclasses import dataclass

from pfn_transformerlens.wandb_utils import create_run_name


@dataclass
class _FakeModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 2


@dataclass
class _FakeTrainingConfig:
    learning_rate: float = 1e-3
    num_steps: int = 500
    batch_size: int = 16


def test_create_run_name_includes_selected_fields() -> None:
    run_name = create_run_name(
        base="linreg",
        model_config=_FakeModelConfig(),
        training_config=_FakeTrainingConfig(),
        data_config={"num_tasks": 8},
        include_fields={
            "model": ("d_model",),
            "training": ("learning_rate", "num_steps"),
            "data": ("num_tasks",),
        },
    )

    assert run_name.startswith("linreg-")
    assert "d_model128" in run_name
    assert "learning_rate0.001" in run_name
    assert "num_steps500" in run_name
    assert run_name.endswith("num_tasks8")


def test_create_run_name_respects_max_length() -> None:
    long_name = create_run_name(
        base="experiment",
        model_config=_FakeModelConfig(d_model=512),
        training_config=_FakeTrainingConfig(num_steps=50000),
        data_config={"num_tasks": 16},
        include_fields={
            "model": ("d_model",),
            "training": ("num_steps",),
            "data": ("num_tasks",),
        },
        max_length=32,
    )

    assert len(long_name) <= 32
