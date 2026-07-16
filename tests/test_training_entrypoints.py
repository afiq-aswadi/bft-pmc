from __future__ import annotations

from dataclasses import replace
import importlib
import sys

import pytest
import torch

import balls_and_urns.train as bau_train
import linear_regression.train as lr_train


def _lr_config() -> lr_train.TrainConfig:
    return lr_train.TrainConfig(
        d_model=4,
        n_layers=1,
        n_heads=2,
        d_mlp=4,
        d_head=2,
        d_vocab=4,
        n_ctx=8,
        input_dim=2,
        num_tasks=2,
        batch_size=2,
        seq_len=2,
        num_steps=1,
        warmup_steps=0,
        eval_every=1,
        eval_batches=1,
    )


def _bau_config() -> bau_train.TrainConfig:
    return bau_train.TrainConfig(
        d_model=4,
        d_mlp=4,
        n_layers=1,
        n_heads=2,
        d_head=2,
        n_ctx=8,
        vocab_size=2,
        num_tasks=2,
        batch_size=2,
        seq_len=2,
        num_steps=1,
        warmup_steps=0,
        eval_every=1,
        eval_batches=1,
    )


@pytest.mark.parametrize(
    "config",
    [
        replace(_lr_config(), d_model=0),
        replace(_lr_config(), d_model=5),
        replace(_lr_config(), learning_rate=0),
        replace(_lr_config(), bucket_type="bad"),
        replace(_lr_config(), y_min=1, y_max=1),
    ],
)
def test_lr_train_config_validation(config: lr_train.TrainConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_lr_train_rejects_unsupported_experiment_types() -> None:
    with pytest.raises(ValueError, match="discrete prior"):
        lr_train._validate_config(replace(_lr_config(), prior_type="gaussian"))
    with pytest.raises(ValueError, match="linear likelihood"):
        lr_train._validate_config(replace(_lr_config(), function_type="quadratic"))


def test_lr_border_estimation_and_training_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lr_train,
        "sample_batch",
        lambda generator, batch_size, seq_len: (
            torch.ones(batch_size, seq_len, 2),
            torch.arange(seq_len).float().expand(batch_size, -1),
        ),
    )
    monkeypatch.setattr(
        lr_train,
        "estimate_riemann_borders",
        lambda values, num_buckets: torch.linspace(
            values.min(), values.max(), num_buckets + 1
        ),
    )
    borders = lr_train.estimate_borders_from_data(object(), 3, 2, 2)
    assert borders.shape == (3,)
    for args in [(0, 2, 2), (2, 1, 2), (2, 2, 0)]:
        with pytest.raises(ValueError):
            lr_train.estimate_borders_from_data(object(), *args)

    calls: list[tuple[object, object, object, object, object]] = []
    monkeypatch.setattr(lr_train, "create_run_name", lambda **kwargs: "run")
    monkeypatch.setattr(
        lr_train,
        "train",
        lambda data, model, config, **kwargs: calls.append(
            (data, model, config, kwargs["eval_data_generator"], kwargs["data_config"])
        ),
    )
    lr_train.main(_lr_config())
    assert calls[0][1].bucket_type == "uniform"
    assert calls[0][1].y_min == -10.0

    monkeypatch.setattr(
        lr_train,
        "estimate_borders_from_data",
        lambda *args: torch.tensor([-1.0, 0.0, 1.0]),
    )
    lr_train.main(replace(_lr_config(), bucket_type="riemann", d_vocab=2))
    assert calls[1][1].bucket_type == "riemann"
    assert calls[1][1].riemann_borders is not None


@pytest.mark.parametrize(
    "config",
    [
        replace(_bau_config(), d_model=0),
        replace(_bau_config(), d_model=5),
        replace(_bau_config(), alpha_value=0),
    ],
)
def test_bau_train_config_validation(config: bau_train.TrainConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_bau_training_dispatch_seeded_and_unseeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object, object, object, object]] = []
    monkeypatch.setattr(bau_train, "create_run_name", lambda **kwargs: "run")
    monkeypatch.setattr(
        bau_train,
        "train",
        lambda data, model, config, **kwargs: calls.append(
            (data, model, config, kwargs["eval_data_generator"], kwargs["data_config"])
        ),
    )
    bau_train.main(_bau_config())
    bau_train.main(replace(_bau_config(), seed=4))
    assert len(calls) == 2
    assert calls[0][1].d_vocab == 3
    assert calls[0][1].d_vocab_out == 2


def test_root_train_and_eval_dispatchers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root_train = importlib.import_module("train")
    root_eval = importlib.import_module("eval")
    root_train.main([])
    root_eval.main(["--help"])
    assert "Usage:" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="Unknown training"):
        root_train.main(["bad"])
    with pytest.raises(SystemExit, match="Unknown eval"):
        root_eval.main(["bad"])

    paths: list[tuple[str, str, list[str]]] = []
    monkeypatch.setattr(
        root_train.runpy,
        "run_path",
        lambda path, run_name: paths.append((str(path), run_name, list(sys.argv))),
    )
    monkeypatch.setattr(
        root_eval.runpy,
        "run_path",
        lambda path, run_name: paths.append((str(path), run_name, list(sys.argv))),
    )
    root_train.main(["lr", "--num-tasks", "2"])
    root_train.main(["beta-bernoulli", "--help"])
    root_eval.main(["plot-lr-sweep", "--help"])
    assert paths[0][1] == paths[1][1] == paths[2][1] == "__main__"
    assert paths[0][2] == ["train.py", "--num-tasks", "2"]
    assert paths[1][2] == ["beta_bernoulli.py", "--help"]

    monkeypatch.setattr(sys, "argv", ["train.py", "bau"])
    root_train.main(None)
    monkeypatch.setattr(sys, "argv", ["eval.py", "plot-bau-sweep"])
    root_eval.main(None)
    assert len(paths) == 5
