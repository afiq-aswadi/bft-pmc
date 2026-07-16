from __future__ import annotations

from typing import cast

import pytest
import torch
from torch.distributions import Normal

from pfn_transformerlens.model.PFN import (
    BasePFN,
    PFNModel,
    SupervisedPFN,
    UnsupervisedPFN,
)
from pfn_transformerlens.model.PFNMasks import create_custom_mask_hook
from pfn_transformerlens.model.bucketizer import Bucketizer
from pfn_transformerlens.model.configs import (
    BasePFNConfig,
    ClassificationPFNConfig,
    SupervisedRegressionPFNConfig,
    UnsupervisedPFNConfig,
)


def _common() -> dict[str, object]:
    return {
        "n_layers": 1,
        "d_model": 8,
        "d_head": 4,
        "n_heads": 2,
        "d_mlp": 8,
        "n_ctx": 8,
        "d_vocab": 4,
        "act_fn": "gelu",
        "device": "cpu",
    }


def _regression_config(
    *,
    prediction_type: str = "point",
    mask_type: str = "autoregressive-pfn",
) -> SupervisedRegressionPFNConfig:
    kwargs = _common()
    kwargs.update(
        {
            "input_dim": 1,
            "prediction_type": prediction_type,
            "mask_type": mask_type,
        }
    )
    if prediction_type == "distribution":
        kwargs.update(
            {
                "bucket_type": "uniform",
                "bucket_support": "bounded",
                "y_min": -1.0,
                "y_max": 1.0,
            }
        )
    return SupervisedRegressionPFNConfig(**kwargs)


def _unsupervised_config(
    *,
    input_type: str = "discrete",
    prediction_type: str = "distribution",
) -> UnsupervisedPFNConfig:
    kwargs = _common()
    kwargs.update({"input_type": input_type, "prediction_type": prediction_type})
    if input_type == "continuous" and prediction_type == "distribution":
        kwargs.update(
            {
                "bucket_type": "uniform",
                "bucket_support": "bounded",
                "y_min": -1.0,
                "y_max": 1.0,
            }
        )
    return UnsupervisedPFNConfig(**kwargs)


def test_base_model_contract_and_bucket_accessors() -> None:
    base = cast(BasePFN, object())
    with pytest.raises(NotImplementedError):
        BasePFN._setup_input_proj(base)

    distribution_model = PFNModel(_regression_config(prediction_type="distribution"))
    buckets = distribution_model.get_bucket_values(torch.tensor([[-0.5, 0.5]]))
    assert buckets.shape == (1, 2)
    assert distribution_model.log_bucket_densities(torch.zeros(1, 4)).shape == (1, 4)

    point_model = PFNModel(_regression_config())
    with pytest.raises(RuntimeError, match="only available"):
        _ = point_model.bucketizer


def test_transformer_input_position_validation_and_shortformer() -> None:
    model = PFNModel(_regression_config())
    hidden = torch.zeros(2, 2, model.config.d_model)
    original_position_type = model.transformer.cfg.positional_embedding_type
    model.transformer.cfg.positional_embedding_type = "unsupported"
    with pytest.raises(ValueError, match="Unsupported"):
        model._prepare_transformer_input(hidden)

    model.transformer.cfg.positional_embedding_type = original_position_type
    original_context = model.transformer.cfg.n_ctx
    model.transformer.cfg.n_ctx = 1
    with pytest.raises(ValueError, match="exceeds"):
        model._prepare_transformer_input(hidden)

    model.transformer.cfg.n_ctx = original_context
    model.transformer.cfg.positional_embedding_type = "shortformer"
    residual, shortformer = model._prepare_transformer_input(hidden)
    assert residual.shape == hidden.shape
    assert shortformer is not None and shortformer.shape == hidden.shape


def test_unsupervised_continuous_generation_and_prediction_edges() -> None:
    model = PFNModel(
        _unsupervised_config(input_type="continuous", prediction_type="distribution")
    )
    assert isinstance(model, UnsupervisedPFN)
    with pytest.raises(ValueError, match="temperature"):
        model.generate(1, temperature=0)
    with pytest.raises(ValueError, match="num_rollouts"):
        model.generate(1, num_rollouts=0)

    generated = model.generate(1, sample=False, num_rollouts=2)
    assert generated.shape == (2, 1)
    prediction = model.predict_on_prompt(torch.tensor([0.0]))
    assert prediction.y_grid.shape == (4,)
    with pytest.raises(ValueError, match="temperature"):
        model.predict_on_prompt(torch.tensor([0.0]), temperature=0)

    default_output = UnsupervisedPFNConfig(**_common(), d_vocab_out=-1)
    assert default_output.d_vocab_out == default_output.d_vocab


def test_supervised_generation_prediction_and_gpt2_edges() -> None:
    x_distribution = Normal(0.0, 1.0)
    point_model = PFNModel(_regression_config())
    assert isinstance(point_model, SupervisedPFN)
    with pytest.raises(ValueError, match="Both prompt_x"):
        point_model.generate(
            x_distribution,
            1,
            prompt_x=torch.zeros(1, 1, 1),
        )

    point_model.config.mask_type = "bad"
    with pytest.raises(ValueError, match="Invalid mask"):
        point_model(torch.zeros(1, 1, 1), torch.zeros(1, 1))

    classification = PFNModel(
        ClassificationPFNConfig(
            **_common(),
            input_dim=1,
            num_classes=2,
        )
    )
    assert isinstance(classification, SupervisedPFN)
    _, classes = classification.generate(
        x_distribution,
        1,
        sample=False,
    )
    assert classes.shape == (1, 1, 1)
    with pytest.raises(ValueError, match="temperature"):
        classification.predict_on_prompt(
            torch.zeros(1, 1),
            torch.zeros(1, dtype=torch.long),
            temperature=0,
        )

    distribution = PFNModel(_regression_config(prediction_type="distribution"))
    assert isinstance(distribution, SupervisedPFN)
    with pytest.raises(ValueError, match="temperature"):
        distribution.predict_on_prompt(
            torch.zeros(1, 1),
            torch.zeros(1),
            temperature=0,
        )

    gpt2_model = PFNModel(_regression_config(mask_type="gpt2"))
    assert isinstance(gpt2_model, SupervisedPFN)
    logits, cache = gpt2_model(
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2),
        return_cache=True,
    )
    assert logits.shape == (1, 2, 1)
    assert len(cache) > 0


def test_model_and_mask_factories_reject_unknown_types() -> None:
    with pytest.raises(ValueError, match="Unknown config"):
        PFNModel(BasePFNConfig(**_common()))
    with pytest.raises(ValueError, match="Unsupported mask"):
        create_custom_mask_hook("bad", 2, 2)
    with pytest.raises(ValueError, match="bucket support"):
        Bucketizer(
            bucket_type="uniform",
            bucket_support="bad",
            d_vocab=2,
            y_min=-1.0,
            y_max=1.0,
        )
