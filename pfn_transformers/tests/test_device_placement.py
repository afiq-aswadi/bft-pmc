"""Tests for device placement behavior.

Check whether models handle device movement correctly across different scenarios.
"""

import pytest
import torch

from pfn_transformerlens.model.configs import (
    ClassificationPFNConfig,
    SupervisedRegressionPFNConfig,
    UnsupervisedPFNConfig,
)
from pfn_transformerlens.model.PFN import PFNModel


class TestDevicePlacement:
    """Test device placement for model components."""

    def test_supervised_model_to_device_moves_all_components(self) -> None:
        """Test that model.to(device) moves transformer AND input_proj (correct usage)."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_dim=2,
            prediction_type="point",
            act_fn="gelu",
        )
        model = PFNModel(config)

        initial_device = next(model.parameters()).device

        # move entire model to different device
        if torch.backends.mps.is_available() and initial_device.type != "mps":
            target_device = torch.device("mps")
        elif torch.cuda.is_available() and initial_device.type != "cuda":
            target_device = torch.device("cuda")
        elif initial_device.type != "cpu":
            target_device = torch.device("cpu")
        else:
            raise pytest.skip.Exception(
                "Cannot test device movement - already on only available device"
            )

        model = model.to(target_device)

        # all components should be on the target device
        assert next(model.parameters()).device.type == target_device.type
        assert model.transformer.W_E.device.type == target_device.type
        assert model.input_proj.weight.device.type == target_device.type

    def test_initial_device_from_config(self) -> None:
        """Test that model starts on the device specified by HookedTransformer config."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_dim=2,
            prediction_type="point",
            act_fn="gelu",
        )
        model = PFNModel(config)

        # model should start on whatever device HookedTransformer defaults to
        # (typically MPS on Mac, CUDA on GPU machines, CPU otherwise)
        transformer_device = next(model.transformer.parameters()).device
        input_proj_device = model.input_proj.weight.device

        # both components should start on the same device
        assert transformer_device == input_proj_device

    def test_unsupervised_model_to_device_moves_all_components(self) -> None:
        """Test device movement for unsupervised models."""
        config = UnsupervisedPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_type="discrete",
            prediction_type="distribution",
        )
        model = PFNModel(config)

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise pytest.skip.Exception("No GPU available for device placement test")

        model = model.to(device)

        assert next(model.parameters()).device.type == device.type
        assert model.transformer.W_E.device.type == device.type
        assert model.input_proj.weight.device.type == device.type

    def test_classification_model_to_device_moves_all_components(self) -> None:
        """Test device movement for classification models."""
        config = ClassificationPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            num_classes=3,
            input_dim=2,
            d_vocab=1000,
            act_fn="gelu",
        )
        model = PFNModel(config)

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise pytest.skip.Exception("No GPU available for device placement test")

        model = model.to(device)

        assert next(model.parameters()).device.type == device.type
        assert model.transformer.W_E.device.type == device.type
        assert model.input_proj.weight.device.type == device.type

    def test_forward_pass_after_device_move(self) -> None:
        """Test that forward pass works correctly after moving to device."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_dim=2,
            prediction_type="point",
            act_fn="gelu",
        )
        model = PFNModel(config)

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise pytest.skip.Exception("No GPU available for device placement test")

        model = model.to(device)

        # create inputs on the same device
        x = torch.randn(2, 5, 2, device=device)
        y = torch.randn(2, 5, device=device)

        # forward pass should work without device errors
        logits = model(x, y)
        assert logits.device.type == device.type
        assert logits.shape == (2, 5, 1)

    def test_predict_on_prompt_handles_device_automatically(self) -> None:
        """Test that predict_on_prompt moves inputs to model device automatically."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_dim=2,
            prediction_type="point",
            act_fn="gelu",
        )
        model = PFNModel(config)

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise pytest.skip.Exception("No GPU available for device placement test")

        model = model.to(device)

        # create inputs on CPU
        x_cpu = torch.randn(5, 2)
        y_cpu = torch.randn(5)

        # predict_on_prompt should handle device placement internally
        pred = model.predict_on_prompt(x_cpu, y_cpu)

        # output should be on CPU (as per implementation)
        assert pred.preds.device.type == "cpu"

    def test_multiple_device_moves(self) -> None:
        """Test moving model between devices multiple times (correct usage)."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_dim=1,
            prediction_type="point",
            act_fn="gelu",
        )
        model = PFNModel(config)

        if torch.backends.mps.is_available():
            gpu_device = torch.device("mps")
        elif torch.cuda.is_available():
            gpu_device = torch.device("cuda")
        else:
            raise pytest.skip.Exception("No GPU available for device placement test")

        # move to GPU
        model = model.to(gpu_device)
        assert next(model.parameters()).device.type == gpu_device.type
        assert model.input_proj.weight.device.type == gpu_device.type

        # move back to CPU
        model = model.to("cpu")
        assert next(model.parameters()).device.type == "cpu"
        assert model.input_proj.weight.device.type == "cpu"
