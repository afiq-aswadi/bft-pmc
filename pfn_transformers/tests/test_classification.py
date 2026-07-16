"""Tests for classification PFN functionality.

Tests cover:
- ClassificationPFNConfig validation
- Forward pass shapes and outputs
- Loss and accuracy computation
- End-to-end training integration
- Multi-class scenarios (binary, 5-class, 10-class)
- Edge cases (single batch, seq_len=1)
"""

import pytest
import torch

from pfn_transformerlens.model.configs import ClassificationPFNConfig
from pfn_transformerlens.model.PFN import PFNModel
from pfn_transformerlens.train import TrainingConfig, compute_loss, train


def make_classification_config(
    num_classes: int = 2,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 2,
    d_head: int = 32,
    n_ctx: int = 128,
    input_dim: int = 2,
    **kwargs: object,
) -> ClassificationPFNConfig:
    """Helper to create ClassificationPFNConfig with sensible defaults."""
    return ClassificationPFNConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_ctx=n_ctx,
        d_head=d_head,
        n_heads=n_heads,
        num_classes=num_classes,
        input_dim=input_dim,
        d_vocab=1000,  # Required by HookedTransformer even though not used
        act_fn="gelu",
        **kwargs,
    )


class TestClassificationPFNConfig:
    """Test ClassificationPFNConfig validation and initialization."""

    def test_default_config(self) -> None:
        """Test that default config initializes correctly."""
        config = make_classification_config()
        assert config.num_classes == 2
        assert config.mask_type == "autoregressive-pfn"
        assert config.d_vocab_out == 2
        assert config.input_dim == 2

    def test_num_classes_must_be_positive(self) -> None:
        """Test that num_classes must be positive."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            make_classification_config(num_classes=0)

        with pytest.raises(ValueError, match="num_classes must be positive"):
            make_classification_config(num_classes=-1)

    def test_custom_num_classes(self) -> None:
        """Test config with custom num_classes."""
        for num_classes in [2, 5, 10, 100]:
            config = make_classification_config(num_classes=num_classes)
            assert config.num_classes == num_classes
            assert config.d_vocab_out == num_classes

    def test_d_vocab_out_matches_num_classes(self) -> None:
        """Test that d_vocab_out is automatically set to num_classes."""
        for num_classes in [2, 5, 10, 50]:
            config = make_classification_config(num_classes=num_classes)
            assert config.d_vocab_out == num_classes

    def test_custom_input_dim(self) -> None:
        """Test config with custom input dimensions."""
        for input_dim in [1, 3, 10]:
            config = make_classification_config(input_dim=input_dim)
            assert config.input_dim == input_dim

    def test_mask_type_options(self) -> None:
        """Test both mask types work for classification."""
        config_pfn = make_classification_config(mask_type="autoregressive-pfn")
        assert config_pfn.mask_type == "autoregressive-pfn"

        config_gpt2 = make_classification_config(mask_type="gpt2")
        assert config_gpt2.mask_type == "gpt2"


class TestClassificationForwardPass:
    """Test forward pass for classification models."""

    def test_forward_output_shape_binary(self) -> None:
        """Test forward pass produces correct output shape for binary classification."""
        config = make_classification_config(num_classes=2)
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        logits = model(x, y)

        assert logits.shape == (batch_size, seq_len, 2)

    def test_forward_output_shape_multiclass(self) -> None:
        """Test forward pass for multi-class classification."""
        for num_classes in [5, 10, 20]:
            config = make_classification_config(num_classes=num_classes)
            model = PFNModel(config)
            device = next(model.parameters()).device

            batch_size, seq_len = 2, 8
            x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
            y = torch.randint(0, num_classes, (batch_size, seq_len), device=device)

            logits = model(x, y)

            assert logits.shape == (batch_size, seq_len, num_classes)

    def test_forward_with_cache(self) -> None:
        """Test forward pass with cache returns both logits and cache."""
        config = make_classification_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        result = model(x, y, return_cache=True)

        assert isinstance(result, tuple)
        logits, cache = result
        assert logits.shape == (batch_size, seq_len, config.num_classes)
        assert cache is not None

    def test_different_input_dims(self) -> None:
        """Test forward pass with various input dimensions."""
        for input_dim in [1, 3, 5, 10]:
            config = make_classification_config(input_dim=input_dim, num_classes=3)
            model = PFNModel(config)
            device = next(model.parameters()).device

            batch_size, seq_len = 2, 8
            x = torch.randn(batch_size, seq_len, input_dim, device=device)
            y = torch.randint(0, 3, (batch_size, seq_len), device=device)

            logits = model(x, y)

            assert logits.shape == (batch_size, seq_len, 3)


class TestClassificationLoss:
    """Test loss computation for classification models."""

    def test_loss_computation(self) -> None:
        """Test that loss is computed correctly."""
        config = make_classification_config()
        model = PFNModel(config)
        # Get device from transformer which HookedTransformer sets automatically
        device = model.transformer.cfg.device

        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long, device=device)

        loss, metrics = compute_loss(model, x, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_accuracy_metric(self) -> None:
        """Test that accuracy metric is computed."""
        config = make_classification_config(num_classes=5)
        model = PFNModel(config)
        device = model.transformer.cfg.device

        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 5, (batch_size, seq_len), dtype=torch.long, device=device)

        _, metrics = compute_loss(model, x, y)

        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_loss_is_finite(self) -> None:
        """Test that loss is always finite."""
        config = make_classification_config()
        model = PFNModel(config)
        device = model.transformer.cfg.device

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long, device=device)

        loss, _ = compute_loss(model, x, y)

        assert torch.isfinite(loss)

    def test_perfect_prediction_low_loss(self) -> None:
        """Test that perfect predictions yield lower loss than random."""
        config = make_classification_config(num_classes=2)
        model = PFNModel(config)
        device = model.transformer.cfg.device

        batch_size, seq_len = 4, 16

        # All zeros
        x_constant = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y_constant = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        # Random targets
        x_random = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y_random = torch.randint(0, 2, (batch_size, seq_len), device=device)

        # Train briefly on constant sequence
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(100):
            loss, _ = compute_loss(model, x_constant, y_constant)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss on trained sequence should be lower
        with torch.no_grad():
            loss_constant, _ = compute_loss(model, x_constant, y_constant)
            loss_random, _ = compute_loss(model, x_random, y_random)

        assert loss_constant.item() < loss_random.item()


class TestClassificationTraining:
    """Test training loop integration for classification models."""

    def test_training_runs_binary(self) -> None:
        """Test that training completes without errors for binary classification."""

        class BinaryClassificationGenerator:
            def __init__(self) -> None:
                self.input_dim = 2

            def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
                x = torch.randn(seq_len, self.input_dim)
                # Simple linearly separable pattern
                y = (x[:, 0] > 0).long()
                return x, y

        config = make_classification_config(num_classes=2)
        training_config = TrainingConfig(
            batch_size=4,
            seq_len=16,
            num_steps=10,
            log_every=5,
            save_checkpoint=False,
        )

        generator = BinaryClassificationGenerator()
        model = train(
            data_generator=generator,
            model_config=config,
            training_config=training_config,
        )

        from pfn_transformerlens.model.PFN import BasePFN

        assert isinstance(model, BasePFN)

    def test_training_runs_multiclass(self) -> None:
        """Test training with multi-class classification."""

        class MulticlassGenerator:
            def __init__(self) -> None:
                self.input_dim = 3

            def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
                x = torch.randn(seq_len, self.input_dim)
                # 5 classes based on simple rules
                y = torch.clamp((x[:, 0] * 2 + x[:, 1]).long(), min=0, max=4)
                return x, y

        config = make_classification_config(num_classes=5, input_dim=3)
        training_config = TrainingConfig(
            batch_size=4,
            seq_len=16,
            num_steps=10,
            log_every=5,
            save_checkpoint=False,
        )

        generator = MulticlassGenerator()
        model = train(
            data_generator=generator,
            model_config=config,
            training_config=training_config,
        )

        from pfn_transformerlens.model.PFN import BasePFN

        assert isinstance(model, BasePFN)

    def test_loss_decreases_during_training(self) -> None:
        """Test that loss decreases on simple pattern."""
        config = make_classification_config(num_classes=2)
        model = PFNModel(config)
        device = model.transformer.cfg.device

        # Simple pattern: always class 0
        x = torch.randn(4, 16, config.input_dim, device=device)
        y = torch.zeros(4, 16, dtype=torch.long, device=device)

        # Initial loss
        with torch.no_grad():
            loss_initial, _ = compute_loss(model, x, y)

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(50):
            loss, _ = compute_loss(model, x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final loss should be lower
        with torch.no_grad():
            loss_final, _ = compute_loss(model, x, y)

        assert loss_final.item() < loss_initial.item()

    def test_accuracy_increases_during_training(self) -> None:
        """Test that accuracy increases on simple pattern."""
        config = make_classification_config(num_classes=2)
        model = PFNModel(config)
        device = model.transformer.cfg.device

        # Simple pattern
        x = torch.randn(4, 16, config.input_dim, device=device)
        y = torch.zeros(4, 16, dtype=torch.long, device=device)

        # Initial accuracy
        with torch.no_grad():
            _, metrics_initial = compute_loss(model, x, y)
            acc_initial = metrics_initial["accuracy"]

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(100):
            loss, _ = compute_loss(model, x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final accuracy should be higher or equal (could start at 100% by chance)
        with torch.no_grad():
            _, metrics_final = compute_loss(model, x, y)
            acc_final = metrics_final["accuracy"]

        assert acc_final >= acc_initial


class TestClassificationEdgeCases:
    """Test edge cases for classification models."""

    def test_seq_len_one(self) -> None:
        """Test with sequence length of 1."""
        config = make_classification_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 1
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        logits = model(x, y)

        assert logits.shape == (batch_size, seq_len, config.num_classes)

    def test_single_batch(self) -> None:
        """Test with batch size of 1."""
        config = make_classification_config()
        model = PFNModel(config)
        device = model.transformer.cfg.device

        batch_size, seq_len = 1, 16
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long, device=device)

        logits = model(x, y)
        loss, metrics = compute_loss(model, x, y)

        assert logits.shape == (batch_size, seq_len, config.num_classes)
        assert torch.isfinite(loss)

    def test_class_boundary_values(self) -> None:
        """Test that class boundary values (0 and num_classes-1) work correctly."""
        config = make_classification_config(num_classes=5)
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 8

        # Test with all zeros
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y_zeros = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        logits_zeros = model(x, y_zeros)
        assert logits_zeros.shape == (batch_size, seq_len, 5)

        # Test with all max values
        y_max = torch.full((batch_size, seq_len), 4, dtype=torch.long, device=device)
        logits_max = model(x, y_max)
        assert logits_max.shape == (batch_size, seq_len, 5)

    def test_large_num_classes(self) -> None:
        """Test with large number of classes."""
        config = make_classification_config(num_classes=100, d_model=128, n_heads=4)
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 100, (batch_size, seq_len), device=device)

        logits = model(x, y)

        assert logits.shape == (batch_size, seq_len, 100)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model correctly."""
        config = make_classification_config()
        model = PFNModel(config)
        device = model.transformer.cfg.device

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long, device=device)

        loss, _ = compute_loss(model, x, y)
        loss.backward()

        # Check that some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and (param.grad != 0).any():
                has_grad = True
                break

        assert has_grad


class TestClassificationPrediction:
    """Test prediction functionality for classification models."""

    def test_predict_on_prompt(self) -> None:
        """Test predict_on_prompt returns ClassificationPrediction."""
        config = make_classification_config(num_classes=3)
        model = PFNModel(config)

        seq_len = 10
        x = torch.randn(seq_len, config.input_dim)
        y = torch.randint(0, 3, (seq_len,))

        preds = model.predict_on_prompt(x, y)

        from pfn_transformerlens.model.PFN import ClassificationPrediction

        assert isinstance(preds, ClassificationPrediction)
        assert preds.probs.shape == (seq_len, 3)
        assert torch.allclose(preds.probs.sum(dim=-1), torch.ones(seq_len), atol=1e-5)

    def test_predict_with_batch_dimension(self) -> None:
        """Test prediction with explicit batch dimension."""
        config = make_classification_config(num_classes=2)
        model = PFNModel(config)

        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, config.input_dim)
        y = torch.randint(0, 2, (batch_size, seq_len))

        preds = model.predict_on_prompt(x, y)

        assert preds.probs.shape == (batch_size, seq_len, 2)

    def test_predict_with_logits(self) -> None:
        """Test prediction returns logits when requested."""
        config = make_classification_config()
        model = PFNModel(config)

        seq_len = 10
        x = torch.randn(seq_len, config.input_dim)
        y = torch.randint(0, 2, (seq_len,))

        preds = model.predict_on_prompt(x, y, return_logits=True)

        assert preds.logits is not None
        assert preds.logits.shape == (seq_len, 2)

    def test_predict_with_temperature(self) -> None:
        """Test prediction with different temperatures."""
        config = make_classification_config()
        model = PFNModel(config)

        seq_len = 10
        x = torch.randn(seq_len, config.input_dim)
        y = torch.randint(0, 2, (seq_len,))

        preds_low_temp = model.predict_on_prompt(x, y, temperature=0.1)
        preds_high_temp = model.predict_on_prompt(x, y, temperature=10.0)

        # Low temperature should give more peaked distribution
        entropy_low = (
            -(preds_low_temp.probs * torch.log(preds_low_temp.probs + 1e-10))
            .sum(dim=-1)
            .mean()
        )
        entropy_high = (
            -(preds_high_temp.probs * torch.log(preds_high_temp.probs + 1e-10))
            .sum(dim=-1)
            .mean()
        )

        assert entropy_low < entropy_high


class TestCategoricalYType:
    """Test categorical y_type functionality."""

    def test_categorical_creates_embedding(self) -> None:
        """Test that categorical y_type creates embedding layer."""
        config = make_classification_config(y_type="categorical", num_classes=5)
        model = PFNModel(config)

        assert hasattr(model, "y_embed")
        assert isinstance(model.y_embed, torch.nn.Embedding)
        assert model.y_embed.num_embeddings == 5

    def test_continuous_creates_linear(self) -> None:
        """Test that continuous y_type creates linear layer."""
        config = make_classification_config(y_type="continuous")
        model = PFNModel(config)

        assert hasattr(model, "input_proj")
        assert isinstance(model.input_proj, torch.nn.Linear)

    def test_categorical_forward_autoregressive(self) -> None:
        """Test categorical y_type with autoregressive-pfn mask."""
        config = make_classification_config(
            y_type="categorical", num_classes=3, mask_type="autoregressive-pfn"
        )
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 3, (batch_size, seq_len), device=device)

        output = model(x, y.float())
        assert output.shape == (batch_size, seq_len, 3)

    def test_categorical_forward_gpt2(self) -> None:
        """Test categorical y_type with gpt2 mask."""
        config = make_classification_config(
            y_type="categorical", num_classes=4, mask_type="gpt2"
        )
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 4, (batch_size, seq_len), device=device)

        output = model(x, y.float())
        assert output.shape == (batch_size, seq_len, 4)

    def test_categorical_training(self) -> None:
        """Test that training works with categorical y_type."""
        config = make_classification_config(
            y_type="categorical", num_classes=2, n_ctx=32
        )
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        initial_loss, _ = compute_loss(model, x, y.float())

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(10):
            optimizer.zero_grad()
            loss, _ = compute_loss(model, x, y.float())
            loss.backward()
            optimizer.step()

        final_loss, _ = compute_loss(model, x, y.float())
        assert final_loss < initial_loss

    def test_categorical_predict(self) -> None:
        """Test predict_on_prompt works with categorical y_type."""
        config = make_classification_config(y_type="categorical", num_classes=3)
        model = PFNModel(config)

        seq_len = 10
        x = torch.randn(seq_len, config.input_dim)
        y = torch.randint(0, 3, (seq_len,))

        preds = model.predict_on_prompt(x, y)
        assert preds.probs.shape == (seq_len, 3)

    def test_categorical_gradient_flow(self) -> None:
        """Test gradients flow through categorical embeddings."""
        config = make_classification_config(y_type="categorical", num_classes=2)
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        loss, _ = compute_loss(model, x, y.float())
        loss.backward()

        assert model.x_proj.weight.grad is not None
        assert (model.x_proj.weight.grad != 0).any()
        assert model.y_embed.weight.grad is not None
        assert (model.y_embed.weight.grad != 0).any()

    def test_categorical_multiclass(self) -> None:
        """Test categorical y_type with many classes."""
        config = make_classification_config(y_type="categorical", num_classes=10)
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
        y = torch.randint(0, 10, (batch_size, seq_len), device=device)

        output = model(x, y.float())
        assert output.shape == (batch_size, seq_len, 10)

        preds = model.predict_on_prompt(x[0], y[0])
        assert preds.probs.shape == (seq_len, 10)
        assert torch.allclose(preds.probs.sum(dim=-1), torch.ones(seq_len))
