"""Tests for unsupervised PFN (next-token prediction) functionality.

Tests cover:
- UnsupervisedPFNConfig validation
- Embedding layer initialization
- Forward pass shapes and outputs
- Training loop integration
- Loss computation
- Edge cases (seq_len=1, various d_vocab values, etc.)
"""

import pytest
import torch

from pfn_transformerlens.model.configs import UnsupervisedPFNConfig
from pfn_transformerlens.model.PFN import PFNModel
from pfn_transformerlens.train import TrainingConfig, compute_loss, train


def make_unsupervised_config(
    d_vocab: int = 2,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 2,
    d_head: int = 32,
    n_ctx: int = 128,
    **kwargs: object,
) -> UnsupervisedPFNConfig:
    """Helper to create UnsupervisedPFNConfig with sensible defaults."""
    return UnsupervisedPFNConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_ctx=n_ctx,
        d_head=d_head,
        n_heads=n_heads,
        d_vocab=d_vocab,
        **kwargs,
    )


class TestUnsupervisedPFNConfig:
    """Test UnsupervisedPFNConfig validation and initialization."""

    def test_default_config(self) -> None:
        """Test that default config initializes correctly."""
        config = make_unsupervised_config()
        assert config.d_vocab == 2
        assert config.mask_type == "gpt2"
        assert config.act_fn == "gelu"
        assert config.d_vocab_out == 2
        assert config.input_dim == 1

    def test_mask_type_must_be_gpt2(self) -> None:
        """Test that mask_type must be 'gpt2'."""
        with pytest.raises(
            ValueError, match="Unsupervised mode only supports mask_type='gpt2'"
        ):
            make_unsupervised_config(mask_type="autoregressive-pfn")

    def test_d_vocab_must_be_positive(self) -> None:
        """Test that d_vocab must be positive."""
        with pytest.raises(ValueError, match="d_vocab must be positive"):
            make_unsupervised_config(d_vocab=0)

        with pytest.raises(ValueError, match="d_vocab must be positive"):
            make_unsupervised_config(d_vocab=-1)

    def test_custom_d_vocab(self) -> None:
        """Test config with custom d_vocab."""
        config = make_unsupervised_config(d_vocab=10)
        assert config.d_vocab == 10
        assert config.d_vocab_out == 10

    def test_d_vocab_out_matches_d_vocab(self) -> None:
        """Test that d_vocab_out is automatically set to d_vocab."""
        for d_vocab in [2, 5, 10, 100]:
            config = make_unsupervised_config(d_vocab=d_vocab)
            assert config.d_vocab_out == d_vocab


class TestUnsupervisedEmbeddingLayer:
    """Test embedding layer initialization for unsupervised models."""

    def test_embedding_layer_exists(self) -> None:
        """Test that unsupervised models use nn.Embedding."""
        config = make_unsupervised_config(d_vocab=2)
        model = PFNModel(config)

        assert isinstance(model.input_proj, torch.nn.Embedding)
        assert not isinstance(model.input_proj, torch.nn.Linear)

    def test_embedding_dimensions(self) -> None:
        """Test embedding layer has correct dimensions."""
        config = make_unsupervised_config(d_model=128, d_vocab=10)
        model = PFNModel(config)

        assert model.input_proj.num_embeddings == 10
        assert model.input_proj.embedding_dim == 128

    def test_supervised_uses_linear(self) -> None:
        """Verify supervised models still use Linear layer."""
        from pfn_transformerlens.model.configs import SupervisedRegressionPFNConfig

        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            act_fn="gelu",
            input_dim=3,
            prediction_type="point",
        )
        model = PFNModel(config)

        assert isinstance(model.input_proj, torch.nn.Linear)
        assert not isinstance(model.input_proj, torch.nn.Embedding)


class TestUnsupervisedForwardPass:
    """Test forward pass for unsupervised models."""

    def test_forward_output_shape(self) -> None:
        """Test that forward pass produces correct output shape."""
        config = make_unsupervised_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 4, 16
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        logits = model(y)

        assert logits.shape == (batch_size, seq_len, config.d_vocab)

    def test_forward_with_cache(self) -> None:
        """Test forward pass with cache returns both logits and cache."""
        config = make_unsupervised_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 8
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        result = model(y, return_cache=True)

        assert isinstance(result, tuple)
        logits, cache = result
        assert logits.shape == (batch_size, seq_len, config.d_vocab)
        assert cache is not None

    def test_long_sequence(self) -> None:
        """Test forward pass with longer sequences."""
        config = make_unsupervised_config(n_ctx=256)
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 128
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        logits = model(y)

        assert logits.shape == (batch_size, seq_len, config.d_vocab)

    def test_different_vocab_sizes(self) -> None:
        """Test forward pass with various vocabulary sizes."""
        for d_vocab in [2, 5, 10, 50]:
            config = make_unsupervised_config(d_vocab=d_vocab)
            model = PFNModel(config)
            device = next(model.parameters()).device

            batch_size, seq_len = 2, 8
            y = torch.randint(0, d_vocab, (batch_size, seq_len), device=device)

            logits = model(y)

            assert logits.shape == (batch_size, seq_len, d_vocab)


class TestUnsupervisedLoss:
    """Test loss computation for unsupervised models."""

    def test_loss_computation(self) -> None:
        """Test that loss is computed correctly."""
        config = make_unsupervised_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 4, 16
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        loss, metrics = compute_loss(model, None, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_accuracy_metric(self) -> None:
        """Test that accuracy metric is computed."""
        config = make_unsupervised_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 4, 16
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        _, metrics = compute_loss(model, None, y)

        assert "accuracy" in metrics
        acc = metrics["accuracy"]
        assert isinstance(acc, float) and 0.0 <= acc <= 1.0

    def test_loss_shape_alignment(self) -> None:
        """Test that loss uses correct shape alignment (logits[:-1] vs targets[1:])."""
        config = make_unsupervised_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 8
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        loss, _ = compute_loss(model, None, y)

        # Should not raise errors and should be finite
        assert torch.isfinite(loss)

    def test_perfect_prediction_low_loss(self) -> None:
        """Test that perfect predictions yield lower loss."""
        config = make_unsupervised_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 4, 16

        # All zeros
        y_constant = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        # Random sequence
        y_random = torch.randint(0, 2, (batch_size, seq_len), device=device)

        # Train briefly on constant sequence
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(100):
            loss, _ = compute_loss(model, None, y_constant)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss on trained sequence should be lower
        with torch.no_grad():
            loss_constant, _ = compute_loss(model, None, y_constant)
            loss_random, _ = compute_loss(model, None, y_random)

        assert loss_constant.item() < loss_random.item()


class TestUnsupervisedTraining:
    """Test training loop integration for unsupervised models."""

    def test_training_runs(self) -> None:
        """Test that training completes without errors."""

        class SimpleGenerator:
            def __init__(self) -> None:
                self.input_dim = 1

            def generate(self, seq_len: int) -> torch.Tensor:
                y = torch.randint(0, 2, (seq_len,))
                return y

        config = make_unsupervised_config()
        training_config = TrainingConfig(
            batch_size=4,
            seq_len=16,
            num_steps=10,
            log_every=5,
            save_checkpoint=False,
        )

        generator = SimpleGenerator()
        model = train(
            data_generator=generator,
            model_config=config,
            training_config=training_config,
        )

        from pfn_transformerlens.model.PFN import BasePFN

        assert isinstance(model, BasePFN)

    def test_loss_decreases_during_training(self) -> None:
        """Test that loss decreases on simple pattern."""

        config = make_unsupervised_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        y = torch.zeros(4, 16, dtype=torch.long, device=device)

        # Initial loss
        with torch.no_grad():
            loss_initial, _ = compute_loss(model, None, y)

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(50):
            loss, _ = compute_loss(model, None, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final loss should be lower
        with torch.no_grad():
            loss_final, _ = compute_loss(model, None, y)

        assert loss_final.item() < loss_initial.item()


class TestUnsupervisedEdgeCases:
    """Test edge cases for unsupervised models."""

    def test_seq_len_one(self) -> None:
        """Test with sequence length of 1."""
        config = make_unsupervised_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 1
        # x removed for unsupervised
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        logits = model(y)

        assert logits.shape == (batch_size, seq_len, config.d_vocab)

    def test_seq_len_two(self) -> None:
        """Test with sequence length of 2 (minimum for next-token prediction loss)."""
        config = make_unsupervised_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 2
        # x removed for unsupervised
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        loss, metrics = compute_loss(model, None, y)

        assert torch.isfinite(loss)
        acc = metrics["accuracy"]
        assert isinstance(acc, float) and acc >= 0.0

    def test_large_vocab(self) -> None:
        """Test with large vocabulary size."""
        config = make_unsupervised_config(d_vocab=1000, d_model=128, n_heads=4)
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 8
        # x removed for unsupervised
        y = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        logits = model(y)

        assert logits.shape == (batch_size, seq_len, 1000)

    def test_single_batch(self) -> None:
        """Test with batch size of 1."""
        config = make_unsupervised_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 1, 16
        # x removed for unsupervised
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        logits = model(y)
        loss, metrics = compute_loss(model, None, y)

        assert logits.shape == (batch_size, seq_len, config.d_vocab)
        assert torch.isfinite(loss)

    def test_vocab_boundary_values(self) -> None:
        """Test that vocabulary boundary values (0 and d_vocab-1) work correctly."""
        config = make_unsupervised_config(d_vocab=5)
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 8

        # Test with all zeros
        # x removed for unsupervised
        y_zeros = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        logits_zeros = model(y_zeros)
        assert logits_zeros.shape == (batch_size, seq_len, 5)

        # Test with all max values
        y_max = torch.full((batch_size, seq_len), 4, dtype=torch.long, device=device)
        logits_max = model(y_max)
        assert logits_max.shape == (batch_size, seq_len, 5)

    def test_no_position_embeddings(self) -> None:
        """Test unsupervised model with positional embeddings disabled."""
        config = make_unsupervised_config(use_pos_emb=False)
        model = PFNModel(config)
        device = next(model.parameters()).device

        # Check that W_pos doesn't require grad
        assert not model.transformer.W_pos.requires_grad

        batch_size, seq_len = 2, 8
        # x removed for unsupervised
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        logits = model(y)
        assert logits.shape == (batch_size, seq_len, config.d_vocab)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model correctly."""
        config = make_unsupervised_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 8
        # x removed for unsupervised
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        loss, _ = compute_loss(model, None, y)
        loss.backward()

        # Check that embedding layer has gradients
        assert model.input_proj.weight.grad is not None
        assert (model.input_proj.weight.grad != 0).any()


class TestUnsupervisedVsSupervisedMode:
    """Test differences between unsupervised and supervised modes."""

    def test_unsupervised_no_interleaving(self) -> None:
        """Test that unsupervised mode doesn't interleave x and y."""
        config = make_unsupervised_config()
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 8
        # x removed for unsupervised
        y = torch.randint(0, 2, (batch_size, seq_len), device=device)

        # Forward pass
        logits = model(y)

        # Output sequence length should match input (not 2*seq_len)
        assert logits.shape == (batch_size, seq_len, config.d_vocab)

    def test_supervised_has_interleaving(self) -> None:
        """Verify supervised mode still does interleaving."""
        from pfn_transformerlens.model.configs import SupervisedRegressionPFNConfig

        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            act_fn="gelu",
            input_dim=1,
            prediction_type="point",
        )
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 1, device=device)
        y = torch.randn(batch_size, seq_len, device=device)

        logits = model(x, y)

        # For point prediction, output should match seq_len (after de-interleaving)
        assert logits.shape == (batch_size, seq_len, 1)


class TestContinuousUnsupervised:
    """Test continuous input unsupervised models (point and distribution predictions)."""

    def test_continuous_point_prediction_config(self) -> None:
        """Test config for continuous input with point prediction."""
        config = UnsupervisedPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_type="continuous",
            prediction_type="point",
        )
        assert config.input_type == "continuous"
        assert config.prediction_type == "point"
        assert config.d_vocab_out == 1
        assert config.bucket_type is None

    def test_continuous_distribution_requires_bucket_type(self) -> None:
        """Test that continuous distribution prediction requires bucket_type."""
        with pytest.raises(
            ValueError,
            match="continuous inputs with distribution predictions require bucket_type",
        ):
            UnsupervisedPFNConfig(
                n_layers=2,
                d_model=64,
                n_ctx=128,
                d_head=32,
                n_heads=2,
                d_vocab=10,
                input_type="continuous",
                prediction_type="distribution",
            )

    def test_continuous_distribution_with_bucket_type(self) -> None:
        """Test config for continuous distribution with bucketing."""
        config = UnsupervisedPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_type="continuous",
            prediction_type="distribution",
            bucket_type="uniform",
            y_min=-1.0,
            y_max=1.0,
        )
        assert config.input_type == "continuous"
        assert config.prediction_type == "distribution"
        assert config.bucket_type == "uniform"
        assert config.d_vocab_out == 10

    def test_continuous_point_uses_linear(self) -> None:
        """Test that continuous input uses Linear layer, not Embedding."""
        config = UnsupervisedPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_type="continuous",
            prediction_type="point",
        )
        model = PFNModel(config)

        assert isinstance(model.input_proj, torch.nn.Linear)
        assert not isinstance(model.input_proj, torch.nn.Embedding)
        assert model.input_proj.in_features == 1
        assert model.input_proj.out_features == 64

    def test_continuous_point_forward_shape(self) -> None:
        """Test forward pass with continuous point prediction."""
        config = UnsupervisedPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_type="continuous",
            prediction_type="point",
        )
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 4, 16
        y = torch.randn(batch_size, seq_len, device=device)

        logits = model(y)

        assert logits.shape == (batch_size, seq_len, 1)

    def test_continuous_distribution_forward_shape(self) -> None:
        """Test forward pass with continuous distribution prediction."""
        config = UnsupervisedPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_type="continuous",
            prediction_type="distribution",
            bucket_type="uniform",
            y_min=-3.0,
            y_max=3.0,
        )
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 4, 16
        y = torch.randn(batch_size, seq_len, device=device)

        logits = model(y)

        assert logits.shape == (batch_size, seq_len, 10)

    def test_continuous_point_loss_computation(self) -> None:
        """Test that loss computation works for continuous point prediction."""
        config = UnsupervisedPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_type="continuous",
            prediction_type="point",
        )
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 4, 16
        y = torch.randn(batch_size, seq_len, device=device)

        loss, metrics = compute_loss(model, None, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert "mse" in metrics

    def test_continuous_distribution_loss_computation(self) -> None:
        """Test that loss computation works for continuous distribution prediction."""
        config = UnsupervisedPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_type="continuous",
            prediction_type="distribution",
            bucket_type="uniform",
            y_min=-3.0,
            y_max=3.0,
        )
        model = PFNModel(config)
        device = next(model.parameters()).device

        batch_size, seq_len = 4, 16
        y = torch.randn(batch_size, seq_len, device=device)

        loss, metrics = compute_loss(model, None, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert "loss" in metrics
