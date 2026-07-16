"""Integration tests for end-to-end PFN workflows.

Tests cover full pipelines from data generation through training to inference and evaluation.
These tests verify that all components work together correctly across:
- Supervised regression (point and distribution predictions)
- Classification
- Unsupervised learning (discrete and continuous)
- Checkpoint save/load/resume workflows
- Device placement across entire pipeline
- Gradient propagation (especially positional embeddings)
"""

import tempfile
from pathlib import Path

import pytest
import torch

from pfn_transformerlens.checkpointing import _get_git_hash, load_checkpoint
from pfn_transformerlens.model.configs import (
    ClassificationPFNConfig,
    SupervisedRegressionPFNConfig,
    UnsupervisedPFNConfig,
)
from pfn_transformerlens.model.PFN import PFNModel
from pfn_transformerlens.train import TrainingConfig, compute_loss, train


class SimpleRegressionGenerator:
    """Simple deterministic generator for supervised regression testing."""

    def __init__(self) -> None:
        self.input_dim = 2

    def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(seq_len, self.input_dim)
        y = x[:, 0] + x[:, 1]
        return x, y


class SimpleClassificationGenerator:
    """Simple generator for classification testing."""

    def __init__(self, num_classes: int = 3) -> None:
        self.input_dim = 4
        self.num_classes = num_classes

    def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(seq_len, self.input_dim)
        y = torch.randint(0, self.num_classes, (seq_len,))
        return x, y


class SimpleUnsupervisedDiscreteGenerator:
    """Simple generator for unsupervised discrete (token) prediction."""

    def __init__(self, vocab_size: int = 10) -> None:
        self.input_dim = 1
        self.vocab_size = vocab_size

    def generate(self, seq_len: int) -> torch.Tensor:
        return torch.randint(0, self.vocab_size, (seq_len,))


class SimpleUnsupervisedContinuousGenerator:
    """Simple generator for unsupervised continuous value prediction."""

    def __init__(self) -> None:
        self.input_dim = 1

    def generate(self, seq_len: int) -> torch.Tensor:
        return torch.randn(seq_len)


class TestSupervisedRegressionPipeline:
    """Test full supervised regression pipeline: generate → train → infer → evaluate."""

    def test_supervised_regression_point_pipeline(self) -> None:
        """Test end-to-end supervised regression with point predictions."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=32,
            n_ctx=64,
            d_head=16,
            n_heads=2,
            d_vocab=10,
            input_dim=2,
            prediction_type="point",
            act_fn="gelu",
        )

        training_config = TrainingConfig(
            batch_size=4,
            seq_len=8,
            num_steps=20,
            save_checkpoint=False,
            log_every=10,
        )

        generator = SimpleRegressionGenerator()

        # step 1: train
        model = train(
            data_generator=generator,
            model_config=config,
            training_config=training_config,
        )

        # step 2: get model device
        device = next(model.parameters()).device

        # step 3: inference - test predict_on_prompt
        x_test, y_test = generator.generate(seq_len=8)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        model.eval()
        with torch.no_grad():
            prediction = model.predict_on_prompt(x_test, y_test)

        # step 4: verify prediction structure
        assert hasattr(prediction, "preds")
        # point predictions can be (seq_len,) or (seq_len, 1)
        assert prediction.preds.shape in [(8,), (8, 1)]

        # step 5: evaluate - compute loss
        x_batch = x_test.unsqueeze(0)
        y_batch = y_test.unsqueeze(0)
        loss, metrics = compute_loss(model, x_batch, y_batch)

        # step 6: verify metrics
        assert "loss" in metrics
        assert "mse" in metrics
        assert metrics["loss"] > 0
        assert metrics["mse"] >= 0

    def test_supervised_regression_distribution_pipeline(self) -> None:
        """Test end-to-end supervised regression with distribution predictions."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=32,
            n_ctx=64,
            d_head=16,
            n_heads=2,
            d_vocab=32,
            input_dim=2,
            prediction_type="distribution",
            bucket_type="uniform",
            y_min=-5.0,
            y_max=5.0,
            act_fn="gelu",
        )

        training_config = TrainingConfig(
            batch_size=4,
            seq_len=8,
            num_steps=20,
            save_checkpoint=False,
            log_every=10,
        )

        generator = SimpleRegressionGenerator()

        # step 1: train
        model = train(
            data_generator=generator,
            model_config=config,
            training_config=training_config,
        )

        device = next(model.parameters()).device

        # step 2: inference
        x_test, y_test = generator.generate(seq_len=8)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        model.eval()
        with torch.no_grad():
            prediction = model.predict_on_prompt(x_test, y_test)

        # step 3: verify distribution prediction structure
        assert hasattr(prediction, "probs")
        assert hasattr(prediction, "y_grid")
        assert prediction.probs.shape == (8, 32)
        assert prediction.y_grid.shape == (32,)
        assert torch.allclose(prediction.probs.sum(dim=-1), torch.ones(8), atol=1e-5)

        # step 4: evaluate
        x_batch = x_test.unsqueeze(0)
        y_batch = y_test.unsqueeze(0)
        loss, metrics = compute_loss(model, x_batch, y_batch)

        assert "loss" in metrics
        assert metrics["loss"] > 0


class TestClassificationPipeline:
    """Test full classification pipeline: generate → train → infer → evaluate."""

    def test_classification_pipeline(self) -> None:
        """Test end-to-end classification pipeline."""
        num_classes = 3
        config = ClassificationPFNConfig(
            n_layers=2,
            d_model=32,
            n_ctx=64,
            d_head=16,
            n_heads=2,
            num_classes=num_classes,
            input_dim=4,
            d_vocab=1000,
            act_fn="gelu",
        )

        training_config = TrainingConfig(
            batch_size=4,
            seq_len=8,
            num_steps=20,
            save_checkpoint=False,
            log_every=10,
        )

        generator = SimpleClassificationGenerator(num_classes=num_classes)

        # step 1: train
        model = train(
            data_generator=generator,
            model_config=config,
            training_config=training_config,
        )

        device = next(model.parameters()).device

        # step 2: inference
        x_test, y_test = generator.generate(seq_len=8)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        model.eval()
        with torch.no_grad():
            prediction = model.predict_on_prompt(x_test, y_test)

        # step 3: verify classification prediction structure
        assert hasattr(prediction, "probs")
        assert prediction.probs.shape == (8, num_classes)
        assert torch.allclose(prediction.probs.sum(dim=-1), torch.ones(8), atol=1e-5)

        # step 4: get predicted classes
        predicted_classes = prediction.probs.argmax(dim=-1)
        assert predicted_classes.shape == (8,)
        assert (predicted_classes >= 0).all()
        assert (predicted_classes < num_classes).all()

        # step 5: evaluate
        x_batch = x_test.unsqueeze(0)
        y_batch = y_test.unsqueeze(0)
        loss, metrics = compute_loss(model, x_batch, y_batch)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] > 0
        assert 0 <= metrics["accuracy"] <= 1


class TestUnsupervisedPipeline:
    """Test full unsupervised pipeline: generate → train → infer → evaluate."""

    def test_unsupervised_discrete_pipeline(self) -> None:
        """Test end-to-end unsupervised discrete (token prediction) pipeline."""
        vocab_size = 10
        config = UnsupervisedPFNConfig(
            n_layers=2,
            d_model=32,
            n_ctx=64,
            d_head=16,
            n_heads=2,
            d_vocab=vocab_size,
            input_type="discrete",
            prediction_type="distribution",
        )

        training_config = TrainingConfig(
            batch_size=4,
            seq_len=8,
            num_steps=20,
            save_checkpoint=False,
            log_every=10,
        )

        generator = SimpleUnsupervisedDiscreteGenerator(vocab_size=vocab_size)

        # step 1: train
        model = train(
            data_generator=generator,
            model_config=config,
            training_config=training_config,
        )

        device = next(model.parameters()).device

        # step 2: inference
        y_test = generator.generate(seq_len=8)
        y_test = y_test.to(device)

        model.eval()
        with torch.no_grad():
            prediction = model.predict_on_prompt(y_test)

        # step 3: verify prediction structure
        assert hasattr(prediction, "probs")
        assert prediction.probs.shape == (8, vocab_size)
        assert torch.allclose(prediction.probs.sum(dim=-1), torch.ones(8), atol=1e-5)

        # step 4: test generation
        with torch.no_grad():
            generated = model.generate(
                num_generate=5,
                prompt=y_test[:3],
                sample=False,
                temperature=1.0,
            )
        assert generated.shape == (8,)
        assert (generated >= 0).all()
        assert (generated < vocab_size).all()

        # step 5: evaluate
        y_batch = y_test.unsqueeze(0)
        loss, metrics = compute_loss(model, None, y_batch)

        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_unsupervised_continuous_pipeline(self) -> None:
        """Test end-to-end unsupervised continuous value prediction pipeline."""
        config = UnsupervisedPFNConfig(
            n_layers=2,
            d_model=32,
            n_ctx=64,
            d_head=16,
            n_heads=2,
            d_vocab=32,
            input_type="continuous",
            prediction_type="point",
        )

        training_config = TrainingConfig(
            batch_size=4,
            seq_len=8,
            num_steps=20,
            save_checkpoint=False,
            log_every=10,
        )

        generator = SimpleUnsupervisedContinuousGenerator()

        # step 1: train
        model = train(
            data_generator=generator,
            model_config=config,
            training_config=training_config,
        )

        device = next(model.parameters()).device

        # step 2: inference
        y_test = generator.generate(seq_len=8)
        y_test = y_test.to(device)

        model.eval()
        with torch.no_grad():
            prediction = model.predict_on_prompt(y_test)

        # step 3: verify point prediction structure
        assert hasattr(prediction, "preds")
        # unsupervised continuous point predictions have shape (seq_len, 1)
        assert prediction.preds.shape in [(8,), (8, 1)]

        # step 4: test generation
        with torch.no_grad():
            generated = model.generate(
                num_generate=5,
                prompt=y_test[:3],
                sample=False,
                temperature=1.0,
            )
        assert generated.shape == (8,)

        # step 5: evaluate
        y_batch = y_test.unsqueeze(0)
        loss, metrics = compute_loss(model, None, y_batch)

        assert "loss" in metrics
        assert "mse" in metrics
        assert metrics["loss"] > 0


class TestCheckpointLifecycle:
    """Test checkpoint lifecycle: train → save → load → resume → verify."""

    def test_checkpoint_save_load_resume_pipeline(self) -> None:
        """Test full checkpoint workflow from training through resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=2,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=10,
                input_dim=2,
                prediction_type="point",
                act_fn="gelu",
            )

            generator = SimpleRegressionGenerator()

            # step 1: train and save checkpoint
            training_config_1 = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=10,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            model_1 = train(
                data_generator=generator,
                model_config=config,
                training_config=training_config_1,
            )

            # step 2: verify checkpoint exists
            ckpt_path = Path(tmpdir) / "checkpoint_step_10.pt"
            assert ckpt_path.exists()

            # step 3: load checkpoint and verify model
            loaded_model, optimizer_state, metadata = load_checkpoint(
                str(ckpt_path), load_optimizer=True
            )
            assert optimizer_state is not None
            assert metadata.git_hash == _get_git_hash()

            # step 4: verify predictions match
            device = next(model_1.parameters()).device
            x_test, y_test = generator.generate(seq_len=8)
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            model_1.eval()
            loaded_model.eval()

            with torch.no_grad():
                pred_1 = model_1(x_test.unsqueeze(0), y_test.unsqueeze(0))
                pred_loaded = loaded_model(x_test.unsqueeze(0), y_test.unsqueeze(0))

            assert torch.allclose(pred_1, pred_loaded, atol=1e-5)

            # step 5: resume training from checkpoint
            training_config_2 = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=20,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            model_2 = train(
                data_generator=generator,
                model_config=config,
                training_config=training_config_2,
                resume_from=str(ckpt_path),
            )

            # step 6: verify resumed training continued
            ckpt_path_20 = Path(tmpdir) / "checkpoint_step_20.pt"
            assert ckpt_path_20.exists()

            # step 7: verify resumed model produces different predictions than original
            model_2.eval()
            with torch.no_grad():
                pred_2 = model_2(x_test.unsqueeze(0), y_test.unsqueeze(0))

            # predictions should differ since model continued training
            assert not torch.allclose(pred_1, pred_2, atol=1e-5)

    def test_checkpoint_works_across_model_types(self) -> None:
        """Test checkpoint save/load for different model types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # test unsupervised model
            config = UnsupervisedPFNConfig(
                n_layers=1,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=10,
            )

            training_config = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleUnsupervisedDiscreteGenerator(vocab_size=10)
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # verify checkpoint can be loaded
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            loaded_model, _, _ = load_checkpoint(str(ckpt_path))

            # verify loaded model works
            device = next(loaded_model.parameters()).device
            y_test = generator.generate(seq_len=8).to(device)

            loaded_model.eval()
            with torch.no_grad():
                prediction = loaded_model(y_test.unsqueeze(0))

            assert prediction.shape == (1, 8, 10)


class TestDevicePlacementAcrossPipeline:
    """Test that device placement works correctly across entire training pipeline."""

    @pytest.mark.skipif(
        not (torch.cuda.is_available() or torch.backends.mps.is_available()),
        reason="requires GPU or MPS",
    )
    def test_device_placement_training_to_inference(self) -> None:
        """Test device consistency from training through inference."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=32,
            n_ctx=64,
            d_head=16,
            n_heads=2,
            d_vocab=10,
            input_dim=2,
            prediction_type="point",
            act_fn="gelu",
        )

        # specify device for training
        if torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cuda"

        training_config = TrainingConfig(
            batch_size=4,
            seq_len=8,
            num_steps=10,
            save_checkpoint=False,
            log_every=5,
            device=device_str,
        )

        generator = SimpleRegressionGenerator()

        # step 1: train on specified device
        model = train(
            data_generator=generator,
            model_config=config,
            training_config=training_config,
        )

        # step 2: verify model is on correct device
        device = next(model.parameters()).device
        assert device.type == device_str

        # step 3: test inference with inputs on correct device
        x_test, y_test = generator.generate(seq_len=8)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        model.eval()
        with torch.no_grad():
            prediction = model.predict_on_prompt(x_test, y_test)

        # step 4: verify prediction works without device errors
        assert prediction.preds.shape in [(8,), (8, 1)]

    @pytest.mark.skipif(
        not (torch.cuda.is_available() or torch.backends.mps.is_available()),
        reason="requires GPU or MPS",
    )
    def test_checkpoint_preserves_device_placement(self) -> None:
        """Test that device placement is preserved across checkpoint save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=2,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=10,
                input_dim=2,
                prediction_type="point",
                act_fn="gelu",
            )

            if torch.backends.mps.is_available():
                device_str = "mps"
            else:
                device_str = "cuda"

            training_config = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
                device=device_str,
            )

            generator = SimpleRegressionGenerator()

            # train on GPU
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # load checkpoint with auto device
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            loaded_model, _, _ = load_checkpoint(str(ckpt_path), device="auto")

            # verify loaded model is on a GPU device
            device = next(loaded_model.parameters()).device
            assert device.type in ["cuda", "mps"]


class TestGradientPropagation:
    """Test that gradients propagate correctly through all model components."""

    def test_positional_embeddings_receive_gradients(self) -> None:
        """Test that positional embeddings receive gradients during training.

        This test specifically checks the bug where positional embeddings
        might not receive gradients due to improper setup.
        """
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=32,
            n_ctx=64,
            d_head=16,
            n_heads=2,
            d_vocab=10,
            input_dim=2,
            prediction_type="point",
            use_pos_emb=True,
            act_fn="gelu",
        )

        model = PFNModel(config)
        model.train()

        device = next(model.parameters()).device
        generator = SimpleRegressionGenerator()

        # create training batch
        x, y = generator.generate(seq_len=16)
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        # forward pass
        output = model(x, y)

        # compute loss
        loss = output.mean()

        # backward pass
        loss.backward()

        # check that positional embeddings have gradients
        assert model.transformer.pos_embed.W_pos.grad is not None
        assert not torch.all(model.transformer.pos_embed.W_pos.grad == 0)

        # check that input projection has gradients
        assert model.input_proj.weight.grad is not None
        assert not torch.all(model.input_proj.weight.grad == 0)

        # check that attention weights have gradients
        for block in model.transformer.blocks:
            assert block.attn.W_Q.grad is not None
            assert block.attn.W_K.grad is not None
            assert block.attn.W_V.grad is not None
            assert not torch.all(block.attn.W_Q.grad == 0)
            assert not torch.all(block.attn.W_K.grad == 0)
            assert not torch.all(block.attn.W_V.grad == 0)

    def test_gradient_flow_without_positional_embeddings(self) -> None:
        """Test gradient flow when positional embeddings are disabled."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=32,
            n_ctx=64,
            d_head=16,
            n_heads=2,
            d_vocab=10,
            input_dim=2,
            prediction_type="point",
            use_pos_emb=False,
            act_fn="gelu",
        )

        model = PFNModel(config)
        model.train()

        device = next(model.parameters()).device
        generator = SimpleRegressionGenerator()

        x, y = generator.generate(seq_len=16)
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        output = model(x, y)
        loss = output.mean()
        loss.backward()

        # input projection should still have gradients
        assert model.input_proj.weight.grad is not None
        assert not torch.all(model.input_proj.weight.grad == 0)

        # attention should still have gradients
        for block in model.transformer.blocks:
            assert block.attn.W_Q.grad is not None
            assert not torch.all(block.attn.W_Q.grad == 0)

    def test_gradient_magnitude_is_reasonable(self) -> None:
        """Test that gradient magnitudes are reasonable (not exploding/vanishing)."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=32,
            n_ctx=64,
            d_head=16,
            n_heads=2,
            d_vocab=10,
            input_dim=2,
            prediction_type="point",
            use_pos_emb=True,
            act_fn="gelu",
        )

        model = PFNModel(config)
        model.train()

        device = next(model.parameters()).device
        generator = SimpleRegressionGenerator()

        x, y = generator.generate(seq_len=16)
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        output = model(x, y)
        loss = output.mean()
        loss.backward()

        # collect gradient norms
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

        # check gradients are not exploding (< 100)
        assert all(grad_norm < 100 for grad_norm in grad_norms)
        # check most gradients are not vanishing (allow some small gradients)
        large_gradients = [g for g in grad_norms if g > 1e-6]
        assert len(large_gradients) > len(grad_norms) * 0.8

    def test_all_model_parameters_receive_gradients(self) -> None:
        """Test that all trainable parameters receive gradients."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=32,
            n_ctx=64,
            d_head=16,
            n_heads=2,
            d_vocab=10,
            input_dim=2,
            prediction_type="point",
            use_pos_emb=True,
            act_fn="gelu",
        )

        model = PFNModel(config)
        model.train()

        device = next(model.parameters()).device
        generator = SimpleRegressionGenerator()

        x, y = generator.generate(seq_len=16)
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        output = model(x, y)
        loss = output.mean()
        loss.backward()

        # check all parameters have gradients (except embedding layer for supervised models)
        parameters_without_gradients = []
        for name, param in model.named_parameters():
            # skip embedding layer - supervised models bypass it via input_proj
            if "transformer.embed.W_E" in name:
                continue
            if param.requires_grad and param.grad is None:
                parameters_without_gradients.append(name)

        assert len(parameters_without_gradients) == 0, (
            f"parameters without gradients: {parameters_without_gradients}"
        )

        # check no gradients are all zeros (except embedding layer)
        parameters_with_zero_gradients = []
        for name, param in model.named_parameters():
            # skip embedding layer
            if "transformer.embed.W_E" in name:
                continue
            if param.requires_grad and param.grad is not None:
                if torch.all(param.grad == 0):
                    parameters_with_zero_gradients.append(name)

        assert len(parameters_with_zero_gradients) == 0, (
            f"parameters with zero gradients: {parameters_with_zero_gradients}"
        )
