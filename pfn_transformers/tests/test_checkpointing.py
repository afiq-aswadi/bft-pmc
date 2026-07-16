"""Tests for model checkpointing and loading.

Tests cover:
- Checkpoint saving during training
- Model loading from checkpoint
- State preservation (weights, optimizer state)
- Prediction consistency after loading
"""

import tempfile
from pathlib import Path

import pytest
import torch

from pfn_transformerlens.model.configs import (
    SupervisedRegressionPFNConfig,
    UnsupervisedPFNConfig,
)
from pfn_transformerlens.train import TrainingConfig, train
from pfn_transformerlens.checkpointing import (
    get_logarithmic_checkpoint_steps,
    load_checkpoint,
)


class SimpleRegressionGenerator:
    """Simple generator for regression testing."""

    def __init__(self) -> None:
        self.input_dim = 2

    def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(seq_len, self.input_dim)
        y = x[:, 0] + x[:, 1]  # Simple linear combination
        return x, y


class SimpleUnsupervisedGenerator:
    """Simple generator for unsupervised testing."""

    def __init__(self) -> None:
        self.input_dim = 1

    def generate(self, seq_len: int) -> torch.Tensor:
        return torch.randint(0, 2, (seq_len,))


class TestCheckpointSaving:
    """Test checkpoint saving functionality."""

    def test_checkpoint_is_saved(self) -> None:
        """Test that checkpoints are saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
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
                num_steps=10,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Check that checkpoints exist
            ckpt_dir = Path(tmpdir)
            checkpoints = list(ckpt_dir.glob("*.pt"))
            assert len(checkpoints) > 0

            # Should have checkpoints at steps 5 and 10
            expected_files = ["checkpoint_step_5.pt", "checkpoint_step_10.pt"]
            saved_files = [ckpt.name for ckpt in checkpoints]
            for expected in expected_files:
                assert expected in saved_files

    def test_checkpoint_contains_required_keys(self) -> None:
        """Test that saved checkpoints contain all required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
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
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Load checkpoint and check keys
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            checkpoint = torch.load(ckpt_path, weights_only=False)

            required_keys = [
                "step",
                "model_state_dict",
                "optimizer_state_dict",
                "model_config",
                "training_config",
            ]
            for key in required_keys:
                assert key in checkpoint

    def test_no_checkpoint_when_disabled(self) -> None:
        """Test that checkpoints are not saved when save_checkpoint=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
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
                num_steps=10,
                save_checkpoint=False,  # Disabled
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # No checkpoints should exist
            ckpt_dir = Path(tmpdir)
            checkpoints = list(ckpt_dir.glob("*.pt"))
            assert len(checkpoints) == 0


class TestCheckpointLoading:
    """Test checkpoint loading functionality."""

    def test_load_model_basic(self) -> None:
        """Test basic model loading from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
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
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Load the model
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            loaded_model, _, _ = load_checkpoint(str(ckpt_path))

            from pfn_transformerlens.model.PFN import BasePFN

            assert isinstance(loaded_model, BasePFN)
            assert not loaded_model.training  # Should be in eval mode

    def test_load_model_missing_file(self) -> None:
        """Test that loading nonexistent checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent_checkpoint.pt")

    def test_load_model_weights_match(self) -> None:
        """Test that loaded model has same weights as trained model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
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
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            trained_model = train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Load the model
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            loaded_model, _, _ = load_checkpoint(str(ckpt_path))

            # Compare weights
            for (name1, param1), (name2, param2) in zip(
                trained_model.named_parameters(), loaded_model.named_parameters()
            ):
                assert name1 == name2
                assert torch.allclose(param1, param2)

    def test_load_model_predictions_match(self) -> None:
        """Test that loaded model produces same predictions as trained model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
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
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            trained_model = train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Generate test data
            device = next(trained_model.parameters()).device
            x_test = torch.randn(1, 8, 2, device=device)
            y_test = torch.randn(1, 8, device=device)

            # Get predictions from trained model
            trained_model.eval()
            with torch.no_grad():
                preds_trained = trained_model(x_test, y_test)

            # Load and get predictions from loaded model
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            loaded_model, _, _ = load_checkpoint(str(ckpt_path))

            with torch.no_grad():
                preds_loaded = loaded_model(x_test, y_test)

            # Predictions should match
            assert torch.allclose(preds_trained, preds_loaded)

    def test_load_model_device_auto(self) -> None:
        """Test automatic device selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
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
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Load with auto device
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            loaded_model, _, _ = load_checkpoint(str(ckpt_path), device="auto")

            # Should be on some device
            device = next(loaded_model.parameters()).device
            assert device.type in ["cuda", "mps", "cpu"]

    def test_load_model_device_cpu(self) -> None:
        """Test loading to CPU device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
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
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Load to CPU
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            checkpoint = torch.load(ckpt_path, weights_only=False)
            checkpoint["model_config"].device = "mps"
            torch.save(checkpoint, ckpt_path)
            loaded_model, _, _ = load_checkpoint(str(ckpt_path), device="cpu")

            device = next(loaded_model.parameters()).device
            assert device.type == "cpu"
            assert loaded_model.config.device == "cpu"
            assert loaded_model.transformer.cfg.device == "cpu"
            x = torch.randn(1, 4, 2)
            y = torch.randn(1, 4)
            with torch.no_grad():
                loaded_model(x, y)


class TestCheckpointingUnsupervisedModels:
    """Test checkpointing with unsupervised models."""

    def test_unsupervised_model_checkpoint_and_load(self) -> None:
        """Test checkpoint save/load with unsupervised models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = UnsupervisedPFNConfig(
                n_layers=1,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=2,
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

            generator = SimpleUnsupervisedGenerator()
            trained_model = train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # Load the model
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            loaded_model, _, _ = load_checkpoint(str(ckpt_path))

            # Test predictions match
            device = next(trained_model.parameters()).device
            y_test = torch.randint(0, 2, (1, 8), device=device)

            trained_model.eval()
            with torch.no_grad():
                preds_trained = trained_model(y_test)

            with torch.no_grad():
                preds_loaded = loaded_model(y_test)

            assert torch.allclose(preds_trained, preds_loaded)


class TestResumeTraining:
    """Test resuming training from checkpoint."""

    def test_resume_from_checkpoint(self) -> None:
        """Test that training can be resumed from a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
                d_model=32,
                n_ctx=64,
                d_head=16,
                n_heads=2,
                d_vocab=10,
                input_dim=2,
                prediction_type="point",
                act_fn="gelu",
            )

            # First training run
            training_config_1 = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=5,
                save_checkpoint=True,
                save_every=5,
                checkpoint_dir=tmpdir,
                log_every=5,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config_1,
            )

            # Resume training
            ckpt_path = Path(tmpdir) / "checkpoint_step_5.pt"
            training_config_2 = TrainingConfig(
                batch_size=4,
                seq_len=8,
                num_steps=10,  # Continue to step 10
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

            # Model should have continued training
            assert model_2 is not None

            # Checkpoint for step 10 should exist
            ckpt_path_10 = Path(tmpdir) / "checkpoint_step_10.pt"
            assert ckpt_path_10.exists()


class TestLogarithmicCheckpointing:
    """Test logarithmic checkpoint scheduling from https://arxiv.org/pdf/2501.17745."""

    def test_paper_settings_checkpoint_count(self) -> None:
        """Test checkpoint count matches paper settings (T=150k, N=1000, interval=100)."""
        T = 150000
        N = 1000
        interval = 100

        steps = get_logarithmic_checkpoint_steps(T, N, interval)

        # C_linear = {0, 100, 200, ..., 150000} gives 1501 steps
        # C_log = 1000 logarithmically spaced checkpoints
        # union should have ~2203 total (some overlap between sets)
        # paper reports |C| = 2203
        assert len(steps) >= 2150  # allow small variance due to implementation details
        assert len(steps) <= 2250

    def test_endpoints_included(self) -> None:
        """Test that first and last steps are always included."""
        steps = get_logarithmic_checkpoint_steps(150000, 1000, 100)
        assert 0 in steps
        assert 150000 in steps

    def test_steps_sorted(self) -> None:
        """Test that checkpoint steps are returned in sorted order."""
        steps = get_logarithmic_checkpoint_steps(150000, 1000, 100)
        assert steps == sorted(steps)

    def test_no_duplicates(self) -> None:
        """Test that there are no duplicate checkpoint steps."""
        steps = get_logarithmic_checkpoint_steps(150000, 1000, 100)
        assert len(steps) == len(set(steps))

    def test_early_steps_are_dense(self) -> None:
        """Test that early checkpoints are densely spaced (logarithmic property)."""
        steps = get_logarithmic_checkpoint_steps(150000, 1000, 100)

        # first 100 steps should have many checkpoints due to logarithmic spacing
        early_checkpoints = [s for s in steps if s < 100]
        # should have at least 50 checkpoints in first 100 steps
        assert len(early_checkpoints) >= 50

    def test_later_steps_follow_linear_interval(self) -> None:
        """Test that later checkpoints follow the linear interval."""
        T = 150000
        interval = 100
        steps = get_logarithmic_checkpoint_steps(T, 1000, interval)

        # check that all linear interval checkpoints are present in later range
        # e.g., {149000, 149100, 149200, ..., 149900, 150000} should all be in steps
        for step in range(149000, 150000 + interval, interval):
            assert step in steps

    def test_edge_case_small_T(self) -> None:
        """Test behavior with small T values."""
        steps = get_logarithmic_checkpoint_steps(10, 5, 2)
        assert 0 in steps
        assert 10 in steps
        assert all(0 <= s <= 10 for s in steps)

    def test_edge_case_N_equals_1(self) -> None:
        """Test behavior when n_log_checkpoints=1."""
        steps = get_logarithmic_checkpoint_steps(1000, 1, 100)
        # should still have linear checkpoints
        assert len(steps) >= 10  # at least {0, 100, 200, ..., 900, 999}

    def test_edge_case_T_equals_0(self) -> None:
        """Test behavior when T=0."""
        steps = get_logarithmic_checkpoint_steps(0, 1000, 100)
        assert steps == [0]

    def test_invalid_parameters_raise(self) -> None:
        """Test that invalid configuration raises ValueError."""
        with pytest.raises(ValueError):
            get_logarithmic_checkpoint_steps(
                training_steps=-1, n_log_checkpoints=10, linear_interval=5
            )
        with pytest.raises(ValueError):
            get_logarithmic_checkpoint_steps(
                training_steps=10, n_log_checkpoints=0, linear_interval=5
            )
        with pytest.raises(ValueError):
            get_logarithmic_checkpoint_steps(
                training_steps=10, n_log_checkpoints=10, linear_interval=0
            )

    def test_all_steps_in_valid_range(self) -> None:
        """Test that all checkpoint steps are in [0, T]."""
        T = 150000
        steps = get_logarithmic_checkpoint_steps(T, 1000, 100)
        assert all(0 <= s <= T for s in steps)

    def test_logarithmic_schedule_during_training(self) -> None:
        """Test that training with logarithmic schedule saves at correct steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
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
                num_steps=100,
                save_checkpoint=True,
                checkpoint_schedule="logarithmic",
                linear_checkpoint_interval=20,
                n_log_checkpoints=10,
                checkpoint_dir=tmpdir,
                log_every=50,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # compute expected checkpoints
            expected_steps = get_logarithmic_checkpoint_steps(
                training_steps=100,
                n_log_checkpoints=10,
                linear_interval=20,
            )

            # check that checkpoints exist for expected steps
            ckpt_dir = Path(tmpdir)
            saved_files = {ckpt.name for ckpt in ckpt_dir.glob("*.pt")}

            # verify that the expected number of checkpoints were saved
            # note: step 0 in the list won't be saved since training loop saves at step+1
            # so we check that we have roughly the right number of checkpoints
            assert len(saved_files) == len(expected_steps) - 1 or len(
                saved_files
            ) == len(expected_steps)

            # check that key checkpoints exist (excluding step 0)
            for step in expected_steps:
                if (
                    step > 0
                ):  # skip step 0 since training loop starts at step=0, saves at step+1
                    expected_file = f"checkpoint_step_{step}.pt"
                    assert expected_file in saved_files

    def test_linear_schedule_backwards_compatible(self) -> None:
        """Test that 'linear' schedule maintains backwards compatibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SupervisedRegressionPFNConfig(
                n_layers=1,
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
                save_checkpoint=True,
                checkpoint_schedule="linear",  # use old behavior
                save_every=10,
                checkpoint_dir=tmpdir,
                log_every=10,
            )

            generator = SimpleRegressionGenerator()
            train(
                data_generator=generator,
                model_config=config,
                training_config=training_config,
            )

            # should save at steps 10 and 20 (every save_every steps)
            ckpt_dir = Path(tmpdir)
            saved_files = {ckpt.name for ckpt in ckpt_dir.glob("*.pt")}

            assert "checkpoint_step_10.pt" in saved_files
            assert "checkpoint_step_20.pt" in saved_files
            # should only have 2 checkpoints
            assert len(saved_files) == 2
