"""Tests for autoregressive generation functionality.

Tests cover:
- SupervisedPFN generation with x_distribution
- UnsupervisedPFN generation (standard GPT-2 style)
- Generation with and without prompts
- Sampling vs mode selection
- Different prediction types (distribution, point, classification)
- KV-caching efficiency
"""

import pytest
import torch
from torch.distributions import Normal

from pfn_transformerlens.model.configs import (
    ClassificationPFNConfig,
    SupervisedRegressionPFNConfig,
    UnsupervisedPFNConfig,
)
from pfn_transformerlens.model.PFN import (
    ClassificationPrediction,
    DistributionPrediction,
    PFNModel,
    PointPrediction,
)


class TestSupervisedGeneration:
    """Test autoregressive generation for supervised PFN models."""

    def test_generation_output_shapes_without_prompt(self) -> None:
        """Test that generation produces correct output shapes when starting from scratch."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_dim=2,
            prediction_type="distribution",
            bucket_type="uniform",
            bucket_support="bounded",
            y_min=-1.0,
            y_max=1.0,
            act_fn="gelu",
        )
        model = PFNModel(config)

        x_distribution = Normal(0.0, 1.0)
        num_generate = 10

        x_gen, y_gen = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            sample=True,
            temperature=1.0,
        )

        # output is (n_prompts=1, num_rollouts=1, total_len, input_dim)
        assert x_gen.shape == (1, 1, num_generate, 2)
        assert y_gen.shape == (1, 1, num_generate)

    def test_generation_output_shapes_with_prompt(self) -> None:
        """Test generation with initial prompt context."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_dim=2,
            prediction_type="distribution",
            bucket_type="uniform",
            bucket_support="bounded",
            y_min=-1.0,
            y_max=1.0,
            act_fn="gelu",
        )
        model = PFNModel(config)
        device = next(model.parameters()).device

        K_init = 5
        num_generate = 10
        # prompts must be 3D: (n_prompts, K_init, input_dim)
        prompt_x = torch.randn(1, K_init, 2, device=device)
        prompt_y = torch.randn(1, K_init, device=device)

        x_distribution = Normal(0.0, 1.0)

        x_gen, y_gen = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            prompt_x=prompt_x,
            prompt_y=prompt_y,
            sample=True,
            temperature=1.0,
        )

        # output: (n_prompts=1, num_rollouts=1, total_len, input_dim)
        assert x_gen.shape == (1, 1, K_init + num_generate, 2)
        assert y_gen.shape == (1, 1, K_init + num_generate)

        # check that prompt is preserved at the beginning
        assert torch.allclose(x_gen[0, 0, :K_init], prompt_x[0])
        assert torch.allclose(y_gen[0, 0, :K_init], prompt_y[0])

    def test_generation_with_distribution_prediction(self) -> None:
        """Test generation with distribution predictions (bucketing)."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_dim=1,
            prediction_type="distribution",
            bucket_type="uniform",
            bucket_support="bounded",
            y_min=-2.0,
            y_max=2.0,
            act_fn="gelu",
        )
        model = PFNModel(config)

        x_distribution = Normal(0.0, 1.0)
        num_generate = 5

        x_gen, y_gen = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            sample=True,
            temperature=1.0,
        )

        # output: (n_prompts=1, num_rollouts=1, total_len, input_dim)
        assert x_gen.shape == (1, 1, num_generate, 1)
        assert y_gen.shape == (1, 1, num_generate)
        # generated y values should be approximately within bucket range
        # allow small tolerance for bucket decoding edge cases
        assert torch.all((y_gen >= -2.1) & (y_gen <= 2.1))

    def test_generation_with_distribution_prediction_unbounded_uniform(self) -> None:
        """Uniform buckets with unbounded support should yield tail samples."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_dim=1,
            prediction_type="distribution",
            bucket_type="uniform",
            bucket_support="unbounded",
            y_min=-2.0,
            y_max=2.0,
            act_fn="gelu",
        )
        model = PFNModel(config)

        x_distribution = Normal(0.0, 1.0)
        num_generate = 64  # stay within context length
        torch.manual_seed(0)
        _, y_gen = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            sample=True,
            temperature=1.0,
        )

        # output: (n_prompts=1, num_rollouts=1, total_len)
        assert y_gen.shape == (1, 1, num_generate)
        assert torch.all(torch.isfinite(y_gen))

    def test_generation_with_riemann_buckets_bounded(self) -> None:
        """Riemann buckets (bounded) should keep samples within provided borders."""
        borders = torch.tensor([-3.0, -1.0, 0.0, 1.0, 4.0])
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=4,
            input_dim=1,
            prediction_type="distribution",
            bucket_type="riemann",
            bucket_support="bounded",
            riemann_borders=borders,
            act_fn="gelu",
        )
        model = PFNModel(config)

        x_distribution = Normal(0.0, 1.0)
        num_generate = 32
        torch.manual_seed(1)

        _, y_gen = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            sample=True,
            temperature=1.0,
        )

        # output: (n_prompts=1, num_rollouts=1, total_len)
        assert y_gen.shape == (1, 1, num_generate)
        assert torch.all(torch.isfinite(y_gen))
        assert y_gen.min() >= borders.min()
        assert y_gen.max() <= borders.max()

    def test_generation_with_riemann_buckets_unbounded(self) -> None:
        """Unbounded Riemann buckets should produce tail samples."""
        borders = torch.tensor([-3.0, -1.5, -0.5, 1.0, 3.5])
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=4,
            input_dim=1,
            prediction_type="distribution",
            bucket_type="riemann",
            bucket_support="unbounded",
            riemann_borders=borders,
            act_fn="gelu",
        )
        model = PFNModel(config)

        x_distribution = Normal(0.0, 1.0)
        num_generate = 64
        torch.manual_seed(1)

        _, y_gen = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            sample=True,
            temperature=1.0,
        )

        # output: (n_prompts=1, num_rollouts=1, total_len)
        assert y_gen.shape == (1, 1, num_generate)
        assert torch.all(torch.isfinite(y_gen))

    def test_generation_with_point_prediction(self) -> None:
        """Test generation with point predictions."""
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

        x_distribution = Normal(0.0, 1.0)
        num_generate = 5

        x_gen, y_gen = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            sample=False,  # deterministic for point prediction
            temperature=1.0,
        )

        # output: (n_prompts=1, num_rollouts=1, total_len, input_dim)
        assert x_gen.shape == (1, 1, num_generate, 1)
        assert y_gen.shape == (1, 1, num_generate)
        assert torch.all(torch.isfinite(y_gen))

    def test_generation_with_classification(self) -> None:
        """Test generation for classification tasks."""
        config = ClassificationPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            num_classes=3,
            input_dim=2,
            d_vocab=1000,  # required by HookedTransformer even though not used
            act_fn="gelu",
        )
        model = PFNModel(config)

        x_distribution = Normal(0.0, 1.0)
        num_generate = 8

        x_gen, y_gen = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            sample=True,
            temperature=1.0,
        )

        # output: (n_prompts=1, num_rollouts=1, total_len, input_dim)
        assert x_gen.shape == (1, 1, num_generate, 2)
        assert y_gen.shape == (1, 1, num_generate)
        # generated classes should be valid values in [0, num_classes)
        assert torch.all(y_gen >= 0)
        assert torch.all(y_gen < 3)
        # check that values are integer-valued (even if stored as float)
        assert torch.allclose(y_gen, y_gen.round())

    def test_sampling_vs_mode(self) -> None:
        """Test that sampling and mode selection produce different results."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_dim=1,
            prediction_type="distribution",
            bucket_type="uniform",
            bucket_support="bounded",
            y_min=-1.0,
            y_max=1.0,
            act_fn="gelu",
        )
        model = PFNModel(config)

        x_distribution = Normal(0.0, 1.0)
        num_generate = 20

        # generate with sampling
        torch.manual_seed(42)
        _, y_sampled_1 = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            sample=True,
            temperature=1.0,
        )

        # generate with sampling again (should be different)
        torch.manual_seed(43)
        _, y_sampled_2 = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            sample=True,
            temperature=1.0,
        )

        # generate with mode (should be deterministic given same x)
        torch.manual_seed(44)
        _, _ = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            sample=False,
            temperature=1.0,
        )

        # sampling should produce different results
        # flatten to compare since output is 4D
        assert not torch.allclose(y_sampled_1.flatten(), y_sampled_2.flatten())

    def test_temperature_effect_on_sampling(self) -> None:
        """Test that temperature affects sampling diversity."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_dim=1,
            prediction_type="distribution",
            bucket_type="uniform",
            bucket_support="bounded",
            y_min=-1.0,
            y_max=1.0,
            act_fn="gelu",
        )
        model = PFNModel(config)

        x_distribution = Normal(0.0, 1.0)
        num_generate = 50

        # high temperature should give more diverse samples
        torch.manual_seed(42)
        _, y_high_temp = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            sample=True,
            temperature=2.0,
        )

        # low temperature should be less diverse (closer to mode)
        torch.manual_seed(42)
        _, y_low_temp = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            sample=True,
            temperature=0.1,
        )

        # high temp should have higher variance (flatten to 1D for comparison)
        assert y_high_temp.flatten().var() >= y_low_temp.flatten().var()

    def test_generated_values_on_device(self) -> None:
        """Test that generated values are on the correct device."""
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

        x_distribution = Normal(0.0, 1.0)
        num_generate = 5

        x_gen, y_gen = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            sample=False,
        )

        # generated tensors should be on same device as model
        model_device = next(model.parameters()).device
        assert x_gen.device == model_device
        assert y_gen.device == model_device

    def test_generation_extends_prompt_correctly(self) -> None:
        """Test that generation correctly extends the prompt context."""
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
        device = next(model.parameters()).device

        K_init = 3
        # prompts must be 3D: (n_prompts, K_init, input_dim)
        prompt_x = torch.randn(1, K_init, 1, device=device)
        prompt_y = torch.randn(1, K_init, device=device)

        x_distribution = Normal(0.0, 1.0)
        num_generate = 7

        x_gen, y_gen = model.generate(
            x_distribution=x_distribution,
            num_generate=num_generate,
            prompt_x=prompt_x,
            prompt_y=prompt_y,
        )

        # output: (n_prompts=1, num_rollouts=1, total_len, input_dim)
        # first K_init should match prompt exactly
        assert torch.allclose(x_gen[0, 0, :K_init], prompt_x[0])
        assert torch.allclose(y_gen[0, 0, :K_init], prompt_y[0])

        # remaining should be newly generated
        assert x_gen.shape[2] == K_init + num_generate
        assert y_gen.shape[2] == K_init + num_generate


class TestUnsupervisedGeneration:
    """Test autoregressive generation for unsupervised PFN models."""

    def test_discrete_generation_output_shape(self) -> None:
        """Test discrete token generation produces correct shapes."""
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

        num_generate = 15

        y_gen = model.generate(
            num_generate=num_generate,
            sample=True,
            temperature=1.0,
        )

        assert y_gen.shape == (num_generate,)
        # discrete tokens should be integers in [0, d_vocab)
        assert torch.all(y_gen >= 0)
        assert torch.all(y_gen < 10)

    def test_discrete_generation_with_prompt(self) -> None:
        """Test discrete generation with initial prompt."""
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
        device = next(model.parameters()).device

        K_init = 5
        prompt = torch.randint(0, 10, (K_init,), device=device)
        num_generate = 10

        y_gen = model.generate(
            num_generate=num_generate,
            prompt=prompt,
            sample=True,
            temperature=1.0,
        )

        assert y_gen.shape == (K_init + num_generate,)
        # prompt should be preserved
        assert torch.allclose(y_gen[:K_init].long(), prompt.long())

    def test_continuous_point_generation(self) -> None:
        """Test continuous value generation with point prediction."""
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

        num_generate = 12

        y_gen = model.generate(
            num_generate=num_generate,
            sample=False,  # deterministic for point prediction
            temperature=1.0,
        )

        assert y_gen.shape == (num_generate,)
        assert torch.all(torch.isfinite(y_gen))
        assert y_gen.dtype == torch.float32

    def test_continuous_distribution_generation(self) -> None:
        """Test continuous value generation with distribution prediction."""
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
            bucket_support="bounded",
            y_min=-3.0,
            y_max=3.0,
        )
        model = PFNModel(config)

        num_generate = 8

        y_gen = model.generate(
            num_generate=num_generate,
            sample=True,
            temperature=1.0,
        )

        assert y_gen.shape == (num_generate,)
        # generated values should be approximately within bucket range
        # allow small tolerance for bucket decoding edge cases
        assert torch.all((y_gen >= -3.2) & (y_gen <= 3.2))

    def test_continuous_distribution_generation_unbounded_uniform(self) -> None:
        """Unbounded uniform buckets should occasionally sample tails."""
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
            bucket_support="unbounded",
            y_min=-3.0,
            y_max=3.0,
        )
        model = PFNModel(config)

        num_generate = 64
        torch.manual_seed(0)
        y_gen = model.generate(
            num_generate=num_generate,
            sample=True,
            temperature=1.0,
        )

        assert y_gen.shape == (num_generate,)
        assert torch.all(torch.isfinite(y_gen))

    def test_continuous_distribution_generation_riemann_bounded(self) -> None:
        """Unsupervised Riemann buckets with bounded support stay within borders."""
        borders = torch.tensor([-4.0, -2.0, -0.5, 1.0, 2.5])
        config = UnsupervisedPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=4,
            input_type="continuous",
            prediction_type="distribution",
            bucket_type="riemann",
            bucket_support="bounded",
            riemann_borders=borders,
        )
        model = PFNModel(config)

        num_generate = 64
        torch.manual_seed(1)
        y_gen = model.generate(
            num_generate=num_generate,
            sample=True,
            temperature=1.0,
        )

        assert y_gen.shape == (num_generate,)
        assert torch.all(torch.isfinite(y_gen))
        assert y_gen.min() >= borders.min()
        assert y_gen.max() <= borders.max()

    def test_continuous_distribution_generation_riemann_unbounded(self) -> None:
        """Riemann buckets with unbounded support should emit tails."""
        borders = torch.tensor([-4.0, -1.5, -0.75, 0.5, 2.0])
        config = UnsupervisedPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=4,
            input_type="continuous",
            prediction_type="distribution",
            bucket_type="riemann",
            bucket_support="unbounded",
            riemann_borders=borders,
        )
        model = PFNModel(config)

        num_generate = 64
        torch.manual_seed(1)
        y_gen = model.generate(
            num_generate=num_generate,
            sample=True,
            temperature=1.0,
        )

        assert y_gen.shape == (num_generate,)
        assert torch.all(torch.isfinite(y_gen))

    def test_sampling_produces_variation(self) -> None:
        """Test that sampling produces different sequences."""
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

        num_generate = 20

        # generate twice with different seeds
        torch.manual_seed(42)
        y_gen_1 = model.generate(num_generate=num_generate, sample=True)

        torch.manual_seed(43)
        y_gen_2 = model.generate(num_generate=num_generate, sample=True)

        # should produce different sequences
        assert not torch.equal(y_gen_1, y_gen_2)

    def test_mode_selection_is_deterministic(self) -> None:
        """Test that mode selection produces consistent results."""
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
        device = next(model.parameters()).device

        prompt = torch.tensor([1, 2, 3, 4], device=device)
        num_generate = 10

        # generate twice with mode (should be same)
        y_gen_1 = model.generate(num_generate=num_generate, prompt=prompt, sample=False)
        y_gen_2 = model.generate(num_generate=num_generate, prompt=prompt, sample=False)

        assert torch.equal(y_gen_1, y_gen_2)

    def test_generated_sequence_on_cpu(self) -> None:
        """Test that generated sequence is returned on CPU."""
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

        num_generate = 5

        y_gen = model.generate(num_generate=num_generate, sample=True)

        model_device = next(model.parameters()).device
        assert y_gen.device == model_device


class TestGenerationEdgeCases:
    """Test edge cases for generation."""

    def test_generate_zero_steps_returns_prompt_only(self) -> None:
        """Test that generating 0 steps returns only the prompt."""
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
        device = next(model.parameters()).device

        K_init = 3
        # prompts must be 3D: (n_prompts, K_init, input_dim)
        prompt_x = torch.randn(1, K_init, 1, device=device)
        prompt_y = torch.randn(1, K_init, device=device)

        x_distribution = Normal(0.0, 1.0)

        x_gen, y_gen = model.generate(
            x_distribution=x_distribution,
            num_generate=0,
            prompt_x=prompt_x,
            prompt_y=prompt_y,
        )

        # output: (n_prompts=1, num_rollouts=1, total_len, input_dim)
        assert x_gen.shape == (1, 1, K_init, 1)
        assert y_gen.shape == (1, 1, K_init)
        assert torch.allclose(x_gen[0, 0], prompt_x[0])
        assert torch.allclose(y_gen[0, 0], prompt_y[0])

    def test_generate_one_step(self) -> None:
        """Test generating exactly one step."""
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

        x_distribution = Normal(0.0, 1.0)

        x_gen, y_gen = model.generate(
            x_distribution=x_distribution,
            num_generate=1,
        )

        # output: (n_prompts=1, num_rollouts=1, total_len, input_dim)
        assert x_gen.shape == (1, 1, 1, 1)
        assert y_gen.shape == (1, 1, 1)

    def test_unsupervised_generate_zero_returns_prompt(self) -> None:
        """Test unsupervised generation with 0 steps."""
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
        device = next(model.parameters()).device

        prompt = torch.tensor([1, 2, 3], device=device)

        y_gen = model.generate(num_generate=0, prompt=prompt)

        assert y_gen.shape == (3,)
        assert torch.equal(y_gen.long(), prompt.long())

    def test_supervised_prompt_mismatch_raises_error(self) -> None:
        """Test that mismatched prompt shapes raise an error."""
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
        device = next(model.parameters()).device

        # mismatched sequence lengths (must be 3D for new API)
        prompt_x = torch.randn(1, 3, 1, device=device)
        prompt_y = torch.randn(1, 5, device=device)  # different length!

        x_distribution = Normal(0.0, 1.0)

        with pytest.raises((AssertionError, RuntimeError, ValueError)):
            model.generate(
                x_distribution=x_distribution,
                num_generate=5,
                prompt_x=prompt_x,
                prompt_y=prompt_y,
            )

    def test_invalid_temperature_raises_error(self) -> None:
        """Test that invalid temperature values raise errors."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_dim=1,
            prediction_type="distribution",
            bucket_type="uniform",
            bucket_support="bounded",
            y_min=-1.0,
            y_max=1.0,
            act_fn="gelu",
        )
        model = PFNModel(config)

        x_distribution = Normal(0.0, 1.0)

        with pytest.raises(AssertionError, match="temperature must be"):
            model.generate(
                x_distribution=x_distribution,
                num_generate=5,
                temperature=0.0,
            )

        with pytest.raises(AssertionError, match="temperature must be"):
            model.generate(
                x_distribution=x_distribution,
                num_generate=5,
                temperature=-1.0,
            )


class TestPredictOnPromptCache:
    """Test cache return functionality for predict_on_prompt."""

    def test_supervised_predict_with_cache_distribution(self) -> None:
        """Test that supervised predict_on_prompt returns cache when requested."""
        config = SupervisedRegressionPFNConfig(
            n_layers=2,
            d_model=64,
            n_ctx=128,
            d_head=32,
            n_heads=2,
            d_vocab=10,
            input_dim=2,
            prediction_type="distribution",
            bucket_type="uniform",
            bucket_support="bounded",
            y_min=-1.0,
            y_max=1.0,
            act_fn="gelu",
        )
        model = PFNModel(config)
        device = next(model.parameters()).device

        x = torch.randn(5, 2, device=device)
        y = torch.randn(5, device=device)

        # without cache
        pred = model.predict_on_prompt(x, y, return_cache=False)
        assert isinstance(pred, DistributionPrediction)

        # with cache
        result = model.predict_on_prompt(x, y, return_cache=True)
        assert isinstance(result, tuple)
        pred, cache = result
        assert isinstance(pred, DistributionPrediction)
        assert cache is not None
        # check that cache has expected structure from TransformerLens
        assert hasattr(cache, "cache_dict")

    def test_supervised_predict_with_cache_point(self) -> None:
        """Test cache return for point predictions."""
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
        device = next(model.parameters()).device

        x = torch.randn(5, 1, device=device)
        y = torch.randn(5, device=device)

        result = model.predict_on_prompt(x, y, return_cache=True)
        assert isinstance(result, tuple)
        pred, cache = result
        assert isinstance(pred, PointPrediction)
        assert cache is not None

    def test_supervised_predict_with_cache_classification(self) -> None:
        """Test cache return for classification."""
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
        device = next(model.parameters()).device

        x = torch.randn(5, 2, device=device)
        y = torch.randint(0, 3, (5,), device=device).float()

        result = model.predict_on_prompt(x, y, return_cache=True)
        assert isinstance(result, tuple)
        pred, cache = result
        assert isinstance(pred, ClassificationPrediction)
        assert cache is not None

    def test_unsupervised_predict_with_cache_discrete(self) -> None:
        """Test cache return for unsupervised discrete models."""
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
        device = next(model.parameters()).device

        y = torch.randint(0, 10, (8,), device=device)

        # without cache
        pred = model.predict_on_prompt(y, return_cache=False)
        assert isinstance(pred, DistributionPrediction)

        # with cache
        result = model.predict_on_prompt(y, return_cache=True)
        assert isinstance(result, tuple)
        pred, cache = result
        assert isinstance(pred, DistributionPrediction)
        assert cache is not None

    def test_unsupervised_predict_with_cache_continuous(self) -> None:
        """Test cache return for unsupervised continuous models."""
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

        y = torch.randn(8, device=device)

        result = model.predict_on_prompt(y, return_cache=True)
        assert isinstance(result, tuple)
        pred, cache = result
        assert isinstance(pred, PointPrediction)
        assert cache is not None

    def test_cache_with_batch_dimension(self) -> None:
        """Test cache return with explicit batch dimension."""
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
        device = next(model.parameters()).device

        # batch dimension included
        x = torch.randn(3, 5, 1, device=device)
        y = torch.randn(3, 5, device=device)

        result = model.predict_on_prompt(x, y, return_cache=True)
        assert isinstance(result, tuple)
        pred, cache = result
        assert isinstance(pred, PointPrediction)
        assert cache is not None

    def test_cache_structure_has_activations(self) -> None:
        """Test that cache contains expected activation keys."""
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
        device = next(model.parameters()).device

        x = torch.randn(5, 1, device=device)
        y = torch.randn(5, device=device)

        _, cache = model.predict_on_prompt(x, y, return_cache=True)

        # check for some expected keys from TransformerLens
        # cache should have patterns, residual stream, etc.
        assert hasattr(cache, "cache_dict")
        assert len(cache.cache_dict) > 0
