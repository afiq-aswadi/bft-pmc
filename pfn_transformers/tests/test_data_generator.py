"""Tests for data generator protocols and implementations.

Tests cover:
- SupervisedDataGenerator protocol compliance
- UnsupervisedDataGenerator protocol compliance
- Sampler handling of both generator types
- Training loop integration with both generator types
- Type safety and runtime checks
"""

import pytest
import torch

from pfn_transformerlens.model.configs import (
    SupervisedRegressionPFNConfig,
    UnsupervisedPFNConfig,
)
from pfn_transformerlens.model.PFN import PFNModel
from pfn_transformerlens.sampler.data_generator import (
    DataGenerator,
    DeterministicFunctionGenerator,
    FixedDatasetGenerator,
    ProbabilisticGenerator,
    SupervisedDataGenerator,
    UnsupervisedDataGenerator,
    UnsupervisedProbabilisticGenerator,
)
from pfn_transformerlens.sampler.dataloader import build_dataloader, sample_batch
from pfn_transformerlens.sampler.prior_likelihood import (
    LikelihoodDistribution,
    PriorDistribution,
)
from pfn_transformerlens.sampler.sampler import Sampler
from pfn_transformerlens.train import TrainingConfig, compute_loss, train


def assert_device_matches(actual: torch.device, expected: torch.device) -> None:
    """Accept the explicit index PyTorch attaches to default accelerator devices."""
    assert actual.type == expected.type
    if expected.index is not None:
        assert actual.index == expected.index


class TestSupervisedDataGeneratorProtocol:
    """Test SupervisedDataGenerator protocol compliance."""

    def _target_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def test_simple_supervised_generator(self) -> None:
        """Test that a simple supervised generator satisfies the protocol."""

        class SimpleSupGen:
            def __init__(self) -> None:
                self.input_dim = 2

            def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
                x = torch.randn(seq_len, self.input_dim)
                y = torch.randn(seq_len)
                return x, y

        gen = SimpleSupGen()
        assert isinstance(gen, SupervisedDataGenerator)

        x, y = gen.generate(10)
        assert x.shape == (10, 2)
        assert y.shape == (10,)

    def test_probabilistic_generator_is_supervised(self) -> None:
        """Test that ProbabilisticGenerator satisfies SupervisedDataGenerator."""
        prior = PriorDistribution(torch.distributions.Normal(0.0, 1.0))

        def linear_parameterizer(theta: torch.Tensor, x: torch.Tensor) -> dict:
            return {
                "loc": x.squeeze(-1) * theta,
                "scale": torch.ones_like(x.squeeze(-1)) * 0.1,
            }

        likelihood = LikelihoodDistribution(
            base_distribution=torch.distributions.Normal(0.0, 1.0),
            parameterizer=linear_parameterizer,
            input_dim=1,
        )

        gen = ProbabilisticGenerator(prior, likelihood)
        assert isinstance(gen, SupervisedDataGenerator)

        x, y = gen.generate(20)
        assert x.shape == (20, 1)
        assert y.shape == (20,)

    def test_deterministic_function_generator_is_supervised(self) -> None:
        """Test that DeterministicFunctionGenerator satisfies SupervisedDataGenerator."""

        def linear_fn(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            return (x * theta).sum(dim=-1)

        prior = torch.distributions.Normal(0.0, 1.0)
        gen = DeterministicFunctionGenerator(
            prior=prior,
            function=linear_fn,
            input_dim=2,
            noise_std=0.1,
        )
        assert isinstance(gen, SupervisedDataGenerator)

        x, y = gen.generate(15)
        assert x.shape == (15, 2)
        assert y.shape == (15,)

    def test_deterministic_generator_tracks_prior_device(self) -> None:
        """Ensure generator moves x to the prior's device."""

        target_device = self._target_device()

        class DevicePrior:
            def __init__(self, input_dim: int) -> None:
                self.device = target_device
                self.input_dim = input_dim

            def sample(self) -> torch.Tensor:
                return torch.randn(self.input_dim, device=self.device)

        capture: dict[str, torch.device] = {}

        def linear_fn(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            capture["x_device"] = x.device
            capture["theta_device"] = theta.device
            return (x * theta).sum(dim=-1)

        prior = DevicePrior(input_dim=2)
        gen = DeterministicFunctionGenerator(
            prior=prior,
            function=linear_fn,
            input_dim=2,
        )

        gen.generate(4)

        assert_device_matches(capture["x_device"], target_device)
        assert_device_matches(capture["theta_device"], target_device)

    def test_deterministic_generator_explicit_device(self) -> None:
        """Explicit device argument should move CPU priors to the target device."""

        target_device = self._target_device()

        class CPUPrior:
            def sample(self) -> torch.Tensor:
                return torch.randn(2)

        capture: dict[str, torch.device] = {}

        def linear_fn(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            capture["x_device"] = x.device
            capture["theta_device"] = theta.device
            return (x * theta).sum(dim=-1)

        gen = DeterministicFunctionGenerator(
            prior=CPUPrior(),
            function=linear_fn,
            input_dim=2,
            device=target_device,
        )

        gen.generate(3)

        assert_device_matches(capture["x_device"], target_device)
        assert_device_matches(capture["theta_device"], target_device)

    def test_deterministic_generator_nested_params(self) -> None:
        """Nested parameter structures should also be moved to the target device."""

        target_device = self._target_device()

        class NestedPrior:
            def sample(self) -> dict[str, torch.Tensor]:
                return {
                    "w": torch.randn(2),
                    "b": torch.randn(1),
                }

        capture: dict[str, torch.device] = {}

        def affine_fn(x: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
            capture["w_device"] = params["w"].device
            capture["b_device"] = params["b"].device
            return (x * params["w"]).sum(dim=-1) + params["b"]

        gen = DeterministicFunctionGenerator(
            prior=NestedPrior(),
            function=affine_fn,
            input_dim=2,
            device=target_device,
        )

        (_, _), info = gen.generate_with_params(5)

        assert_device_matches(capture["w_device"], target_device)
        assert_device_matches(capture["b_device"], target_device)
        assert_device_matches(info["params"]["w"].device, target_device)

    def test_fixed_dataset_generator_is_supervised(self) -> None:
        """Test that FixedDatasetGenerator satisfies SupervisedDataGenerator."""
        x_data = torch.randn(100, 3)
        y_data = torch.randn(100)

        gen = FixedDatasetGenerator(x_data, y_data)
        assert isinstance(gen, SupervisedDataGenerator)

        x, y = gen.generate(10)
        assert x.shape == (10, 3)
        assert y.shape == (10,)


class TestFixedDatasetGenerator:
    """Comprehensive tests for FixedDatasetGenerator."""

    def test_sequential_sampling_consecutive_indices(self) -> None:
        """Test that sequential sampling returns consecutive indices."""
        x_data = torch.arange(100).unsqueeze(1).float()  # [[0], [1], [2], ...]
        y_data = torch.arange(100).float()  # [0, 1, 2, ...]

        gen = FixedDatasetGenerator(x_data, y_data, sequential=True)

        # Generate multiple sequences to test
        for _ in range(10):
            x, y = gen.generate(5)
            # Check that indices are consecutive
            x_vals = x.squeeze(-1)
            for i in range(len(x_vals) - 1):
                # Either consecutive or wrapped around
                assert x_vals[i + 1] == x_vals[i] + 1 or x_vals[i] == 99

    def test_sequential_wraparound_when_dataset_smaller_than_seqlen(self) -> None:
        """Test sequential sampling behavior when dataset < seq_len."""
        x_data = torch.arange(10).unsqueeze(1).float()
        y_data = torch.arange(10).float()

        gen = FixedDatasetGenerator(x_data, y_data, sequential=True)

        # Request more than dataset size
        x, y = gen.generate(20)

        assert x.shape == (20, 1)
        assert y.shape == (20,)
        # Should wrap around with some random fill

    def test_random_sampling_coverage(self) -> None:
        """Test that random sampling covers the dataset reasonably."""
        x_data = torch.arange(100).unsqueeze(1).float()
        y_data = torch.arange(100).float()

        gen = FixedDatasetGenerator(x_data, y_data, sequential=False)

        # Generate many samples
        all_indices = []
        for _ in range(100):
            x, y = gen.generate(10)
            all_indices.extend(x.squeeze(-1).tolist())

        # Check that we see variety in sampled indices (not all same)
        unique_indices = set(all_indices)
        assert len(unique_indices) > 10  # Should see more than 10 unique values

    def test_empty_dataset_raises_error(self) -> None:
        """Test that empty dataset raises ValueError."""
        x_data = torch.zeros(0, 2)
        y_data = torch.zeros(0)

        with pytest.raises(ValueError, match="Dataset cannot be empty"):
            FixedDatasetGenerator(x_data, y_data)

    def test_single_datapoint(self) -> None:
        """Test generator with single datapoint."""
        x_data = torch.tensor([[1.0, 2.0]])
        y_data = torch.tensor([3.0])

        gen = FixedDatasetGenerator(x_data, y_data, sequential=False)

        x, y = gen.generate(5)

        assert x.shape == (5, 2)
        assert y.shape == (5,)
        # All values should be the same since only one datapoint
        assert torch.allclose(x, x_data[0])
        assert torch.allclose(y, y_data[0])

    def test_mismatched_xy_lengths_raises_error(self) -> None:
        """Test that mismatched x/y lengths raise ValueError."""
        x_data = torch.randn(100, 2)
        y_data = torch.randn(50)  # Different length

        with pytest.raises(ValueError, match="x_data and y_data must have same length"):
            FixedDatasetGenerator(x_data, y_data)

    def test_short_dataset_behavior(self) -> None:
        """Test behavior when dataset size < requested seq_len."""
        x_data = torch.randn(5, 2)
        y_data = torch.randn(5)

        # Sequential mode
        gen_seq = FixedDatasetGenerator(x_data, y_data, sequential=True)
        x_seq, y_seq = gen_seq.generate(10)
        assert x_seq.shape == (10, 2)
        assert y_seq.shape == (10,)

        # Random mode
        gen_rand = FixedDatasetGenerator(x_data, y_data, sequential=False)
        x_rand, y_rand = gen_rand.generate(10)
        assert x_rand.shape == (10, 2)
        assert y_rand.shape == (10,)

    def test_sequential_mode_deterministic_from_start(self) -> None:
        """Test that sequential samples are deterministic given same start."""
        x_data = torch.arange(100).unsqueeze(1).float()
        y_data = torch.arange(100).float()

        gen = FixedDatasetGenerator(x_data, y_data, sequential=True)

        # Generate multiple sequences - they should vary due to random start
        samples = []
        for _ in range(10):
            x, y = gen.generate(5)
            samples.append(x[0, 0].item())

        # Not all samples should start at the same point
        assert len(set(samples)) > 1

    def test_dataset_size_property(self) -> None:
        """Test that dataset_size property is correctly set."""
        x_data = torch.randn(42, 3)
        y_data = torch.randn(42)

        gen = FixedDatasetGenerator(x_data, y_data)

        assert gen.dataset_size == 42

    def test_input_dim_property(self) -> None:
        """Test that input_dim is correctly inferred from x_data."""
        for input_dim in [1, 2, 5, 10]:
            x_data = torch.randn(20, input_dim)
            y_data = torch.randn(20)

            gen = FixedDatasetGenerator(x_data, y_data)

            assert gen.input_dim == input_dim

    def test_different_data_types(self) -> None:
        """Test generator works with different tensor dtypes."""
        # Float data
        x_float = torch.randn(10, 2)
        y_float = torch.randn(10)
        gen_float = FixedDatasetGenerator(x_float, y_float)
        x, y = gen_float.generate(5)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

        # Integer y (for classification)
        x_data = torch.randn(10, 2)
        y_int = torch.randint(0, 5, (10,))
        gen_int = FixedDatasetGenerator(x_data, y_int)
        x, y = gen_int.generate(5)
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64

    def test_large_sequence_generation(self) -> None:
        """Test generating sequences longer than typical."""
        x_data = torch.randn(200, 3)
        y_data = torch.randn(200)

        gen = FixedDatasetGenerator(x_data, y_data)

        x, y = gen.generate(100)

        assert x.shape == (100, 3)
        assert y.shape == (100,)

    def test_multiple_generations_independent(self) -> None:
        """Test that multiple generate calls return different samples (random mode)."""
        x_data = torch.arange(100).unsqueeze(1).float()
        y_data = torch.arange(100).float()

        gen = FixedDatasetGenerator(x_data, y_data, sequential=False)

        x1, y1 = gen.generate(10)
        x2, y2 = gen.generate(10)

        # With high probability, random samples should differ
        # (could fail rarely if same indices sampled, but very unlikely)
        assert not torch.allclose(x1, x2) or not torch.allclose(y1, y2)


class TestUnsupervisedDataGeneratorProtocol:
    """Test UnsupervisedDataGenerator protocol compliance."""

    def test_simple_unsupervised_generator(self) -> None:
        """Test that a simple unsupervised generator satisfies the protocol."""

        class SimpleUnsupGen:
            def __init__(self) -> None:
                self.input_dim = 1

            def generate(self, seq_len: int) -> torch.Tensor:
                return torch.randint(0, 10, (seq_len,))

        gen = SimpleUnsupGen()
        assert isinstance(gen, UnsupervisedDataGenerator)

        y = gen.generate(20)
        assert y.shape == (20,)

    def test_discrete_unsupervised_generator(self) -> None:
        """Test discrete unsupervised generator (like Beta-Bernoulli)."""

        class BernoulliGen:
            def __init__(self) -> None:
                self.input_dim = 1

            def generate(self, seq_len: int) -> torch.Tensor:
                theta = torch.distributions.Beta(2.0, 2.0).sample()
                return torch.distributions.Bernoulli(theta).sample((seq_len,))

        gen = BernoulliGen()
        assert isinstance(gen, UnsupervisedDataGenerator)

        y = gen.generate(30)
        assert y.shape == (30,)
        assert torch.all((y == 0) | (y == 1))

    def test_continuous_unsupervised_generator(self) -> None:
        """Test continuous unsupervised generator (like Exponential-Gamma)."""

        class ExponentialGen:
            def __init__(self) -> None:
                self.input_dim = 1

            def generate(self, seq_len: int) -> torch.Tensor:
                rate = torch.distributions.Gamma(2.0, 1.0).sample()
                return torch.distributions.Exponential(rate).sample((seq_len,))

        gen = ExponentialGen()
        assert isinstance(gen, UnsupervisedDataGenerator)

        y = gen.generate(25)
        assert y.shape == (25,)
        assert torch.all(y > 0)


class TestSamplerWithGeneratorTypes:
    """Test Sampler handles both supervised and unsupervised generators."""

    def test_sampler_with_supervised_generator(self) -> None:
        """Test Sampler yields (x, y) for supervised generators."""

        class SupervisedGen:
            def __init__(self) -> None:
                self.input_dim = 2

            def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
                x = torch.randn(seq_len, self.input_dim)
                y = torch.randn(seq_len)
                return x, y

        gen = SupervisedGen()
        sampler = Sampler(seq_len=10, data_generator=gen, internal_batch_size=4)

        iterator = iter(sampler)
        for _ in range(5):
            x, y = next(iterator)
            assert x is not None
            assert x.shape == (10, 2)
            assert y.shape == (10,)

    def test_sampler_with_unsupervised_generator(self) -> None:
        """Test Sampler yields (None, y) for unsupervised generators."""

        class UnsupervisedGen:
            def __init__(self) -> None:
                self.input_dim = 1

            def generate(self, seq_len: int) -> torch.Tensor:
                return torch.randint(0, 5, (seq_len,))

        gen = UnsupervisedGen()
        sampler = Sampler(seq_len=15, data_generator=gen, internal_batch_size=3)

        iterator = iter(sampler)
        for _ in range(5):
            x, y = next(iterator)
            assert x is None
            assert y.shape == (15,)

    def test_dataloader_with_supervised_generator(self) -> None:
        """Test DataLoader integration with supervised generator."""

        class SupervisedGen:
            def __init__(self) -> None:
                self.input_dim = 1

            def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
                x = torch.randn(seq_len, self.input_dim)
                y = torch.randn(seq_len)
                return x, y

        gen = SupervisedGen()
        training_config = TrainingConfig(batch_size=8, seq_len=12)

        dataloader = build_dataloader(gen, training_config)
        iterator = iter(dataloader)

        for _ in range(3):
            x_batch, y_batch = next(iterator)
            assert x_batch is not None
            assert x_batch.shape == (8, 12, 1)
            assert y_batch.shape == (8, 12)

    def test_dataloader_with_unsupervised_generator(self) -> None:
        """Test DataLoader integration with unsupervised generator."""

        class UnsupervisedGen:
            def __init__(self) -> None:
                self.input_dim = 1

            def generate(self, seq_len: int) -> torch.Tensor:
                return torch.randint(0, 2, (seq_len,))

        gen = UnsupervisedGen()
        training_config = TrainingConfig(batch_size=4, seq_len=16)

        dataloader = build_dataloader(gen, training_config)
        iterator = iter(dataloader)

        for _ in range(3):
            x_batch, y_batch = next(iterator)
            assert x_batch is None
            assert y_batch.shape == (4, 16)


class TestTrainingLoopWithGeneratorTypes:
    """Test training loop handles both supervised and unsupervised generators."""

    def test_compute_loss_with_supervised_model_requires_x(self) -> None:
        """Test that compute_loss raises error when x is None for supervised models."""
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
        model = PFNModel(config)
        device = next(model.parameters()).device

        y = torch.randn(4, 16, device=device)

        with pytest.raises(
            AssertionError, match="x must be provided for supervised models"
        ):
            compute_loss(model, None, y)

    def test_compute_loss_with_unsupervised_model_requires_none_x(self) -> None:
        """Test that compute_loss raises error when x is not None for unsupervised models."""
        config = UnsupervisedPFNConfig(
            n_layers=1,
            d_model=32,
            n_ctx=64,
            d_head=16,
            n_heads=2,
            d_vocab=2,
        )
        model = PFNModel(config)
        device = next(model.parameters()).device

        x = torch.randn(4, 16, 1, device=device)
        y = torch.randint(0, 2, (4, 16), device=device)

        with pytest.raises(
            AssertionError, match="x must be None for unsupervised models"
        ):
            compute_loss(model, x, y)

    def test_training_with_supervised_generator(self) -> None:
        """Test training loop with supervised data generator."""

        class SupervisedGen:
            def __init__(self) -> None:
                self.input_dim = 1

            def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
                x = torch.randn(seq_len, self.input_dim)
                y = torch.randn(seq_len)
                return x, y

        config = SupervisedRegressionPFNConfig(
            n_layers=1,
            d_model=32,
            n_ctx=64,
            d_head=16,
            n_heads=2,
            d_vocab=10,
            input_dim=1,
            prediction_type="point",
            act_fn="gelu",
        )

        training_config = TrainingConfig(
            batch_size=4,
            seq_len=16,
            num_steps=5,
            log_every=2,
            save_checkpoint=False,
        )

        gen = SupervisedGen()
        model = train(
            data_generator=gen,
            model_config=config,
            training_config=training_config,
        )

        from pfn_transformerlens.model.PFN import BasePFN

        assert isinstance(model, BasePFN)

    def test_training_with_unsupervised_generator(self) -> None:
        """Test training loop with unsupervised data generator."""

        class UnsupervisedGen:
            def __init__(self) -> None:
                self.input_dim = 1

            def generate(self, seq_len: int) -> torch.Tensor:
                return torch.randint(0, 2, (seq_len,))

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
            seq_len=16,
            num_steps=5,
            log_every=2,
            save_checkpoint=False,
        )

        gen = UnsupervisedGen()
        model = train(
            data_generator=gen,
            model_config=config,
            training_config=training_config,
        )

        from pfn_transformerlens.model.PFN import BasePFN

        assert isinstance(model, BasePFN)

    def test_training_loop_doesnt_move_none_x_to_device(self) -> None:
        """Test that training loop handles x=None correctly."""

        class UnsupervisedGen:
            def __init__(self) -> None:
                self.input_dim = 1

            def generate(self, seq_len: int) -> torch.Tensor:
                return torch.randint(0, 2, (seq_len,))

        config = UnsupervisedPFNConfig(
            n_layers=1,
            d_model=16,
            n_ctx=32,
            d_head=8,
            n_heads=2,
            d_vocab=2,
        )

        training_config = TrainingConfig(
            batch_size=2,
            seq_len=8,
            num_steps=3,
            log_every=1,
            save_checkpoint=False,
        )

        gen = UnsupervisedGen()
        model = train(
            data_generator=gen,
            model_config=config,
            training_config=training_config,
        )

        assert model is not None


class TestTypeUnionBehavior:
    """Test DataGenerator type union behavior."""

    def test_data_generator_accepts_supervised(self) -> None:
        """Test that DataGenerator type accepts supervised generators."""

        class SupervisedGen:
            def __init__(self) -> None:
                self.input_dim = 2

            def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
                x = torch.randn(seq_len, self.input_dim)
                y = torch.randn(seq_len)
                return x, y

        gen: DataGenerator = SupervisedGen()
        assert isinstance(gen, SupervisedDataGenerator)

    def test_data_generator_accepts_unsupervised(self) -> None:
        """Test that DataGenerator type accepts unsupervised generators."""

        class UnsupervisedGen:
            def __init__(self) -> None:
                self.input_dim = 1

            def generate(self, seq_len: int) -> torch.Tensor:
                return torch.randint(0, 5, (seq_len,))

        gen: DataGenerator = UnsupervisedGen()
        assert isinstance(gen, UnsupervisedDataGenerator)

    def test_can_distinguish_generator_types_at_runtime(self) -> None:
        """Test runtime type checking by checking return values."""

        class SupervisedGen:
            def __init__(self) -> None:
                self.input_dim = 2

            def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
                x = torch.randn(seq_len, self.input_dim)
                y = torch.randn(seq_len)
                return x, y

        class UnsupervisedGen:
            def __init__(self) -> None:
                self.input_dim = 1

            def generate(self, seq_len: int) -> torch.Tensor:
                return torch.randint(0, 5, (seq_len,))

        supervised_gen = SupervisedGen()
        unsupervised_gen = UnsupervisedGen()

        # Check by inspecting return values (structural protocols don't check return types)
        sup_result = supervised_gen.generate(10)
        unsup_result = unsupervised_gen.generate(10)

        assert isinstance(sup_result, tuple)
        assert len(sup_result) == 2
        assert not isinstance(unsup_result, tuple)
        assert isinstance(unsup_result, torch.Tensor)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_supervised_generator_with_zero_input_dim(self) -> None:
        """Test supervised generator with input_dim=0 still returns x."""

        class ZeroInputGen:
            def __init__(self) -> None:
                self.input_dim = 0

            def generate(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
                x = torch.zeros(seq_len, 0)
                y = torch.randn(seq_len)
                return x, y

        gen = ZeroInputGen()
        assert isinstance(gen, SupervisedDataGenerator)

        x, y = gen.generate(10)
        assert x.shape == (10, 0)
        assert y.shape == (10,)

    def test_unsupervised_generator_with_seq_len_one(self) -> None:
        """Test unsupervised generator with seq_len=1."""

        class SimpleGen:
            def __init__(self) -> None:
                self.input_dim = 1

            def generate(self, seq_len: int) -> torch.Tensor:
                return torch.randint(0, 5, (seq_len,))

        gen = SimpleGen()
        y = gen.generate(1)
        assert y.shape == (1,)


class TestUnsupervisedProbabilisticGenerator:
    """Test UnsupervisedProbabilisticGenerator shape handling."""

    def test_bernoulli_generator_returns_correct_shape(self) -> None:
        """Test that Bernoulli unsupervised generator returns shape (seq_len,)."""
        prior = PriorDistribution(torch.distributions.Beta(2.0, 2.0))

        def bernoulli_parameterizer(
            theta: torch.Tensor, x: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            # must broadcast theta to match sequence length
            seq_len = x.shape[0]
            theta_broadcast = theta.expand(seq_len)
            return {"probs": theta_broadcast}

        likelihood = LikelihoodDistribution(
            base_distribution=torch.distributions.Bernoulli(0.5),
            parameterizer=bernoulli_parameterizer,
            input_dim=1,
        )

        gen = UnsupervisedProbabilisticGenerator(prior=prior, likelihood=likelihood)

        # test various sequence lengths
        for seq_len in [1, 10, 32, 64, 128]:
            y = gen.generate(seq_len)
            assert y.shape == (seq_len,), (
                f"expected shape ({seq_len},) but got {y.shape}"
            )
            assert y.ndim == 1, f"expected 1D tensor but got {y.ndim}D"

    def test_unsupervised_probabilistic_generator_with_sample_batch(self) -> None:
        """Test that sample_batch returns correct 2D shape for unsupervised."""
        prior = PriorDistribution(torch.distributions.Beta(1.0, 1.0))

        def bernoulli_parameterizer(
            theta: torch.Tensor, x: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            seq_len = x.shape[0]
            return {"probs": theta.expand(seq_len)}

        likelihood = LikelihoodDistribution(
            base_distribution=torch.distributions.Bernoulli(0.5),
            parameterizer=bernoulli_parameterizer,
            input_dim=1,
        )

        gen = UnsupervisedProbabilisticGenerator(prior=prior, likelihood=likelihood)

        # test sample_batch function
        batch_size = 8
        seq_len = 16
        x_batch, y_batch = sample_batch(gen, batch_size=batch_size, seq_len=seq_len)

        assert x_batch is None, "unsupervised should return x=None"
        assert y_batch.shape == (batch_size, seq_len), (
            f"expected shape ({batch_size}, {seq_len}) but got {y_batch.shape}"
        )
        assert y_batch.ndim == 2, f"expected 2D tensor but got {y_batch.ndim}D"

    def test_unsupervised_probabilistic_with_dataloader(self) -> None:
        """Test UnsupervisedProbabilisticGenerator with dataloader."""
        prior = PriorDistribution(torch.distributions.Beta(2.0, 5.0))

        def bernoulli_parameterizer(
            theta: torch.Tensor, x: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            return {"probs": theta.expand(x.shape[0])}

        likelihood = LikelihoodDistribution(
            base_distribution=torch.distributions.Bernoulli(0.5),
            parameterizer=bernoulli_parameterizer,
            input_dim=1,
        )

        gen = UnsupervisedProbabilisticGenerator(prior=prior, likelihood=likelihood)
        training_config = TrainingConfig(batch_size=4, seq_len=32)

        dataloader = build_dataloader(gen, training_config)
        iterator = iter(dataloader)

        for _ in range(3):
            x_batch, y_batch = next(iterator)
            assert x_batch is None
            assert y_batch.shape == (4, 32)
            assert y_batch.ndim == 2

    def test_unsupervised_probabilistic_trains_without_error(self) -> None:
        """Test training with UnsupervisedProbabilisticGenerator."""
        prior = PriorDistribution(torch.distributions.Beta(1.0, 1.0))

        def bernoulli_parameterizer(
            theta: torch.Tensor, x: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            return {"probs": theta.expand(x.shape[0])}

        likelihood = LikelihoodDistribution(
            base_distribution=torch.distributions.Bernoulli(0.5),
            parameterizer=bernoulli_parameterizer,
            input_dim=1,
        )

        # wrap generator to cast to long for discrete models
        class DiscreteBernoulliGenerator:
            def __init__(self, base_gen: UnsupervisedProbabilisticGenerator) -> None:
                self.base_gen = base_gen
                self.input_dim = base_gen.input_dim

            def generate(self, seq_len: int) -> torch.Tensor:
                y = self.base_gen.generate(seq_len)
                return y.long()

        base_gen = UnsupervisedProbabilisticGenerator(
            prior=prior, likelihood=likelihood
        )
        gen = DiscreteBernoulliGenerator(base_gen)

        config = UnsupervisedPFNConfig(
            d_vocab=2,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_head=16,
            n_ctx=64,
        )

        training_config = TrainingConfig(
            batch_size=4,
            seq_len=16,
            num_steps=3,
            log_every=2,
            save_checkpoint=False,
        )

        model = train(
            data_generator=gen,
            model_config=config,
            training_config=training_config,
        )

        assert model is not None

    def test_parameterizer_without_broadcast_fails(self) -> None:
        """Test that parameterizer without broadcasting causes shape mismatch."""
        prior = PriorDistribution(torch.distributions.Beta(2.0, 2.0))

        # intentionally broken parameterizer (doesn't broadcast)
        def broken_parameterizer(
            theta: torch.Tensor, x: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            return {"probs": theta}  # scalar, not broadcasted!

        likelihood = LikelihoodDistribution(
            base_distribution=torch.distributions.Bernoulli(0.5),
            parameterizer=broken_parameterizer,
            input_dim=1,
        )

        gen = UnsupervisedProbabilisticGenerator(prior=prior, likelihood=likelihood)

        # this should return wrong shape
        y = gen.generate(seq_len=32)
        # the bug is that y will be scalar or wrong shape
        assert y.shape != (32,), "broken parameterizer should not return correct shape"
