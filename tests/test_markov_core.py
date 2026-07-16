from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from markov.analysis_common import (
    _load_checkpoint_payload,
    _load_config_from_checkpoint,
    _load_task_distribution,
    load_markov_state_dict,
    load_trained_markov_artifacts,
)
from markov.baselines import (
    bi_inf_posterior_alpha,
    bi_inf_predictive,
    bi_ret_posterior_weights,
    bi_ret_predictive,
    sample_bi_inf_posterior,
    sample_bi_ret_posterior,
    uni_inf_predictive,
    uni_ret_predictive,
)
from markov.config import MarkovConfig, apply_overrides, dump_config, load_config
from markov.data import MarkovChainDataset
from markov.evals import (
    _sample_ood_tokens_bos,
    estimate_transition_matrix,
    evaluate_baseline_deltas,
    evaluate_kl,
)
from markov.model import (
    CausalSelfAttention,
    DecoderBlock,
    MarkovTransformer,
    apply_rotary_embedding,
    precompute_rotary_frequencies,
)
from markov.predictive_monte_carlo import (
    _estimate_transition_matrices_from_rollouts,
    _prepare_prompt_batch,
    predictive_monte_carlo_transition_matrix,
    predictive_monte_carlo_transition_matrix_chunked,
    prepare_model_for_long_rollout,
)


class UniformMarkovModel(nn.Module):
    def __init__(self, k: int, max_seq_len: int = 12) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.k = k
        self._max_seq_len = max_seq_len

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return (
            torch.zeros(
                (*tokens.shape, self.k + 1),
                dtype=torch.float32,
                device=tokens.device,
            )
            + self.anchor
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("k", 1, "k must"),
        ("seq_len", 1, "seq_len must"),
        ("n_chains", 0, "n_chains must"),
        ("batch_size", 0, "batch_size must"),
        ("learning_rate", 0.0, "learning_rate must"),
        ("eval_interval", 0, "eval_interval must"),
        ("max_steps", 0, "max_steps must"),
        ("d_model", 0, "d_model must"),
        ("num_layers", 0, "num_layers must"),
        ("num_heads", 0, "num_heads must"),
        ("d_model", 63, "divisible"),
        ("expansion_factor", 0, "expansion_factor must"),
        ("context_len", 1, "context_len must"),
        ("context_len", 512, "less than or equal"),
        ("num_eval_trials", 0, "num_eval_trials must"),
        ("delta_eval_batch_size", 0, "delta_eval_batch_size must"),
    ],
)
def test_markov_config_rejects_invalid_values(
    field: str,
    value: int | float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(MarkovConfig(), **{field: value}).validate()


def test_markov_config_yaml_round_trip_and_overrides(tmp_path: Path) -> None:
    config = MarkovConfig(seq_len=8, context_len=7, d_model=8, num_heads=2)
    output_path = tmp_path / "nested" / "config.yaml"
    dump_config(config, output_path)
    assert load_config(output_path) == config

    updated = apply_overrides(config, n_chains=3, learning_rate=0.01, max_steps=None)
    assert updated.n_chains == 3
    assert updated.learning_rate == pytest.approx(0.01)
    assert updated.max_steps == config.max_steps

    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("", encoding="utf-8")
    assert load_config(empty_path) == MarkovConfig()

    list_path = tmp_path / "list.yaml"
    list_path.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Expected a mapping"):
        load_config(list_path)

    unknown_path = tmp_path / "unknown.yaml"
    unknown_path.write_text("unknown: 1\n", encoding="utf-8")
    with pytest.raises(KeyError, match="unknown"):
        load_config(unknown_path)


def test_markov_dataset_sampling_and_stationary_distribution() -> None:
    dataset = MarkovChainDataset(2, seq_len=5, num_chains=3, device="cpu", seed=2)
    assert dataset.transition_matrices.shape == (3, 2, 2)
    assert dataset.stationary_distributions.shape == (3, 2)
    assert torch.allclose(dataset.stationary_distributions.sum(-1), torch.ones(3))
    assert dataset.sample_ood_matrix().shape == (2, 2)

    one_dimensional = dataset.prepend_bos(torch.tensor([0, 1]))
    two_dimensional = dataset.prepend_bos(torch.tensor([[0, 1], [1, 0]]))
    assert one_dimensional.tolist() == [[2, 0, 1]]
    assert two_dimensional.shape == (2, 3)
    assert dataset.sample_batch(4).shape == (4, 6)
    assert dataset.sample_eval_chains(dataset.transition_matrices[0], 4).shape == (4,)

    uniform = torch.full((2, 2, 2), 0.5)
    chains = dataset._generate_chains(uniform, torch.full((2, 2), 0.5), 3)
    assert chains.shape == (2, 3)


def test_markov_dataset_rejects_invalid_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    for args in [(1, 4, 2), (2, 1, 2), (2, 4, 0)]:
        with pytest.raises(ValueError):
            MarkovChainDataset(*args, device="cpu")

    dataset = MarkovChainDataset(2, 4, 2, "cpu")
    with pytest.raises(ValueError, match="batch, k, k"):
        dataset._compute_stationary_batch(torch.ones(2, 2))
    with pytest.raises(ValueError, match="must end"):
        dataset._compute_stationary_batch(torch.ones(1, 3, 3) / 3)
    with pytest.raises(ValueError, match="finite"):
        dataset._compute_stationary_batch(torch.full((1, 2, 2), torch.nan))
    with pytest.raises(ValueError, match="non-negative"):
        dataset._compute_stationary_batch(torch.tensor([[[-1.0, 2.0], [0.5, 0.5]]]))
    with pytest.raises(ValueError, match="sum to one"):
        dataset._compute_stationary_batch(torch.ones(1, 2, 2))

    monkeypatch.setattr(torch.linalg, "solve", lambda *_: torch.full((1, 2), torch.nan))
    with pytest.raises(ValueError, match="valid stationary"):
        dataset._compute_stationary_batch(torch.full((1, 2, 2), 0.5))

    with pytest.raises(ValueError, match="1D or 2D"):
        dataset.prepend_bos(torch.zeros(1, 1, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="length must"):
        dataset._generate_chains(torch.full((1, 2, 2), 0.5), torch.ones(1, 2) / 2, 0)
    with pytest.raises(ValueError, match="transition_matrices must"):
        dataset._generate_chains(torch.ones(1, 2), torch.ones(1, 2) / 2, 2)
    with pytest.raises(ValueError, match="stationary must"):
        dataset._generate_chains(torch.full((1, 2, 2), 0.5), torch.ones(2), 2)
    with pytest.raises(ValueError, match="batch_size"):
        dataset.sample_batch(0)
    with pytest.raises(ValueError, match="transition_matrix must"):
        dataset.sample_eval_chains(torch.ones(3, 3) / 3, 2)


def test_markov_baselines_match_direct_calculations() -> None:
    tokens = torch.tensor([[0, 1, 1, 0]])
    alpha = torch.tensor([1.0, 2.0])
    alpha_matrix = torch.ones(2, 2)
    stationaries = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    matrices = torch.tensor(
        [
            [[0.8, 0.2], [0.1, 0.9]],
            [[0.2, 0.8], [0.7, 0.3]],
        ]
    )

    uni_inf = uni_inf_predictive(tokens, alpha)
    expected_counts = torch.tensor([2.0, 2.0])
    assert torch.allclose(uni_inf[0, -1], (alpha + expected_counts) / 7)
    assert torch.allclose(uni_inf.sum(-1), torch.ones(1, 4))

    uni_ret = uni_ret_predictive(tokens, stationaries)
    bi_inf = bi_inf_predictive(tokens, alpha_matrix)
    bi_ret = bi_ret_predictive(tokens, matrices)
    for predictions in [uni_ret, bi_inf, bi_ret]:
        assert predictions.shape == (1, 4, 2)
        assert torch.allclose(predictions.sum(-1), torch.ones(1, 4))
    assert torch.allclose(bi_ret[0, 0], matrices[:, 0].mean(0))

    posterior_alpha = bi_inf_posterior_alpha(tokens[0], alpha_matrix)
    assert torch.equal(posterior_alpha, torch.tensor([[1.0, 2.0], [2.0, 2.0]]))
    weights = bi_ret_posterior_weights(tokens[0], matrices)
    assert weights.sum() == pytest.approx(1.0)
    assert sample_bi_inf_posterior(tokens[0], alpha_matrix, 3).shape == (3, 2, 2)
    retained = sample_bi_ret_posterior(tokens[0], matrices, 3)
    assert retained.shape == (3, 2, 2)
    assert all(
        any(torch.equal(sample, matrix) for matrix in matrices) for sample in retained
    )

    empty = torch.empty(0, dtype=torch.long)
    assert torch.equal(bi_inf_posterior_alpha(empty, alpha_matrix), alpha_matrix)
    assert torch.equal(bi_ret_posterior_weights(empty, matrices), torch.full((2,), 0.5))
    with pytest.raises(AssertionError, match="1-D"):
        bi_inf_posterior_alpha(tokens, alpha_matrix)
    with pytest.raises(AssertionError, match="1-D"):
        bi_ret_posterior_weights(tokens, matrices)
    with pytest.raises(AssertionError, match="alpha must"):
        bi_inf_predictive(tokens, torch.ones(2, 3))


def test_retained_baseline_stays_normalised_for_default_sequence_length() -> None:
    torch.manual_seed(42)
    stationaries = torch.rand(128, 10)
    stationaries /= stationaries.sum(dim=-1, keepdim=True)
    tokens = torch.randint(0, 10, (1, 1024))

    predictions = uni_ret_predictive(tokens, stationaries)

    assert torch.allclose(
        predictions.sum(dim=-1),
        torch.ones(1, 1024),
        atol=1e-5,
    )


def test_markov_model_components_and_context_limits() -> None:
    cosine, sine = precompute_rotary_frequencies(4, 6, 10_000.0)
    assert cosine.shape == sine.shape == (6, 4)
    with pytest.raises(ValueError, match="even head"):
        precompute_rotary_frequencies(3, 4, 10_000.0)

    tensor = torch.randn(2, 3, 2, 4)
    rotated = apply_rotary_embedding(tensor, cosine[:3], sine[:3])
    assert rotated.shape == tensor.shape

    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(5, 2, 4, 10_000.0)
    attention = CausalSelfAttention(4, 2, 4, 10_000.0)
    assert attention(torch.randn(2, 3, 4)).shape == (2, 3, 4)
    with pytest.raises(ValueError, match="exceeds"):
        attention(torch.randn(1, 5, 4))

    block = DecoderBlock(4, 2, 2, 4, 10_000.0)
    assert block(torch.randn(1, 3, 4)).shape == (1, 3, 4)
    model = MarkovTransformer(3, 4, 4, 2, 2, 2, 10_000.0)
    assert model.max_seq_len == 4
    assert model(torch.tensor([[2, 0, 1]])).shape == (1, 3, 3)


def test_markov_predictive_monte_carlo_paths() -> None:
    dataset = MarkovChainDataset(2, 6, 2, "cpu", seed=1)
    model = UniformMarkovModel(k=2, max_seq_len=8)
    model.train()

    matrices, rollouts = predictive_monte_carlo_transition_matrix(
        model,
        dataset,
        forward_recursion_steps=4,
        forward_recursion_samples=3,
        sample=False,
        save_rollouts=True,
    )
    assert model.training
    assert matrices.shape == (3, 2, 2)
    assert rollouts.shape == (3, 4)
    np.testing.assert_allclose(matrices.sum(-1), 1.0)

    prompt = torch.tensor([[0, 1], [1, 0]])
    sampled = predictive_monte_carlo_transition_matrix(
        model,
        dataset,
        3,
        2,
        prompt=prompt,
        sample=True,
        temperature=0.5,
    )
    assert sampled.shape == (2, 2, 2, 2)

    chunked, chunked_rollouts = predictive_monte_carlo_transition_matrix_chunked(
        model,
        dataset,
        3,
        5,
        chunk_size=2,
        prompt=torch.tensor([0]),
        sample=False,
        save_rollouts=True,
    )
    assert chunked.shape == (5, 2, 2)
    assert chunked_rollouts.shape == (5, 4)
    batched_chunked = predictive_monte_carlo_transition_matrix_chunked(
        model,
        dataset,
        3,
        3,
        chunk_size=2,
        prompt=prompt,
        sample=False,
    )
    assert batched_chunked.shape == (2, 3, 2, 2)


def test_markov_predictive_monte_carlo_rejects_invalid_inputs() -> None:
    dataset = MarkovChainDataset(2, 4, 2, "cpu")
    model = UniformMarkovModel(k=2, max_seq_len=4)

    assert prepare_model_for_long_rollout(model, 2, 1) == 2
    assert prepare_model_for_long_rollout(model, 5, 1) == 3
    for rollout_length, prompt_length in [(-1, 0), (1, -1)]:
        with pytest.raises(ValueError):
            prepare_model_for_long_rollout(model, rollout_length, prompt_length)
    with pytest.raises(ValueError, match="Prompt length"):
        prepare_model_for_long_rollout(model, 1, 4)

    empty, squeeze = _prepare_prompt_batch(None, device=torch.device("cpu"), k=2)
    assert empty.shape == (1, 0) and squeeze
    one, squeeze = _prepare_prompt_batch(
        torch.tensor([0]), device=torch.device("cpu"), k=2
    )
    assert one.shape == (1, 1) and squeeze
    two, squeeze = _prepare_prompt_batch(
        torch.tensor([[0]]), device=torch.device("cpu"), k=2
    )
    assert two.shape == (1, 1) and not squeeze
    with pytest.raises(ValueError, match="1D or 2D"):
        _prepare_prompt_batch(torch.zeros(1, 1, 1), device=torch.device("cpu"), k=2)
    for prompt in [torch.tensor([-1]), torch.tensor([2])]:
        with pytest.raises(ValueError, match="must lie"):
            _prepare_prompt_batch(prompt, device=torch.device("cpu"), k=2)

    with pytest.raises(ValueError, match="shape"):
        _estimate_transition_matrices_from_rollouts(torch.ones(3), k=2, smoothing=1)
    with pytest.raises(ValueError, match="at least two"):
        _estimate_transition_matrices_from_rollouts(torch.ones(2, 1), k=2, smoothing=1)
    with pytest.raises(ValueError, match="smoothing"):
        _estimate_transition_matrices_from_rollouts(torch.ones(2, 2), k=2, smoothing=0)

    invalid_calls = [
        {"forward_recursion_steps": 0, "forward_recursion_samples": 1},
        {"forward_recursion_steps": 2, "forward_recursion_samples": 0},
        {
            "forward_recursion_steps": 2,
            "forward_recursion_samples": 1,
            "temperature": 0,
        },
        {"forward_recursion_steps": 1, "forward_recursion_samples": 1},
        {
            "forward_recursion_steps": 4,
            "forward_recursion_samples": 1,
            "prompt": torch.tensor([0]),
        },
    ]
    for kwargs in invalid_calls:
        with pytest.raises(ValueError):
            predictive_monte_carlo_transition_matrix(model, dataset, **kwargs)

    for kwargs in [
        {"chunk_size": 0, "forward_recursion_samples": 1},
        {"chunk_size": 1, "forward_recursion_samples": 0},
    ]:
        with pytest.raises(ValueError):
            predictive_monte_carlo_transition_matrix_chunked(
                model,
                dataset,
                forward_recursion_steps=2,
                **kwargs,
            )


def test_markov_evaluators_cover_id_and_ood_paths() -> None:
    dataset = MarkovChainDataset(2, 5, 3, "cpu", seed=3)
    model = UniformMarkovModel(k=2, max_seq_len=8)
    model.train()
    estimated = estimate_transition_matrix(
        model, dataset, dataset.transition_matrices[0], 4
    )
    assert estimated.shape == (2, 2)
    assert evaluate_kl(model, dataset, 1, 4, is_ood=False) >= 0
    assert model.training
    model.eval()
    assert evaluate_kl(model, dataset, 1, 4, is_ood=True) >= 0
    assert not model.training

    ood_tokens = _sample_ood_tokens_bos(dataset, 2)
    assert ood_tokens.shape == (2, 6)
    for is_ood in [False, True]:
        model.train(is_ood)
        deltas = evaluate_baseline_deltas(model, dataset, 2, is_ood)
        assert set(deltas) == {
            "wellspec_generalising",
            "wellspec_memorising",
            "misspec_generalising",
            "misspec_memorising",
        }
        assert all(value >= 0 for value in deltas.values())


def _tiny_markov_config() -> MarkovConfig:
    return MarkovConfig(
        k=2,
        seq_len=4,
        n_chains=2,
        batch_size=2,
        d_model=4,
        num_layers=1,
        num_heads=2,
        expansion_factor=2,
        context_len=3,
        num_eval_trials=1,
    )


def _save_tiny_markov_run(
    run_dir: Path,
) -> tuple[MarkovConfig, MarkovChainDataset, Path]:
    config = _tiny_markov_config()
    dataset = MarkovChainDataset(2, 4, 2, "cpu", seed=config.seed)
    model = MarkovTransformer(3, 4, 4, 1, 2, 2, config.rope_theta)
    checkpoint_path = run_dir / "model.pt"
    run_dir.mkdir(parents=True)
    torch.save(model.state_dict(), checkpoint_path)
    dump_config(config, run_dir / "resolved_config.yaml")
    np.save(run_dir / "transition_matrices.npy", dataset.transition_matrices.numpy())
    np.save(
        run_dir / "stationary_distributions.npy",
        dataset.stationary_distributions.numpy(),
    )
    return config, dataset, checkpoint_path


def test_markov_artifact_loading_sources_and_full_loader(tmp_path: Path) -> None:
    config, dataset, checkpoint_path = _save_tiny_markov_run(tmp_path / "run")
    device = torch.device("cpu")

    state = load_markov_state_dict(checkpoint_path, device)
    assert "embedding.weight" in state
    wrapped_path = tmp_path / "wrapped.pt"
    torch.save(
        {
            "model_state_dict": state,
            "config": config.__dict__ if hasattr(config, "__dict__") else {},
        },
        wrapped_path,
    )
    assert load_markov_state_dict(wrapped_path, device).keys() == state.keys()
    assert _load_checkpoint_payload(wrapped_path, device).keys() >= {
        "model_state_dict",
        "config",
    }

    artifacts = load_trained_markov_artifacts(None, checkpoint_path, device=device)
    assert artifacts.task_distribution_source.startswith("npy:")
    assert not artifacts.model.training
    assert torch.equal(artifacts.transition_matrices, dataset.transition_matrices)

    npy_source = _load_task_distribution(
        [checkpoint_path.parent], config, device, False
    )
    assert npy_source[2].startswith("npy:")

    pmc_dir = tmp_path / "pmc"
    pmc_dir.mkdir()
    np.savez(
        pmc_dir / "pmc_samples.npz",
        training_matrices=dataset.transition_matrices.numpy(),
    )
    pmc_source = _load_task_distribution([pmc_dir], config, device, False)
    assert pmc_source[2].startswith("pmc_samples:")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    np.savez(empty_dir / "pmc_samples.npz", unrelated=np.ones(1))
    with pytest.raises(FileNotFoundError, match="original task distribution"):
        _load_task_distribution([empty_dir], config, device, False)
    rehydrated = _load_task_distribution([empty_dir], config, device, True)
    assert rehydrated[2] == "seed_rehydration"


def test_markov_artifact_loading_config_fallbacks_and_errors(tmp_path: Path) -> None:
    config = _tiny_markov_config()
    model = MarkovTransformer(3, 4, 4, 1, 2, 2, config.rope_theta)
    payload_path = tmp_path / "payload.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                field: getattr(config, field) for field in config.__dataclass_fields__
            },
        },
        payload_path,
    )
    dataset = MarkovChainDataset(2, 4, 2, "cpu", seed=config.seed)
    np.save(tmp_path / "transition_matrices.npy", dataset.transition_matrices.numpy())
    np.save(
        tmp_path / "stationary_distributions.npy",
        dataset.stationary_distributions.numpy(),
    )
    assert _load_config_from_checkpoint(payload_path) == config
    artifacts = load_trained_markov_artifacts(
        None, payload_path, device=torch.device("cpu")
    )
    assert artifacts.config == config

    missing_checkpoint = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_trained_markov_artifacts(None, missing_checkpoint)

    invalid_payload = tmp_path / "invalid.pt"
    torch.save({"model_state_dict": model.state_dict()}, invalid_payload)
    with pytest.raises(FileNotFoundError, match="does not contain"):
        _load_config_from_checkpoint(invalid_payload)

    external_config = tmp_path / "external.yaml"
    dump_config(config, external_config)
    external_run = tmp_path / "external_run"
    external_run.mkdir()
    external_checkpoint = external_run / "model.pt"
    torch.save(model.state_dict(), external_checkpoint)
    np.save(
        external_config.parent / "transition_matrices.npy",
        dataset.transition_matrices.numpy(),
    )
    np.save(
        external_config.parent / "stationary_distributions.npy",
        dataset.stationary_distributions.numpy(),
    )
    loaded = load_trained_markov_artifacts(
        external_config, external_checkpoint, device=torch.device("cpu")
    )
    assert loaded.config == config

    np.save(
        external_config.parent / "transition_matrices.npy",
        np.ones((1, 2, 2), dtype=np.float32) / 2,
    )
    with pytest.raises(AssertionError, match="expected transition_matrices"):
        load_trained_markov_artifacts(
            external_config, external_checkpoint, device=torch.device("cpu")
        )
