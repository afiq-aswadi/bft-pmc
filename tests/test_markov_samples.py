from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from markov.data import MarkovChainDataset
import markov.samples_saving as samples_saving


class SampleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


def _fake_matrix_samples(
    *,
    dataset: MarkovChainDataset,
    forward_recursion_samples: int,
    **kwargs: object,
) -> np.ndarray:
    del kwargs
    return np.full(
        (forward_recursion_samples, dataset.k, dataset.k),
        1.0 / dataset.k,
    )


@pytest.mark.parametrize(
    ("prompt_len", "num_samples", "generation_length"),
    [(-1, 1, 2), (0, 0, 2), (0, 1, 1), (4, 1, 2), (3, 1, 2)],
)
def test_pmc_generation_request_validation(
    prompt_len: int,
    num_samples: int,
    generation_length: int,
) -> None:
    dataset = MarkovChainDataset(2, 4, 2, "cpu")
    with pytest.raises(ValueError):
        samples_saving._validate_generation_request(
            dataset,
            prompt_len=prompt_len,
            num_samples=num_samples,
            generation_length=generation_length,
        )


def test_pmc_eval_bundle_build_save_load_and_cache_validation(tmp_path: Path) -> None:
    dataset = MarkovChainDataset(2, 6, 2, "cpu", seed=1)
    empty_prompt = samples_saving.build_pmc_eval_bundle(dataset, 0, 4)
    prompted = samples_saving.build_pmc_eval_bundle(dataset, 2, 3)
    assert empty_prompt.prompt_tokens.size == 0
    assert prompted.prompt_tokens.shape == (2,)

    path = tmp_path / "nested/eval.npz"
    samples_saving.save_pmc_eval_bundle(prompted, path)
    reloaded = samples_saving.load_pmc_eval_bundle(path)
    np.testing.assert_array_equal(
        reloaded.training_matrices, prompted.training_matrices
    )
    np.testing.assert_array_equal(reloaded.prompt_tokens, prompted.prompt_tokens)
    assert reloaded.prompt_chain_index == prompted.prompt_chain_index

    config = samples_saving.PMCSamplingConfig(
        num_samples=2,
        prompt_len=2,
        generation_length=3,
        seed=4,
    )
    loaded = samples_saving.resolve_or_create_pmc_eval_bundle(
        dataset,
        sampling=config,
        output_path=path,
    )
    assert loaded.prompt_len == 2
    with pytest.raises(ValueError, match="prompt_len"):
        samples_saving.resolve_or_create_pmc_eval_bundle(
            dataset,
            sampling=replace(config, prompt_len=1),
            output_path=path,
        )
    with pytest.raises(ValueError, match="generation_length"):
        samples_saving.resolve_or_create_pmc_eval_bundle(
            dataset,
            sampling=replace(config, generation_length=2),
            output_path=path,
        )

    created_path = tmp_path / "created.npz"
    created = samples_saving.resolve_or_create_pmc_eval_bundle(
        dataset,
        sampling=config,
        output_path=created_path,
    )
    assert created_path.exists()
    assert created.prompt_len == 2


def test_pmc_sample_generation_and_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = MarkovChainDataset(2, 6, 2, "cpu", seed=2)
    model = SampleModel()
    model.train()
    eval_bundle = samples_saving.build_pmc_eval_bundle(dataset, 2, 3)
    monkeypatch.setattr(
        samples_saving,
        "predictive_monte_carlo_transition_matrix",
        _fake_matrix_samples,
    )
    bundle = samples_saving.get_prior_and_posterior_samples(
        model,
        dataset,
        eval_bundle,
        num_samples=3,
        seed=5,
    )
    assert model.training
    assert bundle.prior_samples.shape == (3, 2, 2)
    model.eval()
    samples_saving.get_prior_and_posterior_samples(
        model,
        dataset,
        eval_bundle,
        num_samples=2,
        seed=5,
    )
    assert not model.training

    path = tmp_path / "samples/bundle.npz"
    samples_saving.save_pmc_samples(bundle, path)
    loaded = samples_saving.load_pmc_samples(path)
    np.testing.assert_array_equal(loaded.posterior_samples, bundle.posterior_samples)
    assert loaded.prompt_chain_index == bundle.prompt_chain_index


def test_pmc_sample_generation_rejects_rollout_trace_tuples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = MarkovChainDataset(2, 6, 2, "cpu", seed=2)
    model = SampleModel()
    eval_bundle = samples_saving.build_pmc_eval_bundle(dataset, 1, 3)
    rollout_tuple = (
        np.full((2, 2, 2), 0.5),
        np.zeros((2, 3), dtype=np.int64),
    )
    monkeypatch.setattr(
        samples_saving,
        "predictive_monte_carlo_transition_matrix",
        lambda **kwargs: rollout_tuple,
    )
    with pytest.raises(TypeError, match="Prior PMC"):
        samples_saving.get_prior_and_posterior_samples(
            model,
            dataset,
            eval_bundle,
            num_samples=2,
            seed=1,
        )

    results = iter([np.full((2, 2, 2), 0.5), rollout_tuple])
    monkeypatch.setattr(
        samples_saving,
        "predictive_monte_carlo_transition_matrix",
        lambda **kwargs: next(results),
    )
    with pytest.raises(TypeError, match="Posterior PMC"):
        samples_saving.get_prior_and_posterior_samples(
            model,
            dataset,
            eval_bundle,
            num_samples=2,
            seed=1,
        )


def test_pmc_sampling_resolution_and_generation_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="prompt_len"):
        samples_saving.resolve_pmc_sampling_config(
            samples_saving.PMCSamplingConfig(prompt_len=-1),
            seq_len=4,
        )
    with pytest.raises(ValueError, match="too small"):
        samples_saving.resolve_pmc_sampling_config(
            samples_saving.PMCSamplingConfig(prompt_len=3),
            seq_len=4,
        )
    capped = samples_saving.resolve_pmc_sampling_config(
        samples_saving.PMCSamplingConfig(prompt_len=2, generation_length=10),
        seq_len=6,
    )
    assert capped.generation_length == 4

    dataset = MarkovChainDataset(2, 6, 2, "cpu", seed=3)
    model = SampleModel()
    monkeypatch.setattr(
        samples_saving,
        "predictive_monte_carlo_transition_matrix",
        _fake_matrix_samples,
    )
    sampling = samples_saving.PMCSamplingConfig(
        num_samples=2,
        prompt_len=1,
        generation_length=3,
        seed=1,
    )
    default_eval_bundle = samples_saving.generate_and_save_pmc_samples(
        model,
        dataset,
        sampling=sampling,
        output_path=tmp_path / "default/pmc_samples.npz",
    )
    assert default_eval_bundle.prior_samples.shape == (2, 2, 2)
    assert (tmp_path / "default/pmc_eval_bundle.npz").exists()

    explicit_eval_path = tmp_path / "explicit/eval.npz"
    samples_saving.generate_and_save_pmc_samples(
        model,
        dataset,
        sampling=sampling,
        output_path=tmp_path / "explicit/samples.npz",
        eval_bundle_path=explicit_eval_path,
    )
    assert explicit_eval_path.exists()


def test_generate_full_pmc_artifact_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = MarkovChainDataset(2, 6, 2, "cpu", seed=4)
    model = SampleModel()
    sampling = samples_saving.PMCSamplingConfig(2, 1, 3, 1)
    monkeypatch.setattr(
        samples_saving,
        "predictive_monte_carlo_transition_matrix",
        _fake_matrix_samples,
    )
    distribution_calls: list[Path] = []
    summary_calls: list[Path] = []
    monkeypatch.setattr(
        samples_saving,
        "plot_pmc_distributions",
        lambda samples, training, save_path, **kwargs: distribution_calls.append(
            Path(save_path)
        ),
    )
    monkeypatch.setattr(
        samples_saving,
        "plot_pmc_matrix_summary",
        lambda samples, training, save_path, **kwargs: summary_calls.append(
            Path(save_path)
        ),
    )
    bundle = samples_saving.generate_and_save_pmc_artifacts(
        model,
        dataset,
        sampling=sampling,
        output_dir=tmp_path / "artifacts",
    )
    assert bundle.prompt_len == 1
    assert len(distribution_calls) == 4
    assert len(summary_calls) == 2
