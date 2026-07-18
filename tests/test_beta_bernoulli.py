from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
import torch

import balls_and_urns.beta_bernoulli as beta_bernoulli
from pfn_transformerlens import PFN, UnsupervisedConfig
from pfn_transformerlens.model.PFN import UnsupervisedPFN


def _config(tmp_path: Path) -> beta_bernoulli.BetaBernoulliConfig:
    return beta_bernoulli.BetaBernoulliConfig(
        output_dir=tmp_path,
        d_model=8,
        d_mlp=16,
        n_layers=1,
        n_heads=1,
        d_head=8,
        seq_len=8,
        batch_size=2,
        num_steps=1,
        warmup_steps=0,
        num_workers=0,
        log_every=1,
        prompt_len=2,
        forward_recursion_steps=4,
        num_rollouts=2,
        chunk_size=1,
        device="cpu",
        plot_dpi=50,
    )


def _model(*, n_ctx: int = 8, d_vocab: int = 3) -> UnsupervisedPFN:
    model = PFN(
        UnsupervisedConfig(
            d_model=8,
            d_mlp=16,
            n_layers=1,
            n_heads=1,
            d_head=8,
            n_ctx=n_ctx,
            d_vocab=d_vocab,
            d_vocab_out=2,
            input_type="discrete",
            prediction_type="distribution",
        )
    )
    assert isinstance(model, UnsupervisedPFN)
    return model


def _results(config: beta_bernoulli.BetaBernoulliConfig) -> beta_bernoulli.PMCResults:
    prompts = np.tile(np.array([[0, 1]]), (9, 1))
    return beta_bernoulli.PMCResults(
        theta_stars=np.asarray(config.theta_stars),
        prompts=prompts,
        prior_samples=np.array([0.25, 0.75]),
        posterior_samples=np.tile(np.array([[0.4, 0.6]]), (9, 1)),
        posterior_alpha=np.full(9, 2.0),
        posterior_beta=np.full(9, 2.0),
    )


def test_beta_bernoulli_defaults_match_appendix_e() -> None:
    config = beta_bernoulli.BetaBernoulliConfig()
    config.validate()
    assert (config.prior_alpha, config.prior_beta) == (1.0, 1.0)
    assert (config.d_model, config.d_mlp) == (128, 512)
    assert (config.n_layers, config.n_heads, config.d_head) == (2, 2, 32)
    assert config.seq_len == 512
    assert (config.batch_size, config.num_steps) == (128, 100_000)
    assert (config.learning_rate, config.warmup_steps) == (1e-4, 500)
    assert config.prompt_len == 32
    assert config.theta_stars == tuple(index / 10 for index in range(1, 10))


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (replace(beta_bernoulli.BetaBernoulliConfig(), d_model=0), "positive"),
        (replace(beta_bernoulli.BetaBernoulliConfig(), num_workers=-1), "num_workers"),
        (replace(beta_bernoulli.BetaBernoulliConfig(), prior_alpha=0), "prior"),
        (
            replace(beta_bernoulli.BetaBernoulliConfig(), learning_rate=0),
            "learning_rate",
        ),
        (
            replace(beta_bernoulli.BetaBernoulliConfig(), warmup_steps=-1),
            "warmup_steps",
        ),
        (replace(beta_bernoulli.BetaBernoulliConfig(), theta_stars=(0.5,)), "nine"),
        (
            replace(
                beta_bernoulli.BetaBernoulliConfig(),
                theta_stars=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            ),
            "strictly",
        ),
        (replace(beta_bernoulli.BetaBernoulliConfig(), seq_len=482), "context"),
    ],
)
def test_beta_bernoulli_config_rejects_invalid_values(
    config: beta_bernoulli.BetaBernoulliConfig,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        config.validate()


def test_beta_bernoulli_training_uses_continuous_prior_and_paper_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_config(tmp_path), prior_alpha=2.0, prior_beta=3.0)
    model = _model()
    calls: list[tuple[object, object, object, object]] = []
    monkeypatch.setattr(
        beta_bernoulli,
        "train",
        lambda data, model_config, training_config, **kwargs: (
            calls.append((data, model_config, training_config, kwargs["data_config"]))
            or model
        ),
    )

    assert beta_bernoulli.train_beta_bernoulli(config) is model
    data_generator, model_config, training_config, data_config = calls[0]
    tokens = data_generator.generate(config.seq_len)
    assert tokens.shape == (config.seq_len,)
    assert tokens[0].item() == 2
    torch.testing.assert_close(
        data_generator.prior.base_distribution.concentration,
        torch.tensor([3.0, 2.0]),
    )
    assert model_config.d_vocab == 3
    assert model_config.d_vocab_out == 2
    assert model_config.d_mlp == config.d_mlp
    assert training_config.save_every == config.num_steps
    assert training_config.checkpoint_dir == str(tmp_path / "checkpoints")
    assert training_config.wandb_run_name == "beta-bernoulli"
    assert data_config == beta_bernoulli.BetaBernoulliDataConfig(2.0, 3.0)


def test_beta_bernoulli_prompt_sampling_and_pmc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    prompts = beta_bernoulli.sample_prompts(config)
    assert prompts.shape == (9, config.prompt_len)
    assert torch.equal(prompts, beta_bernoulli.sample_prompts(config))
    assert set(prompts.unique().tolist()) <= {0, 1}

    paper_prompts = beta_bernoulli.sample_prompts(beta_bernoulli.BetaBernoulliConfig())
    assert paper_prompts.sum(dim=1).tolist() == [2, 5, 16, 11, 13, 20, 23, 26, 25]

    calls: list[torch.Tensor | None] = []

    def fake_pmc(**kwargs: object) -> np.ndarray:
        prompt = kwargs["prompt"]
        assert prompt is None or isinstance(prompt, torch.Tensor)
        calls.append(prompt)
        probability = 0.25 if prompt is None else float(prompt.float().mean())
        return np.tile(
            np.array([[1.0 - probability, probability]], dtype=np.float32),
            (config.num_rollouts, 1),
        )

    monkeypatch.setattr(
        beta_bernoulli,
        "predictive_monte_carlo_theta_chunked",
        fake_pmc,
    )
    model = _model()
    model.train()
    results = beta_bernoulli.compute_pmc_results(model, config)
    assert not model.training
    assert len(calls) == 10
    np.testing.assert_array_equal(results.prompts, prompts.numpy())
    np.testing.assert_allclose(results.prior_samples, 0.25)
    prompt_ones = prompts.sum(dim=1).numpy()
    np.testing.assert_allclose(
        results.posterior_samples[:, 0],
        prompts.float().mean(dim=1).numpy(),
    )
    np.testing.assert_array_equal(
        results.posterior_alpha,
        config.prior_alpha + prompt_ones,
    )
    np.testing.assert_array_equal(
        results.posterior_beta,
        config.prior_beta + config.prompt_len - prompt_ones,
    )


def test_beta_bernoulli_pmc_rejects_incompatible_models(tmp_path: Path) -> None:
    config = _config(tmp_path)
    finite_pool_model = _model()
    finite_pool_model.task_distribution = {"tasks": torch.ones(2, 2)}
    with pytest.raises(ValueError, match="finite task pool"):
        beta_bernoulli.compute_pmc_results(finite_pool_model, config)
    with pytest.raises(ValueError, match="binary BOS"):
        beta_bernoulli.compute_pmc_results(_model(d_vocab=4), config)
    with pytest.raises(ValueError, match="shorter"):
        beta_bernoulli.compute_pmc_results(_model(n_ctx=6), config)


def test_beta_bernoulli_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_stem = tmp_path / "figures" / "beta_bernoulli_pmc_grid"
    beta_bernoulli.plot_pmc_grid(_results(config), config, output_stem)
    assert output_stem.with_suffix(".png").is_file()
    assert output_stem.with_suffix(".pdf").is_file()


def test_beta_bernoulli_main_trains_or_loads_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    train_calls: list[Path] = []
    load_calls: list[tuple[Path, str]] = []
    plot_calls: list[Path] = []
    monkeypatch.setattr(
        beta_bernoulli,
        "train_beta_bernoulli",
        lambda config: train_calls.append(config.output_dir) or model,
    )
    monkeypatch.setattr(
        beta_bernoulli,
        "load_checkpoint",
        lambda path, device: (
            load_calls.append((path, device)) or (model, None, object())
        ),
    )
    monkeypatch.setattr(
        beta_bernoulli,
        "compute_pmc_results",
        lambda received_model, config: _results(config),
    )
    monkeypatch.setattr(
        beta_bernoulli,
        "plot_pmc_grid",
        lambda results, config, output_stem: plot_calls.append(output_stem),
    )

    train_config = _config(tmp_path / "train")
    beta_bernoulli.main(train_config)
    assert train_calls == [train_config.output_dir]
    assert (train_config.output_dir / "config.json").is_file()
    with np.load(train_config.output_dir / "beta_bernoulli_pmc_samples.npz") as data:
        assert data["posterior_samples"].shape == (9, 2)

    checkpoint = tmp_path / "checkpoint.pt"
    load_config = replace(
        _config(tmp_path / "load"),
        checkpoint_path=checkpoint,
    )
    beta_bernoulli.main(load_config)
    assert load_calls == [(checkpoint, "cpu")]
    assert plot_calls == [
        train_config.output_dir / "beta_bernoulli_pmc_grid",
        load_config.output_dir / "beta_bernoulli_pmc_grid",
    ]


def test_beta_bernoulli_load_pmc_results_both_schemas(tmp_path: Path) -> None:
    config = _config(tmp_path)
    results = _results(config)

    native = tmp_path / "native.npz"
    np.savez_compressed(
        native,
        theta_stars=results.theta_stars,
        prompts=results.prompts,
        prior_samples=results.prior_samples,
        posterior_samples=results.posterior_samples,
        posterior_alpha=results.posterior_alpha,
        posterior_beta=results.posterior_beta,
    )
    loaded_native = beta_bernoulli.load_pmc_results(native)
    np.testing.assert_array_equal(
        loaded_native.posterior_samples, results.posterior_samples
    )

    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(
        legacy,
        grid_theta_stars=results.theta_stars,
        grid_prompts=results.prompts,
        grid_prior_theta_samples=results.prior_samples,
        grid_theta_samples_post=results.posterior_samples,
        grid_alpha_post=results.posterior_alpha,
        grid_beta_post=results.posterior_beta,
    )
    loaded_legacy = beta_bernoulli.load_pmc_results(legacy)
    np.testing.assert_array_equal(
        loaded_legacy.posterior_samples, results.posterior_samples
    )
    np.testing.assert_array_equal(loaded_legacy.posterior_alpha, results.posterior_alpha)


def test_beta_bernoulli_main_replot_from_skips_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    samples = tmp_path / "samples.npz"
    results = _results(config)
    np.savez_compressed(
        samples,
        theta_stars=results.theta_stars,
        prompts=results.prompts,
        prior_samples=results.prior_samples,
        posterior_samples=results.posterior_samples,
        posterior_alpha=results.posterior_alpha,
        posterior_beta=results.posterior_beta,
    )

    def _fail_train(_config: beta_bernoulli.BetaBernoulliConfig) -> NoReturn:
        raise AssertionError("training must not run in --replot-from mode")

    plot_calls: list[Path] = []
    monkeypatch.setattr(beta_bernoulli, "train_beta_bernoulli", _fail_train)
    monkeypatch.setattr(
        beta_bernoulli,
        "plot_pmc_grid",
        lambda results, config, output_stem: plot_calls.append(output_stem),
    )

    replot_config = replace(config, replot_from=samples)
    beta_bernoulli.main(replot_config)
    assert plot_calls == [config.output_dir / "beta_bernoulli_pmc_grid"]
