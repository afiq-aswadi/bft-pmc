from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from markov.config import MarkovConfig
from markov.data import MarkovChainDataset
from markov.model import MarkovTransformer
import markov.sweep_analysis as sweep


def _fake_pmc(
    *,
    dataset: MarkovChainDataset,
    forward_recursion_samples: int,
    prompt: torch.Tensor | None,
    **kwargs: object,
) -> np.ndarray:
    del kwargs
    matrix = np.full((dataset.k, dataset.k), 1.0 / dataset.k)
    if prompt is not None and prompt.ndim == 2:
        return np.tile(matrix, (prompt.shape[0], forward_recursion_samples, 1, 1))
    return np.tile(matrix, (forward_recursion_samples, 1, 1))


@pytest.mark.parametrize(
    "config",
    [
        sweep.SweepConfig(n_samples=0),
        sweep.SweepConfig(n_prompts=-1),
        sweep.SweepConfig(prompt_len=-1),
        sweep.SweepConfig(generation_length=1),
        sweep.SweepConfig(n_projections=0),
        sweep.SweepConfig(chunk_size=0),
    ],
)
def test_markov_sweep_config_validation(config: sweep.SweepConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_markov_run_discovery_from_tree_and_summary(tmp_path: Path) -> None:
    assert sweep._resolve_device("cpu").type == "cpu"
    run_dir = tmp_path / "checkpoints/run_keep"
    run_dir.mkdir(parents=True)
    older = run_dir / "checkpoint_step_2.pt"
    latest = run_dir / "checkpoint_step_10.pt"
    older.touch()
    latest.touch()
    assert sweep._step_from_checkpoint_name(latest) == 10
    assert sweep._latest_checkpoint_in_dir(run_dir) == latest
    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        sweep._latest_checkpoint_in_dir(tmp_path / "empty")

    tree_config = sweep.SweepConfig(
        checkpoint_root=str(tmp_path / "checkpoints"),
        run_name_contains="keep",
    )
    tree_specs = sweep._discover_run_specs(tree_config)
    assert len(tree_specs) == 1
    assert tree_specs[0].latest_checkpoint_path == latest
    with pytest.raises(FileNotFoundError, match="No checkpoint runs"):
        sweep._discover_run_specs(replace(tree_config, run_name_contains="missing"))

    output_dir = tmp_path / "outputs/run_keep"
    output_dir.mkdir(parents=True)
    config_path = output_dir / "resolved_config.yaml"
    config_path.touch()
    assert sweep._candidate_run_output_dirs(
        replace(tree_config, training_output_root=str(tmp_path / "outputs")),
        "run_keep",
    ) == [output_dir]
    assert (
        sweep._resolve_run_config_path(
            replace(tree_config, training_output_root=str(tmp_path / "outputs")),
            "run_keep",
        )
        == config_path
    )
    assert sweep._candidate_run_output_dirs(tree_config, "run_keep") == [
        Path("outputs/markov/training/run_keep")
    ]

    summary_path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "run_name": "run_keep",
                "latest_checkpoint_path": str(latest),
                "checkpoint_dir": "",
                "output_dir": str(output_dir),
                "n_chains": 2,
            },
            {
                "run_name": "run_other",
                "latest_checkpoint_path": "",
                "checkpoint_dir": str(run_dir),
                "output_dir": "",
                "n_chains": np.nan,
            },
        ]
    ).to_csv(summary_path, index=False)
    summary_specs = sweep._discover_run_specs(
        sweep.SweepConfig(summary_csv_path=str(summary_path))
    )
    assert len(summary_specs) == 2
    assert summary_specs[0].config_path == config_path
    assert summary_specs[0].expected_n_chains == 2
    assert summary_specs[1].expected_n_chains is None

    filtered = sweep._discover_run_specs(
        sweep.SweepConfig(
            summary_csv_path=str(summary_path),
            run_name_contains="other",
        )
    )
    assert [spec.run_name for spec in filtered] == ["run_other"]
    with pytest.raises(FileNotFoundError, match="No runs matched"):
        sweep._discover_run_specs(
            sweep.SweepConfig(
                summary_csv_path=str(summary_path),
                run_name_contains="missing",
            )
        )

    invalid_summary = tmp_path / "invalid.csv"
    pd.DataFrame([{"run_name": "bad"}]).to_csv(invalid_summary, index=False)
    with pytest.raises(ValueError, match="must provide"):
        sweep._discover_run_specs(
            sweep.SweepConfig(summary_csv_path=str(invalid_summary))
        )


def test_markov_reference_sampling_and_bundles() -> None:
    rng = np.random.default_rng(1)
    training = np.array(
        [
            [[0.8, 0.2], [0.3, 0.7]],
            [[0.2, 0.8], [0.7, 0.3]],
        ],
        dtype=np.float32,
    )
    assert sweep._flatten_transition_samples(training).shape == (2, 4)
    assert sweep._sample_discrete_reference(training, n_samples=3, rng=rng).shape == (
        3,
        2,
        2,
    )
    weighted = sweep._sample_discrete_reference(
        training,
        n_samples=3,
        rng=rng,
        weights=np.array([1.0, 0.0]),
    )
    np.testing.assert_array_equal(weighted, np.tile(training[0], (3, 1, 1)))
    assert sweep._sample_dirichlet_reference(2, n_samples=3, rng=rng).shape == (
        3,
        2,
        2,
    )
    counts = np.array([[1, 0], [0, 1]])
    assert sweep._sample_dirichlet_reference(
        2,
        n_samples=3,
        rng=rng,
        counts=counts,
    ).shape == (3, 2, 2)

    dataset = MarkovChainDataset(2, 6, 2, "cpu", seed=2)
    id_prompts = sweep._sample_prompt_batch(
        dataset,
        prompt_source="in_distribution",
        prompt_len=2,
        n_prompts=2,
        seed=1,
    )
    ood_prompts = sweep._sample_prompt_batch(
        dataset,
        prompt_source="out_of_distribution",
        prompt_len=2,
        n_prompts=2,
        seed=1,
    )
    assert id_prompts.shape == ood_prompts.shape == (2, 2)
    with pytest.raises(ValueError, match="Unsupported"):
        sweep._sample_prompt_batch(
            dataset,
            prompt_source="bad",
            prompt_len=2,
            n_prompts=1,
            seed=1,
        )

    prior = sweep._prepare_prior_reference_bundle(
        dataset,
        n_samples=3,
        id_seed=1,
        ood_seed=2,
    )
    posterior = sweep._prepare_posterior_reference_bundle(
        dataset,
        prompt_source="in_distribution",
        sampling=sweep.PMCSamplingConfig(3, 2, 3, 0),
        n_prompts=2,
        prompt_seed=1,
        baseline_seed=2,
    )
    assert prior.archive["baseline_in_distribution"].shape == (1, 3, 2, 2)
    assert posterior.archive["baseline_in_distribution"].shape == (2, 3, 2, 2)


def test_markov_metric_computation_and_sample_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = MarkovChainDataset(2, 6, 2, "cpu", seed=3)
    model = MarkovTransformer(3, 4, 6, 1, 2, 2, 10_000.0)
    sampling = sweep.PMCSamplingConfig(3, 2, 3, 0)
    monkeypatch.setattr(
        sweep,
        "predictive_monte_carlo_transition_matrix_chunked",
        _fake_pmc,
    )
    prior_references = sweep._prepare_prior_reference_bundle(
        dataset,
        n_samples=3,
        id_seed=1,
        ood_seed=2,
    )
    prior_metrics, prior_samples = sweep._compute_prior_metrics(
        model=model,
        dataset=dataset,
        sampling=sampling,
        n_projections=2,
        chunk_size=2,
        seed=1,
        references=prior_references,
    )
    assert prior_metrics.keys() >= {"ed_vs_baseline_in_distribution"}

    posterior_references = sweep._prepare_posterior_reference_bundle(
        dataset,
        prompt_source="in_distribution",
        sampling=sampling,
        n_prompts=2,
        prompt_seed=1,
        baseline_seed=2,
    )
    posterior_metrics, rows, posterior_samples = sweep._compute_posterior_metrics(
        model=model,
        dataset=dataset,
        references=posterior_references,
        sampling=sampling,
        n_projections=2,
        chunk_size=2,
        seed=1,
    )
    assert len(rows) == 2
    assert posterior_metrics.keys() == prior_metrics.keys()
    assert sweep._with_prompt_suffix({"ed": 1.0}, "id") == {"ed_from_prompts_id": 1.0}

    path = tmp_path / "samples/sample.npz"
    sweep._save_predictive_samples(
        path,
        **prior_samples,
        step=1,
        prompt_source="prior",
        n_chains=2,
    )
    assert path.exists()
    with pytest.raises(AssertionError, match="square"):
        sweep._save_predictive_samples(
            tmp_path / "bad.npz",
            model_samples=np.zeros((1, 2, 2, 3)),
            baseline_in_distribution=np.zeros((1, 2, 2, 3)),
            baseline_out_of_distribution=np.zeros((1, 2, 2, 3)),
            posterior_training_weights=np.zeros((1, 2)),
            posterior_dirichlet_alpha=np.zeros((1, 2, 3)),
            prior_dirichlet_alpha=np.zeros((2, 3)),
            training_transition_matrices=np.zeros((2, 2, 3)),
            prompt_tokens=np.empty((1, 0)),
            step=1,
            prompt_source="prior",
            n_chains=2,
        )
    assert posterior_samples["model_samples"].shape == (2, 3, 2, 2)


def _tiny_artifacts() -> SimpleNamespace:
    config = MarkovConfig(
        k=2,
        seq_len=6,
        n_chains=2,
        d_model=4,
        num_layers=1,
        num_heads=2,
        expansion_factor=2,
        context_len=3,
        num_eval_trials=1,
    )
    dataset = MarkovChainDataset(2, 6, 2, "cpu", seed=4)
    model = MarkovTransformer(3, 4, 6, 1, 2, 2, 10_000.0)
    return SimpleNamespace(
        config=config,
        dataset=dataset,
        model=model,
        device=torch.device("cpu"),
    )


def test_analyze_run_prior_and_posterior_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "checkpoints/run"
    run_dir.mkdir(parents=True)
    for step in [1, 2]:
        (run_dir / f"checkpoint_step_{step}.pt").touch()
    artifacts = _tiny_artifacts()
    monkeypatch.setattr(
        sweep, "load_trained_markov_artifacts", lambda *args, **kwargs: artifacts
    )
    monkeypatch.setattr(
        sweep,
        "load_markov_state_dict",
        lambda path, device: artifacts.model.state_dict(),
    )
    monkeypatch.setattr(
        sweep,
        "predictive_monte_carlo_transition_matrix_chunked",
        _fake_pmc,
    )
    plotted: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        sweep,
        "plot_distribution_distance_dynamics",
        lambda metrics, save_path, mode, log_xscale=False: plotted.append(
            (mode, log_xscale)
        ),
    )
    run_spec = sweep.RunSpec(
        "run", run_dir, run_dir / "checkpoint_step_2.pt", expected_n_chains=2
    )
    config = sweep.SweepConfig(
        output_dir=str(tmp_path / "output"),
        n_samples=3,
        n_prompts=2,
        prompt_len=2,
        generation_length=3,
        n_projections=2,
        chunk_size=2,
    )
    n_chains, rows = sweep._analyze_run(
        run_spec,
        config=config,
        device=torch.device("cpu"),
        runs_output_root=tmp_path / "runs",
        sweep_samples_dir=tmp_path / "sweep_samples",
    )
    assert n_chains == 2
    assert len(rows) == 3
    assert plotted == [
        ("prior", False),
        ("prior", True),
        ("posterior", False),
        ("posterior", True),
    ]
    assert (tmp_path / "runs/run/metrics.csv").exists()
    assert (tmp_path / "runs/run/per_prompt_metrics.csv").exists()

    plotted.clear()
    prior_n_chains, prior_rows = sweep._analyze_run(
        run_spec,
        config=replace(config, prompt_len=0, n_prompts=0),
        device=torch.device("cpu"),
        runs_output_root=tmp_path / "prior_runs",
        sweep_samples_dir=tmp_path / "prior_samples",
    )
    assert prior_n_chains == 2
    assert len(prior_rows) == 1
    assert plotted == [("prior", False), ("prior", True)]

    with pytest.raises(ValueError, match="expected n_chains"):
        sweep._analyze_run(
            replace(run_spec, expected_n_chains=3),
            config=config,
            device=torch.device("cpu"),
            runs_output_root=tmp_path / "bad",
            sweep_samples_dir=tmp_path / "bad_samples",
        )

    empty_dir = tmp_path / "checkpoints/empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No checkpoints"):
        sweep._analyze_run(
            sweep.RunSpec("empty", empty_dir, empty_dir / "missing.pt"),
            config=config,
            device=torch.device("cpu"),
            runs_output_root=tmp_path / "empty_runs",
            sweep_samples_dir=tmp_path / "empty_samples",
        )


def test_markov_sweep_main_and_duplicate_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [
        sweep.RunSpec("one", tmp_path, tmp_path / "one.pt"),
        sweep.RunSpec("two", tmp_path, tmp_path / "two.pt"),
    ]
    monkeypatch.setattr(sweep, "_discover_run_specs", lambda config: specs)
    calls = iter(
        [
            (2, [{"n_chains": 2, "prompt_length": 0, "prompt_source": "N/A"}]),
            (4, [{"n_chains": 4, "prompt_length": 0, "prompt_source": "N/A"}]),
        ]
    )
    monkeypatch.setattr(sweep, "_analyze_run", lambda *args, **kwargs: next(calls))
    plotted: list[int] = []
    monkeypatch.setattr(
        sweep,
        "plot_distribution_distance_sweep",
        lambda metrics, save_path, prompt_length: plotted.append(prompt_length),
    )
    config = sweep.SweepConfig(
        output_dir=str(tmp_path / "output"),
        prompt_len=0,
    )
    sweep.main(config)
    assert plotted == [0]
    assert (tmp_path / "output/metrics.csv").exists()

    posterior_calls = iter(
        [
            (2, [{"n_chains": 2, "prompt_length": 0, "prompt_source": "N/A"}]),
            (4, [{"n_chains": 4, "prompt_length": 2, "prompt_source": "data"}]),
        ]
    )
    monkeypatch.setattr(
        sweep,
        "_analyze_run",
        lambda *args, **kwargs: next(posterior_calls),
    )
    plotted.clear()
    sweep.main(replace(config, output_dir=str(tmp_path / "posterior"), prompt_len=2))
    assert plotted == [0, 2]

    monkeypatch.setattr(
        sweep,
        "_analyze_run",
        lambda *args, **kwargs: (
            2,
            [{"n_chains": 2, "prompt_length": 0, "prompt_source": "N/A"}],
        ),
    )
    with pytest.raises(ValueError, match="Multiple runs"):
        sweep.main(replace(config, output_dir=str(tmp_path / "duplicate")))
