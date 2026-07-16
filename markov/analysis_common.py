"""Shared loading utilities for Markov experiment analyses."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from markov.config import MarkovConfig, load_config
from markov.data import MarkovChainDataset
from markov.model import MarkovTransformer


@dataclass(slots=True)
class LoadedMarkovArtifacts:
    """Loaded config, model, dataset, and device for analysis scripts."""

    config: MarkovConfig
    model: MarkovTransformer
    dataset: MarkovChainDataset
    device: torch.device
    checkpoint_path: Path
    transition_matrices: torch.Tensor
    stationary_distributions: torch.Tensor
    task_distribution_source: str


def load_markov_state_dict(
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load a state dict from either a raw model file or a full training checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def _load_checkpoint_payload(
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    """Load a raw checkpoint payload."""
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _load_config_from_checkpoint(checkpoint_path: Path) -> MarkovConfig:
    """Load a MarkovConfig embedded in a training checkpoint."""
    payload = _load_checkpoint_payload(checkpoint_path, torch.device("cpu"))
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise FileNotFoundError(
            "No resolved_config.yaml was found and checkpoint does not contain a "
            f"training config: {checkpoint_path}"
        )
    config = MarkovConfig(**raw_config)
    config.validate()
    return config


def _load_task_distribution(
    run_dirs: list[Path],
    config: MarkovConfig,
    device: torch.device,
    allow_seed_rehydration: bool,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Recover (transition_matrices, stationary_distributions, source_tag) for a run.

    Preference order:
      1. transition_matrices.npy + stationary_distributions.npy in a candidate run dir.
      2. pmc_samples.npz:training_matrices; stationaries re-derived via
         MarkovChainDataset._compute_stationary_batch.
      3. Seed rehydration when explicitly enabled by the caller.
    """
    for run_dir in run_dirs:
        tm_path = run_dir / "transition_matrices.npy"
        pi_path = run_dir / "stationary_distributions.npy"
        if tm_path.exists() and pi_path.exists():
            transition_matrices = torch.from_numpy(np.load(tm_path)).to(device).float()
            stationary_distributions = (
                torch.from_numpy(np.load(pi_path)).to(device).float()
            )
            return transition_matrices, stationary_distributions, f"npy:{run_dir}"

        pmc_path = run_dir / "pmc_samples.npz"
        if pmc_path.exists():
            with np.load(pmc_path) as archive:
                if "training_matrices" in archive:
                    transition_matrices = (
                        torch.from_numpy(archive["training_matrices"])
                        .to(device)
                        .float()
                    )
                    rehydrator = MarkovChainDataset(
                        num_states=config.k,
                        seq_len=config.seq_len,
                        num_chains=config.n_chains,
                        device=device,
                        seed=None,
                    )
                    stationary_distributions = rehydrator._compute_stationary_batch(
                        transition_matrices
                    )
                    return (
                        transition_matrices,
                        stationary_distributions,
                        f"pmc_samples:{run_dir}",
                    )

    if not allow_seed_rehydration:
        raise FileNotFoundError(
            "No transition_matrices.npy or pmc_samples.npz was found in "
            f"{', '.join(str(path) for path in run_dirs)}. The original task "
            "distribution is required for reproducible analysis."
        )
    rehydrated = MarkovChainDataset(
        num_states=config.k,
        seq_len=config.seq_len,
        num_chains=config.n_chains,
        device=device,
        seed=config.seed,
    )
    return (
        rehydrated.transition_matrices,
        rehydrated.stationary_distributions,
        "seed_rehydration",
    )


def load_trained_markov_artifacts(
    config_path: str | Path | None,
    checkpoint_path: str | Path,
    *,
    device: torch.device | None = None,
    allow_seed_rehydration: bool = False,
) -> LoadedMarkovArtifacts:
    """Load a trained Markov model plus its dataset/config for downstream analysis."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    resolved_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else device
    )

    sibling_config = checkpoint_path.parent / "resolved_config.yaml"
    effective_config_path: Path | None = None
    if sibling_config.exists():
        effective_config_path = sibling_config
    elif config_path is not None:
        candidate = Path(config_path)
        if candidate.exists():
            effective_config_path = candidate

    if effective_config_path is not None:
        config = load_config(effective_config_path)
    else:
        config = _load_config_from_checkpoint(checkpoint_path)

    run_dirs: list[Path] = []
    if effective_config_path is not None:
        run_dirs.append(effective_config_path.parent)
    if checkpoint_path.parent not in run_dirs:
        run_dirs.append(checkpoint_path.parent)

    transition_matrices, stationary_distributions, source = _load_task_distribution(
        run_dirs=run_dirs,
        config=config,
        device=resolved_device,
        allow_seed_rehydration=allow_seed_rehydration,
    )

    assert transition_matrices.shape == (config.n_chains, config.k, config.k), (
        f"expected transition_matrices shape {(config.n_chains, config.k, config.k)}, "
        f"got {tuple(transition_matrices.shape)}"
    )
    assert stationary_distributions.shape == (config.n_chains, config.k), (
        f"expected stationary_distributions shape {(config.n_chains, config.k)}, "
        f"got {tuple(stationary_distributions.shape)}"
    )

    dataset = MarkovChainDataset(
        num_states=config.k,
        seq_len=config.seq_len,
        num_chains=config.n_chains,
        device=resolved_device,
        seed=None,
    )
    dataset.transition_matrices = transition_matrices
    dataset.stationary_distributions = stationary_distributions

    model = MarkovTransformer(
        vocab_size=dataset.vocab_size,
        d_model=config.d_model,
        seq_len=config.seq_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        expansion_factor=config.expansion_factor,
        rope_theta=config.rope_theta,
    ).to(resolved_device)
    model.load_state_dict(load_markov_state_dict(checkpoint_path, resolved_device))
    model.eval()

    return LoadedMarkovArtifacts(
        config=config,
        model=model,
        dataset=dataset,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        transition_matrices=transition_matrices,
        stationary_distributions=stationary_distributions,
        task_distribution_source=source,
    )
