"""PMC sample generation and persistence helpers for Markov analyses."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from markov.data import MarkovChainDataset
from markov.model import MarkovTransformer
from markov.plotting import plot_pmc_distributions, plot_pmc_matrix_summary
from markov.predictive_monte_carlo import predictive_monte_carlo_transition_matrix


@dataclass(slots=True)
class PMCSampleBundle:
    """Prior and posterior transition-matrix samples for PMC analysis."""

    prior_samples: np.ndarray
    posterior_samples: np.ndarray
    training_matrices: np.ndarray
    target_matrix: np.ndarray
    prompt_tokens: np.ndarray
    prompt_chain_index: int
    prompt_len: int
    generation_length: int


@dataclass(slots=True)
class PMCEvalBundle:
    """Cached prompt/target bundle reused across PMC reruns."""

    training_matrices: np.ndarray
    target_matrix: np.ndarray
    prompt_tokens: np.ndarray
    prompt_chain_index: int
    prompt_len: int
    generation_length: int


@dataclass(slots=True)
class PMCSamplingConfig:
    """Default PMC sampling settings shared across Markov experiments."""

    num_samples: int = 128
    prompt_len: int = 8
    generation_length: int = 400
    seed: int = 0


def _validate_generation_request(
    dataset: MarkovChainDataset,
    *,
    prompt_len: int,
    num_samples: int,
    generation_length: int,
) -> None:
    """Validate PMC sampling lengths against the configured model context."""
    if prompt_len < 0:
        raise ValueError("prompt_len must be non-negative.")
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if generation_length < 2:
        raise ValueError("generation_length must be at least 2.")
    if prompt_len >= dataset.seq_len:
        raise ValueError(
            f"prompt_len must be smaller than the model context length ({dataset.seq_len})."
        )
    if prompt_len + generation_length > dataset.seq_len:
        raise ValueError(
            "prompt_len + generation_length exceeds the model context length "
            f"({dataset.seq_len}) once BOS is included in the input stream."
        )


@torch.inference_mode()
def build_pmc_eval_bundle(
    dataset: MarkovChainDataset,
    prompt_len: int,
    generation_length: int,
) -> PMCEvalBundle:
    """Build a deterministic prompt/target bundle for PMC analysis."""
    _validate_generation_request(
        dataset,
        prompt_len=prompt_len,
        num_samples=1,
        generation_length=generation_length,
    )

    target_index = int(
        torch.randint(0, dataset.n_chains, (1,), device=dataset.device).item()
    )
    target_matrix = dataset.transition_matrices[target_index]
    prompt_tokens = (
        dataset.sample_eval_chains(target_matrix, prompt_len)
        if prompt_len > 0
        else torch.empty(0, dtype=torch.long, device=dataset.device)
    )

    return PMCEvalBundle(
        training_matrices=dataset.transition_matrices.cpu().numpy(),
        target_matrix=target_matrix.cpu().numpy(),
        prompt_tokens=prompt_tokens.cpu().numpy(),
        prompt_chain_index=target_index,
        prompt_len=prompt_len,
        generation_length=generation_length,
    )


def save_pmc_eval_bundle(bundle: PMCEvalBundle, path: str | Path) -> None:
    """Persist the cached prompt/target bundle used by PMC analysis."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        training_matrices=bundle.training_matrices,
        target_matrix=bundle.target_matrix,
        prompt_tokens=bundle.prompt_tokens,
        prompt_chain_index=np.array(bundle.prompt_chain_index, dtype=np.int64),
        prompt_len=np.array(bundle.prompt_len, dtype=np.int64),
        generation_length=np.array(bundle.generation_length, dtype=np.int64),
    )


def load_pmc_eval_bundle(path: str | Path) -> PMCEvalBundle:
    """Load a cached prompt/target bundle from disk."""
    with np.load(path) as archive:
        return PMCEvalBundle(
            training_matrices=archive["training_matrices"],
            target_matrix=archive["target_matrix"],
            prompt_tokens=archive["prompt_tokens"],
            prompt_chain_index=int(archive["prompt_chain_index"]),
            prompt_len=int(archive["prompt_len"]),
            generation_length=int(archive["generation_length"]),
        )


def resolve_or_create_pmc_eval_bundle(
    dataset: MarkovChainDataset,
    *,
    sampling: PMCSamplingConfig,
    output_path: str | Path,
) -> PMCEvalBundle:
    """Load an existing PMC eval bundle or create it deterministically."""
    output_path = Path(output_path)
    if output_path.exists():
        bundle = load_pmc_eval_bundle(output_path)
        if bundle.prompt_len != sampling.prompt_len:
            raise ValueError(
                f"Cached PMC eval bundle at {output_path} has prompt_len={bundle.prompt_len}, "
                f"expected {sampling.prompt_len}."
            )
        if bundle.generation_length != sampling.generation_length:
            raise ValueError(
                "Cached PMC eval bundle at "
                f"{output_path} has generation_length={bundle.generation_length}, "
                f"expected {sampling.generation_length}."
            )
        return bundle

    torch.manual_seed(sampling.seed)
    bundle = build_pmc_eval_bundle(
        dataset,
        prompt_len=sampling.prompt_len,
        generation_length=sampling.generation_length,
    )
    save_pmc_eval_bundle(bundle, output_path)
    return bundle


@torch.inference_mode()
def get_prior_and_posterior_samples(
    model: MarkovTransformer,
    dataset: MarkovChainDataset,
    eval_bundle: PMCEvalBundle,
    num_samples: int,
    seed: int,
) -> PMCSampleBundle:
    """Generate prior and posterior PMC samples for a fixed cached eval bundle."""
    was_training = model.training
    model.eval()
    try:
        _validate_generation_request(
            dataset,
            prompt_len=eval_bundle.prompt_len,
            num_samples=num_samples,
            generation_length=eval_bundle.generation_length,
        )

        empty_prefix = torch.empty(0, dtype=torch.long, device=dataset.device)
        prompt_tokens = torch.from_numpy(eval_bundle.prompt_tokens).to(
            device=dataset.device,
            dtype=torch.long,
        )

        torch.manual_seed(seed)
        prior_samples = predictive_monte_carlo_transition_matrix(
            model=model,
            dataset=dataset,
            forward_recursion_steps=eval_bundle.generation_length,
            forward_recursion_samples=num_samples,
            prompt=empty_prefix,
        )
        torch.manual_seed(seed + 1)
        posterior_samples = predictive_monte_carlo_transition_matrix(
            model=model,
            dataset=dataset,
            forward_recursion_steps=eval_bundle.generation_length,
            forward_recursion_samples=num_samples,
            prompt=prompt_tokens,
        )
        if not isinstance(prior_samples, np.ndarray):
            raise TypeError("Prior PMC unexpectedly returned rollout traces.")
        if not isinstance(posterior_samples, np.ndarray):
            raise TypeError("Posterior PMC unexpectedly returned rollout traces.")

        return PMCSampleBundle(
            prior_samples=prior_samples,
            posterior_samples=posterior_samples,
            training_matrices=eval_bundle.training_matrices,
            target_matrix=eval_bundle.target_matrix,
            prompt_tokens=eval_bundle.prompt_tokens,
            prompt_chain_index=eval_bundle.prompt_chain_index,
            prompt_len=eval_bundle.prompt_len,
            generation_length=eval_bundle.generation_length,
        )
    finally:
        if was_training:
            model.train()


def save_pmc_samples(bundle: PMCSampleBundle, path: str | Path) -> None:
    """Persist PMC samples and metadata to a compressed NumPy archive."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        prior_samples=bundle.prior_samples,
        posterior_samples=bundle.posterior_samples,
        training_matrices=bundle.training_matrices,
        target_matrix=bundle.target_matrix,
        prompt_tokens=bundle.prompt_tokens,
        prompt_chain_index=np.array(bundle.prompt_chain_index, dtype=np.int64),
        prompt_len=np.array(bundle.prompt_len, dtype=np.int64),
        generation_length=np.array(bundle.generation_length, dtype=np.int64),
    )


def load_pmc_samples(path: str | Path) -> PMCSampleBundle:
    """Load a saved PMC sample archive."""
    with np.load(path) as archive:
        return PMCSampleBundle(
            prior_samples=archive["prior_samples"],
            posterior_samples=archive["posterior_samples"],
            training_matrices=archive["training_matrices"],
            target_matrix=archive["target_matrix"],
            prompt_tokens=archive["prompt_tokens"],
            prompt_chain_index=int(archive["prompt_chain_index"]),
            prompt_len=int(archive["prompt_len"]),
            generation_length=int(archive["generation_length"]),
        )


def resolve_pmc_sampling_config(
    config: PMCSamplingConfig,
    *,
    seq_len: int,
) -> PMCSamplingConfig:
    """Adjust PMC defaults to fit within a run's configured sequence length."""
    if config.prompt_len < 0:
        raise ValueError("prompt_len must be non-negative.")

    max_generation_length = seq_len - config.prompt_len
    if max_generation_length < 2:
        raise ValueError(
            "seq_len is too small to save PMC samples with the configured prompt_len. "
            f"Need at least prompt_len + 2 generated states, got seq_len={seq_len} and "
            f"prompt_len={config.prompt_len}."
        )

    return replace(
        config,
        generation_length=min(config.generation_length, max_generation_length),
    )


def generate_and_save_pmc_samples(
    model: MarkovTransformer,
    dataset: MarkovChainDataset,
    *,
    sampling: PMCSamplingConfig,
    output_path: str | Path,
    eval_bundle_path: str | Path | None = None,
) -> PMCSampleBundle:
    """Generate prior/posterior samples and save them to disk."""
    resolved_eval_bundle_path = (
        Path(output_path).with_name("pmc_eval_bundle.npz")
        if eval_bundle_path is None
        else Path(eval_bundle_path)
    )
    eval_bundle = resolve_or_create_pmc_eval_bundle(
        dataset,
        sampling=sampling,
        output_path=resolved_eval_bundle_path,
    )
    bundle = get_prior_and_posterior_samples(
        model=model,
        dataset=dataset,
        eval_bundle=eval_bundle,
        num_samples=sampling.num_samples,
        seed=sampling.seed,
    )
    save_pmc_samples(bundle, output_path)
    return bundle


def generate_and_save_pmc_artifacts(
    model: MarkovTransformer,
    dataset: MarkovChainDataset,
    *,
    sampling: PMCSamplingConfig,
    output_dir: str | Path,
) -> PMCSampleBundle:
    """Generate and save the full prior/posterior PMC artifact set.

    Writes the same artifact family as ``markov/run_pmc.py``:
    - ``pmc_eval_bundle.npz``
    - ``pmc_samples.npz``
    - ``pmc_prior.png``
    - ``pmc_posterior.png``
    - ``pmc_prior_marginals.png``
    - ``pmc_posterior_marginals.png``
    - ``pmc_prior_overview.png``
    - ``pmc_posterior_overview.png``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = generate_and_save_pmc_samples(
        model=model,
        dataset=dataset,
        sampling=sampling,
        output_path=output_dir / "pmc_samples.npz",
        eval_bundle_path=output_dir / "pmc_eval_bundle.npz",
    )
    bundle = load_pmc_samples(output_dir / "pmc_samples.npz")

    plot_pmc_distributions(
        bundle.prior_samples,
        bundle.training_matrices,
        output_dir / "pmc_prior.png",
        title_prefix="Prior samples",
    )
    plot_pmc_distributions(
        bundle.posterior_samples,
        bundle.training_matrices,
        output_dir / "pmc_posterior.png",
        title_prefix=f"Posterior samples (prompt len={bundle.prompt_len})",
        prompt_tokens=bundle.prompt_tokens,
        target_matrix=bundle.target_matrix,
    )
    plot_pmc_distributions(
        bundle.prior_samples,
        bundle.training_matrices,
        output_dir / "pmc_prior_marginals.png",
        title_prefix="Prior marginal densities",
        transitions_to_plot=[
            (0, dst) for dst in range(bundle.training_matrices.shape[1])
        ],
    )
    plot_pmc_distributions(
        bundle.posterior_samples,
        bundle.training_matrices,
        output_dir / "pmc_posterior_marginals.png",
        title_prefix=f"Posterior marginal densities (prompt len={bundle.prompt_len})",
        prompt_tokens=bundle.prompt_tokens,
        target_matrix=bundle.target_matrix,
        transitions_to_plot=[
            (0, dst) for dst in range(bundle.training_matrices.shape[1])
        ],
    )
    plot_pmc_matrix_summary(
        bundle.prior_samples,
        bundle.training_matrices,
        output_dir / "pmc_prior_overview.png",
        title="Prior PMC Overview",
    )
    plot_pmc_matrix_summary(
        bundle.posterior_samples,
        bundle.training_matrices,
        output_dir / "pmc_posterior_overview.png",
        title=f"Posterior PMC Overview (prompt len={bundle.prompt_len})",
        target_matrix=bundle.target_matrix,
    )

    return bundle
