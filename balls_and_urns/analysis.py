"""Shared BAU predictive and distribution-metric computations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from balls_and_urns.baselines import (
    dirichlet_posterior_alpha,
    discrete_posterior_weights,
    generalising_predictive,
    memorising_predictive,
    sample_generalising_posterior,
    sample_memorising_posterior,
)
from balls_and_urns.dataset import (
    load_generalising_dataset,
    load_memorising_dataset,
    save_predictive_samples,
)
from balls_and_urns.predictive_monte_carlo import predictive_monte_carlo_theta_chunked
from metrics import energy_distance_multidim, sliced_wasserstein, symmetrised_kl
from pfn_transformerlens.model.PFN import DistributionPrediction, UnsupervisedPFN


def load_evaluation_tokens(
    dataset_dir: Path,
    run_id: str,
    theta_pool: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load fixed BAU tokens and verify that they match the analyzed run."""
    shared_path = dataset_dir / "shared.npz"
    run_path = dataset_dir / f"{run_id}.npz"
    if not shared_path.is_file():
        raise FileNotFoundError(
            f"Generalising BAU evaluation dataset is missing: {shared_path}"
        )
    if not run_path.is_file():
        raise FileNotFoundError(
            f"Memorising BAU evaluation dataset is missing: {run_path}"
        )

    shared = load_generalising_dataset(shared_path)
    memorising = load_memorising_dataset(run_path)
    vocab_size = theta_pool.shape[1]
    if shared["vocab_size"] != vocab_size:
        raise ValueError("Shared BAU dataset vocabulary does not match the checkpoint.")
    if not np.allclose(shared["alpha"], alpha.cpu().numpy()):
        raise ValueError("Shared BAU dataset alpha does not match the analysis config.")
    if memorising["num_tasks"] != theta_pool.shape[0]:
        raise ValueError(
            "Run-specific BAU dataset task count does not match the checkpoint."
        )
    if not np.allclose(memorising["theta_pool"], theta_pool.cpu().numpy()):
        raise ValueError(
            "Run-specific BAU dataset task pool does not match the checkpoint."
        )

    generalising_tokens = np.asarray(shared["generalising_tokens"])
    memorising_tokens = np.asarray(memorising["memorising_tokens"])
    for name, tokens in (
        ("generalising", generalising_tokens),
        ("memorising", memorising_tokens),
    ):
        if tokens.ndim != 2:
            raise ValueError(f"{name} BAU tokens must be 2D, got {tokens.shape}.")
        if tokens.size and (tokens.min() < 0 or tokens.max() >= vocab_size):
            raise ValueError(f"{name} BAU tokens lie outside the model vocabulary.")
    return torch.from_numpy(generalising_tokens), torch.from_numpy(memorising_tokens)


def prepend_bos(tokens: torch.Tensor, bos_token: int) -> torch.Tensor:
    """Prepend a BOS column to a batch of categorical sequences."""
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be 2D, got {tuple(tokens.shape)}.")
    bos = torch.full(
        (tokens.shape[0], 1),
        bos_token,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    return torch.cat([bos, tokens], dim=1)


def compute_predictive_metrics(
    model: UnsupervisedPFN,
    eval_tokens: dict[str, torch.Tensor],
    alpha: torch.Tensor,
    theta_pool: torch.Tensor,
) -> dict[str, float]:
    """Compute model-to-baseline symmetrised KL for each data source."""
    results: dict[str, float] = {}
    device = next(model.parameters()).device
    model.eval()
    for source, tokens_with_bos in eval_tokens.items():
        data_tokens = tokens_with_bos[:, 1:]
        with torch.no_grad():
            prediction = model.predict_on_prompt(tokens_with_bos.to(device))
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            assert isinstance(prediction, DistributionPrediction)
            model_probs = prediction.probs[:, :-1].cpu()

        baselines = {
            "generalising": generalising_predictive(data_tokens, alpha),
            "memorising": memorising_predictive(data_tokens, theta_pool),
        }
        for baseline, probabilities in baselines.items():
            results[f"{source}/delta_vs_baseline_{baseline}"] = symmetrised_kl(
                model_probs,
                probabilities,
            )
    return results


def precompute_baseline_samples(
    n_samples: int,
    alpha: torch.Tensor,
    theta_pool: torch.Tensor,
    prompts: torch.Tensor | None,
) -> dict[str, np.ndarray] | list[dict[str, np.ndarray]]:
    """Sample prior or per-prompt baseline distributions once per analysis."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    if prompts is None:
        return {
            "baseline_generalising": torch.distributions.Dirichlet(alpha)
            .sample((n_samples,))
            .numpy(),
            "baseline_memorising": theta_pool[
                torch.randint(len(theta_pool), (n_samples,))
            ].numpy(),
        }

    return [
        {
            "baseline_generalising": sample_generalising_posterior(
                prompt,
                alpha,
                n_samples,
            ).numpy(),
            "baseline_memorising": sample_memorising_posterior(
                prompt,
                theta_pool,
                n_samples,
            ).numpy(),
        }
        for prompt in prompts
    ]


def compute_distribution_metrics(
    model: UnsupervisedPFN,
    *,
    vocab_size: int,
    bos_token: int,
    alpha: torch.Tensor,
    theta_pool: torch.Tensor,
    effective_steps: int,
    n_samples: int,
    n_projections: int,
    chunk_size: int,
    prompts: torch.Tensor | None,
    baseline_samples: dict[str, np.ndarray] | list[dict[str, np.ndarray]],
    samples_save_path: Path,
    step: int,
    prompt_source: str,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Compute ED/SW and persist the Predictive Monte Carlo sample bundle."""
    if effective_steps < 1:
        raise ValueError("effective_steps must be positive.")
    device = next(model.parameters()).device

    if prompts is None:
        assert isinstance(baseline_samples, dict)
        model_samples = predictive_monte_carlo_theta_chunked(
            model=model,
            vocab_size=vocab_size,
            forward_recursion_steps=effective_steps,
            num_rollouts=n_samples,
            prompt=None,
            bos_token=bos_token,
            chunk_size=chunk_size,
        )
        per_prompt = [_distance_metrics(model_samples, baseline_samples, n_projections)]
        task_count = theta_pool.shape[0]
        save_predictive_samples(
            samples_save_path,
            model_samples=model_samples[None],
            baseline_generalising=baseline_samples["baseline_generalising"][None],
            baseline_memorising=baseline_samples["baseline_memorising"][None],
            posterior_dirichlet_alpha=alpha.numpy()[None],
            posterior_pool_weights=np.full((1, task_count), 1.0 / task_count),
            prior_dirichlet_alpha=alpha.numpy(),
            theta_pool=theta_pool.numpy(),
            prompt_tokens=np.empty((1, 0), dtype=np.int64),
            step=step,
            prompt_source=prompt_source,
        )
        return per_prompt[0], per_prompt

    assert isinstance(baseline_samples, list)
    if len(baseline_samples) != prompts.shape[0]:
        raise ValueError("baseline sample count must match the number of prompts.")

    prompt_count = prompts.shape[0]
    task_count = theta_pool.shape[0]
    model_stack = np.empty((prompt_count, n_samples, vocab_size))
    generalising_stack = np.empty_like(model_stack)
    memorising_stack = np.empty_like(model_stack)
    posterior_alpha = np.empty((prompt_count, vocab_size))
    posterior_weights = np.empty((prompt_count, task_count))
    per_prompt: list[dict[str, float]] = []

    for index, prompt in enumerate(prompts):
        model_samples = predictive_monte_carlo_theta_chunked(
            model=model,
            vocab_size=vocab_size,
            forward_recursion_steps=effective_steps,
            num_rollouts=n_samples,
            prompt=prompt.to(device),
            bos_token=bos_token,
            chunk_size=chunk_size,
        )
        references = baseline_samples[index]
        per_prompt.append(_distance_metrics(model_samples, references, n_projections))
        model_stack[index] = model_samples
        generalising_stack[index] = references["baseline_generalising"]
        memorising_stack[index] = references["baseline_memorising"]
        posterior_alpha[index] = dirichlet_posterior_alpha(prompt, alpha).numpy()
        posterior_weights[index] = discrete_posterior_weights(
            prompt, theta_pool
        ).numpy()

    save_predictive_samples(
        samples_save_path,
        model_samples=model_stack,
        baseline_generalising=generalising_stack,
        baseline_memorising=memorising_stack,
        posterior_dirichlet_alpha=posterior_alpha,
        posterior_pool_weights=posterior_weights,
        prior_dirichlet_alpha=alpha.numpy(),
        theta_pool=theta_pool.numpy(),
        prompt_tokens=prompts.numpy(),
        step=step,
        prompt_source=prompt_source,
    )
    averaged = {
        key: float(np.mean([metrics[key] for metrics in per_prompt]))
        for key in per_prompt[0]
    }
    return averaged, per_prompt


def _distance_metrics(
    model_samples: np.ndarray,
    baseline_samples: dict[str, np.ndarray],
    n_projections: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, references in baseline_samples.items():
        metrics[f"dist/ed_vs_{name}"] = energy_distance_multidim(
            model_samples,
            references,
        )
        metrics[f"dist/sw_vs_{name}"] = sliced_wasserstein(
            model_samples,
            references,
            n_projections=n_projections,
        )
    return metrics
