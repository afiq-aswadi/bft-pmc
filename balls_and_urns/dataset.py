"""Dataset saving/loading for the balls and urns setting.

Saves evaluation datasets as .npz files for reproducible experiments.
Two types:
- Generalising dataset: prior-independent eval data (Dirichlet/generalising tokens)
- Memorising dataset: per-run data tied to a specific discrete prior (memorising tokens + theta pool)
"""

from pathlib import Path

import numpy as np


def save_generalising_dataset(
    path: Path,
    generalising_tokens: np.ndarray,
    alpha: np.ndarray,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        generalising_tokens=generalising_tokens,
        alpha=alpha,
        vocab_size=np.int64(vocab_size),
        seq_len=np.int64(seq_len),
        batch_size=np.int64(batch_size),
    )


def load_generalising_dataset(path: Path) -> dict:
    data = np.load(path)
    return {
        "generalising_tokens": data["generalising_tokens"],
        "alpha": data["alpha"],
        "vocab_size": int(data["vocab_size"]),
        "seq_len": int(data["seq_len"]),
        "batch_size": int(data["batch_size"]),
    }


def save_memorising_dataset(
    path: Path,
    memorising_tokens: np.ndarray,
    theta_pool: np.ndarray,
    num_tasks: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        memorising_tokens=memorising_tokens,
        theta_pool=theta_pool,
        num_tasks=np.int64(num_tasks),
    )


def load_memorising_dataset(path: Path) -> dict:
    data = np.load(path)
    return {
        "memorising_tokens": data["memorising_tokens"],
        "theta_pool": data["theta_pool"],
        "num_tasks": int(data["num_tasks"]),
    }


def save_predictive_samples(
    path: Path,
    model_samples: np.ndarray,
    baseline_generalising: np.ndarray,
    baseline_memorising: np.ndarray,
    posterior_dirichlet_alpha: np.ndarray,
    posterior_pool_weights: np.ndarray,
    prior_dirichlet_alpha: np.ndarray,
    theta_pool: np.ndarray,
    prompt_tokens: np.ndarray,
    step: int,
    prompt_source: str,
) -> None:
    n_prompts, _n_samples, vocab_size = model_samples.shape
    M = theta_pool.shape[0]
    assert (
        model_samples.shape == baseline_generalising.shape == baseline_memorising.shape
    ), (
        f"shape mismatch: model={model_samples.shape} gen={baseline_generalising.shape} mem={baseline_memorising.shape}"
    )
    assert posterior_dirichlet_alpha.shape == (n_prompts, vocab_size), (
        f"posterior_dirichlet_alpha shape {posterior_dirichlet_alpha.shape} != ({n_prompts}, {vocab_size})"
    )
    assert posterior_pool_weights.shape == (n_prompts, M), (
        f"posterior_pool_weights shape {posterior_pool_weights.shape} != ({n_prompts}, {M})"
    )
    assert prior_dirichlet_alpha.shape == (vocab_size,), (
        f"prior_dirichlet_alpha shape {prior_dirichlet_alpha.shape} != ({vocab_size},)"
    )
    assert theta_pool.shape[1] == vocab_size, (
        f"theta_pool vocab {theta_pool.shape[1]} != {vocab_size}"
    )
    assert prompt_source in ("prior", "data_generalising", "data_memorising"), (
        f"bad prompt_source {prompt_source!r}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        model_samples=model_samples,
        baseline_generalising=baseline_generalising,
        baseline_memorising=baseline_memorising,
        posterior_dirichlet_alpha=posterior_dirichlet_alpha,
        posterior_pool_weights=posterior_pool_weights,
        prior_dirichlet_alpha=prior_dirichlet_alpha,
        theta_pool=theta_pool,
        prompt_tokens=prompt_tokens,
        step=np.int64(step),
        prompt_source=np.array(prompt_source),
    )


def load_predictive_samples(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {
        "model_samples": data["model_samples"],
        "baseline_generalising": data["baseline_generalising"],
        "baseline_memorising": data["baseline_memorising"],
        "posterior_dirichlet_alpha": data["posterior_dirichlet_alpha"],
        "posterior_pool_weights": data["posterior_pool_weights"],
        "prior_dirichlet_alpha": data["prior_dirichlet_alpha"],
        "theta_pool": data["theta_pool"],
        "prompt_tokens": data["prompt_tokens"],
        "step": int(data["step"]),
        "prompt_source": str(data["prompt_source"]),
    }
