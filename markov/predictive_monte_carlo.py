"""Predictive Monte Carlo utilities for Markov-transformer analyses.

Generate rollouts from a trained Markov transformer and recover transition-matrix
samples via empirical transition counts. An empty prompt gives prior samples;
conditioning on observed states gives posterior samples.
"""

from __future__ import annotations

import numpy as np
import torch

from markov.data import MarkovChainDataset
from markov.model import MarkovTransformer


def prepare_model_for_long_rollout(
    model: MarkovTransformer,
    rollout_length: int,
    prompt_length: int = 0,
) -> int:
    """Cap rollout length so BOS + prompt + generated states fit the model context."""
    if rollout_length < 0:
        raise ValueError("rollout_length must be non-negative.")
    if prompt_length < 0:
        raise ValueError("prompt_length must be non-negative.")

    current_ctx = model.max_seq_len
    max_prompt_states = current_ctx - 1
    if prompt_length > max_prompt_states:
        raise ValueError(
            f"Prompt length {prompt_length} exceeds model capacity "
            f"({max_prompt_states} observed states plus BOS)."
        )

    if prompt_length + rollout_length <= current_ctx:
        return rollout_length

    return current_ctx - prompt_length


def _prepare_prompt_batch(
    prompt: torch.Tensor | None,
    *,
    device: torch.device,
    k: int,
) -> tuple[torch.Tensor, bool]:
    """Normalize prompt states to a batched 2D tensor."""
    if prompt is None:
        return torch.empty((1, 0), dtype=torch.long, device=device), True

    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
        squeeze_output = True
    elif prompt.dim() == 2:
        squeeze_output = False
    else:
        raise ValueError(f"prompt must be 1D or 2D, got shape {tuple(prompt.shape)}")

    prompt = prompt.to(device=device, dtype=torch.long)
    if prompt.numel() > 0 and (
        int(prompt.min().item()) < 0 or int(prompt.max().item()) >= k
    ):
        raise ValueError(f"Prompt tokens must lie in [0, {k - 1}].")
    return prompt, squeeze_output


def _estimate_transition_matrices_from_rollouts(
    states: torch.Tensor,
    *,
    k: int,
    smoothing: float,
) -> torch.Tensor:
    """Estimate a transition matrix from each rollout via smoothed counts."""
    if states.dim() < 2:
        raise ValueError("states must have shape (..., seq_len).")
    if states.shape[-1] < 2:
        raise ValueError("Each rollout must contain at least two states.")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive.")

    seq_len = states.shape[-1]
    flat_states = states.reshape(-1, seq_len).long()
    flat_batch = flat_states.shape[0]

    counts = torch.full(
        (flat_batch, k * k),
        fill_value=smoothing,
        dtype=torch.float32,
        device=states.device,
    )
    transition_indices = flat_states[:, :-1] * k + flat_states[:, 1:]
    updates = torch.ones_like(transition_indices, dtype=torch.float32)
    counts.scatter_add_(1, transition_indices, updates)

    matrices = counts.view(flat_batch, k, k)
    matrices = matrices / matrices.sum(dim=-1, keepdim=True)
    return matrices.view(*states.shape[:-1], k, k)


@torch.inference_mode()
def predictive_monte_carlo_transition_matrix(
    model: MarkovTransformer,
    dataset: MarkovChainDataset,
    forward_recursion_steps: int,
    forward_recursion_samples: int,
    prompt: torch.Tensor | None = None,
    sample: bool = True,
    temperature: float = 1.0,
    smoothing: float = 1.0,
    save_rollouts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Sample Markov transition matrices by rolling out the transformer.

    Parameters
    ----------
    prompt : tensor or None
        Optional prompt of observed states without BOS. Accepts:
        - Single prompt: (prompt_len,)
        - Batched prompts: (n_prompts, prompt_len)

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        - Single prompt or no prompt: (forward_recursion_samples, k, k)
        - Batched prompts: (n_prompts, forward_recursion_samples, k, k)
        If save_rollouts=True, also returns rollout state sequences with matching
        batch structure and shape (..., prompt_len + forward_recursion_steps).
    """
    if forward_recursion_steps < 1:
        raise ValueError("forward_recursion_steps must be positive.")
    if forward_recursion_samples < 1:
        raise ValueError("forward_recursion_samples must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    try:
        prompt_batch, squeeze_output = _prepare_prompt_batch(
            prompt,
            device=device,
            k=dataset.k,
        )
        prompt_len = prompt_batch.shape[1]
        if prompt_len + forward_recursion_steps < 2:
            raise ValueError(
                "prompt_len + forward_recursion_steps must be at least 2 so that "
                "transition matrices are identifiable from the rollout."
            )

        effective_steps = prepare_model_for_long_rollout(
            model,
            rollout_length=forward_recursion_steps,
            prompt_length=prompt_len,
        )
        if effective_steps != forward_recursion_steps:
            raise ValueError(
                "Requested rollout length exceeds the model context. "
                f"Prompt length {prompt_len} only allows {effective_steps} generated states."
            )

        n_prompts = prompt_batch.shape[0]
        expanded_prompt = prompt_batch.repeat_interleave(
            forward_recursion_samples,
            dim=0,
        )
        tokens = dataset.prepend_bos(expanded_prompt)

        for _ in range(forward_recursion_steps):
            logits = model(tokens)
            next_logits = logits[:, -1, : dataset.k]
            if sample:
                probabilities = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

        rollouts = tokens[:, 1:].view(
            n_prompts,
            forward_recursion_samples,
            prompt_len + forward_recursion_steps,
        )
        matrices = _estimate_transition_matrices_from_rollouts(
            rollouts,
            k=dataset.k,
            smoothing=smoothing,
        )

        matrices_np = matrices.detach().cpu().numpy()
        rollouts_np = rollouts.detach().cpu().numpy()

        if squeeze_output:
            matrices_np = matrices_np[0]
            rollouts_np = rollouts_np[0]

        if save_rollouts:
            return matrices_np, rollouts_np
        return matrices_np
    finally:
        if was_training:
            model.train()


@torch.inference_mode()
def predictive_monte_carlo_transition_matrix_chunked(
    model: MarkovTransformer,
    dataset: MarkovChainDataset,
    forward_recursion_steps: int,
    forward_recursion_samples: int,
    chunk_size: int = 128,
    prompt: torch.Tensor | None = None,
    sample: bool = True,
    temperature: float = 1.0,
    smoothing: float = 1.0,
    save_rollouts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Chunked Markov Predictive Monte Carlo over the rollout sample axis."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if forward_recursion_samples < 1:
        raise ValueError("forward_recursion_samples must be positive.")

    single_prompt = prompt is None or prompt.dim() == 1

    all_matrices: list[np.ndarray] = []
    all_rollouts: list[np.ndarray] = []

    for start in range(0, forward_recursion_samples, chunk_size):
        batch = min(chunk_size, forward_recursion_samples - start)
        result = predictive_monte_carlo_transition_matrix(
            model=model,
            dataset=dataset,
            forward_recursion_steps=forward_recursion_steps,
            forward_recursion_samples=batch,
            prompt=prompt,
            sample=sample,
            temperature=temperature,
            smoothing=smoothing,
            save_rollouts=save_rollouts,
        )

        if save_rollouts:
            matrices_chunk, rollouts_chunk = result
            all_matrices.append(matrices_chunk)
            all_rollouts.append(rollouts_chunk)
        else:
            assert isinstance(result, np.ndarray)
            all_matrices.append(result)

    concat_axis = 0 if single_prompt else 1
    matrices = np.concatenate(all_matrices, axis=concat_axis)
    if save_rollouts:
        rollouts = np.concatenate(all_rollouts, axis=concat_axis)
        return matrices, rollouts
    return matrices
