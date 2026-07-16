"""Predictive Monte Carlo for the Dirichlet-Multinomial setting.

Generate rollouts from an unsupervised PFN and extract theta (probability simplex)
estimates via empirical token frequencies.
"""

import numpy as np
import torch
from jaxtyping import Float

from pfn_transformerlens.model.PFN import UnsupervisedPFN


@torch.no_grad()
def predictive_monte_carlo_theta(
    model: UnsupervisedPFN,
    vocab_size: int,
    forward_recursion_steps: int,
    num_rollouts: int,
    prompt: Float[torch.Tensor, " K_init"] | None = None,
    bos_token: int | None = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """Generate rollouts and extract theta estimates via empirical frequencies.

    Args:
        model: Trained unsupervised PFN.
        vocab_size: Number of data tokens (excluding BOS).
        forward_recursion_steps: Number of tokens to generate per rollout.
        num_rollouts: Number of independent rollouts.
        prompt: Optional context tokens (without BOS). Shape (K_init,).
        bos_token: BOS token index. If provided, prepended to prompt.
        temperature: Sampling temperature.

    Returns:
        Theta estimates, shape (num_rollouts, vocab_size). Each row is an
        empirical frequency vector from one rollout.
    """
    device = next(model.parameters()).device

    # build the full prompt: [BOS] or [BOS, prompt_tokens]
    parts = []
    if bos_token is not None:
        parts.append(torch.tensor([bos_token], dtype=torch.float32, device=device))
    if prompt is not None:
        parts.append(prompt.to(device))

    full_prompt = torch.cat(parts) if parts else None

    # generate rollouts
    generated = model.generate(
        num_generate=forward_recursion_steps,
        prompt=full_prompt,
        sample=True,
        temperature=temperature,
        num_rollouts=num_rollouts,
    )
    # shape: (num_rollouts, total_len) where total_len = len(prompt) + forward_recursion_steps

    if generated.dim() == 1:
        generated = generated.unsqueeze(0)

    # extract data tokens: strip BOS, keep prompt + generated tokens
    bos_offset = 1 if bos_token is not None else 0
    data_tokens = generated[
        :, bos_offset:
    ]  # (num_rollouts, prompt_len + forward_recursion_steps)

    # compute empirical frequencies per rollout via one-hot
    one_hot = torch.nn.functional.one_hot(data_tokens.long(), vocab_size).float()
    thetas = one_hot.mean(dim=1)

    return thetas.cpu().numpy()


@torch.no_grad()
def predictive_monte_carlo_theta_chunked(
    model: UnsupervisedPFN,
    vocab_size: int,
    forward_recursion_steps: int,
    num_rollouts: int,
    prompt: Float[torch.Tensor, " K_init"] | None = None,
    bos_token: int | None = None,
    temperature: float = 1.0,
    chunk_size: int = 100,
) -> np.ndarray:
    """Chunked Predictive Monte Carlo over the rollout sample axis.

    Processes rollouts in chunks to avoid OOM on large num_rollouts.

    Returns:
        Theta estimates, shape (num_rollouts, vocab_size).
    """
    all_thetas = []
    remaining = num_rollouts

    while remaining > 0:
        chunk = min(remaining, chunk_size)
        thetas = predictive_monte_carlo_theta(
            model=model,
            vocab_size=vocab_size,
            forward_recursion_steps=forward_recursion_steps,
            num_rollouts=chunk,
            prompt=prompt,
            bos_token=bos_token,
            temperature=temperature,
        )
        all_thetas.append(thetas)
        remaining -= chunk

    return np.concatenate(all_thetas, axis=0)
