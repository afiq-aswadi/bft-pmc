from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Float
from torch.distributions import Distribution

from pfn_transformerlens.model.PFN import SupervisedPFN


def prepare_model_for_long_rollout(
    model: SupervisedPFN,
    rollout_length: int,
    prompt_length: int = 0,
) -> int:
    """Cap rollout length to fit within the model's trained context window.

    Args:
        model: The model (not modified).
        rollout_length: Number of (x, y) pairs to generate after the prompt.
        prompt_length: Number of (x, y) pairs in the prompt.

    Returns:
        Effective rollout length (may be less than requested).
    """
    required_ctx = 2 * (rollout_length + prompt_length)
    current_ctx = model.transformer.cfg.n_ctx

    max_prompt_pairs = current_ctx // 2
    assert max_prompt_pairs > prompt_length, (
        f"Prompt length {prompt_length} exceeds model context capacity "
        f"({max_prompt_pairs} pairs). The model cannot process this prompt."
    )

    if required_ctx <= current_ctx:
        return rollout_length

    return current_ctx // 2 - prompt_length


@torch.no_grad()
def predictive_monte_carlo_beta(
    model: SupervisedPFN,
    x_distribution: Distribution,
    forward_recursion_steps: int,
    forward_recursion_samples: int,
    sample_y: bool = True,
    temperature: float = 1.0,
    init_x: (
        Float[torch.Tensor, "K_init input_dim"]
        | Float[torch.Tensor, "n_prompts K_init input_dim"]
        | None
    ) = None,
    init_y: (
        Float[torch.Tensor, "K_init"] | Float[torch.Tensor, "n_prompts K_init"] | None
    ) = None,
    save_y: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate rollouts and estimate regression weights via least squares.

    Supports both single prompt and batched prompts. Each prompt gets
    forward_recursion_samples independent rollouts.

    Parameters
    ----------
    init_x : tensor or None
        Initial x context. Accepts:
        - Single prompt: (K_init, input_dim)
        - Batched prompts: (n_prompts, K_init, input_dim)
    init_y : tensor or None
        Initial y context. Accepts:
        - Single prompt: (K_init,)
        - Batched prompts: (n_prompts, K_init)

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        - Single prompt or no prompt: (forward_recursion_samples, input_dim)
        - Batched prompts: (n_prompts, forward_recursion_samples, input_dim)
        If save_y=True, also returns y values with matching batch structure.
    """
    if (init_x is None) != (init_y is None):
        raise ValueError("init_x and init_y must both be provided or both be None.")

    # track if we need to squeeze output (single prompt case)
    squeeze_output = False

    if init_x is not None:
        assert init_y is not None
        # handle single prompt: unsqueeze to batch dim
        if init_x.dim() == 2:
            assert init_y.dim() == 1, (
                f"init_y must be 1D for single prompt, got {init_y.shape}"
            )
            init_x = init_x.unsqueeze(0)
            init_y = init_y.unsqueeze(0)
            squeeze_output = True
        # validate batched prompts
        assert init_x.dim() == 3, f"init_x must be 2D or 3D, got {init_x.shape}"
        assert init_y.dim() == 2, f"init_y must be 1D or 2D, got {init_y.shape}"
        assert init_x.shape[0] == init_y.shape[0], (
            f"batch size mismatch: init_x has {init_x.shape[0]}, init_y has {init_y.shape[0]}"
        )
        assert init_x.shape[1] == init_y.shape[1], (
            f"prompt length mismatch: init_x has {init_x.shape[1]}, init_y has {init_y.shape[1]}"
        )
    else:
        # no prompt case: output will be (1, n_samples, D), squeeze to (n_samples, D)
        squeeze_output = True

    assert forward_recursion_steps >= 1, "forward_recursion_steps must be positive"
    assert forward_recursion_samples >= 1, "forward_recursion_samples must be positive"
    assert temperature > 0, "temperature must be > 0"

    device = next(model.parameters()).device

    x_gen, y_gen = model.generate(
        x_distribution=x_distribution,
        num_generate=forward_recursion_steps,
        prompt_x=init_x,
        prompt_y=init_y,
        sample=sample_y,
        temperature=temperature,
        num_rollouts=forward_recursion_samples,
    )

    # output: (n_prompts, forward_recursion_samples, total_len, input_dim)
    x_gen = x_gen.to(device)
    y_gen = y_gen.to(device)

    # solve least squares per prompt per rollout
    y_targets = y_gen.unsqueeze(-1).to(dtype=x_gen.dtype)
    beta_hat = torch.linalg.lstsq(x_gen, y_targets).solution.squeeze(-1)
    # shape: (n_prompts, forward_recursion_samples, input_dim)

    betas_np = beta_hat.detach().cpu().numpy()
    ys_np = y_gen.detach().cpu().numpy()

    # squeeze batch dim for single prompt / no prompt case
    if squeeze_output:
        betas_np = betas_np[0]
        ys_np = ys_np[0]

    if save_y:
        return betas_np, ys_np
    return betas_np


@torch.no_grad()
def predictive_monte_carlo_beta_chunked(
    model: SupervisedPFN,
    x_distribution: Distribution,
    forward_recursion_steps: int,
    forward_recursion_samples: int,
    chunk_size: int = 200,
    sample_y: bool = True,
    temperature: float = 1.0,
    init_x: (
        Float[torch.Tensor, "K_init input_dim"]
        | Float[torch.Tensor, "n_prompts K_init input_dim"]
        | None
    ) = None,
    init_y: (
        Float[torch.Tensor, "K_init"] | Float[torch.Tensor, "n_prompts K_init"] | None
    ) = None,
    save_y: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Run Predictive Monte Carlo in memory-friendly chunks.

    Chunks over the number of samples per prompt, not prompts.
    Supports both single prompt and batched prompts.

    Parameters
    ----------
    init_x : tensor or None
        Initial x context. Accepts:
        - Single prompt: (K_init, input_dim)
        - Batched prompts: (n_prompts, K_init, input_dim)
    init_y : tensor or None
        Initial y context. Accepts:
        - Single prompt: (K_init,)
        - Batched prompts: (n_prompts, K_init)

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        - Single prompt or no prompt: (forward_recursion_samples, input_dim)
        - Batched prompts: (n_prompts, forward_recursion_samples, input_dim)
        If save_y=True, also returns y values with matching batch structure.
    """
    assert chunk_size >= 1, "chunk_size must be positive"
    assert forward_recursion_samples >= 1, "forward_recursion_samples must be positive"

    # detect if single prompt (will get 2D output from base function)
    single_prompt = init_x is None or init_x.dim() == 2

    all_betas: list[np.ndarray] = []
    all_ys: list[np.ndarray] = []

    for start in range(0, forward_recursion_samples, chunk_size):
        batch = min(chunk_size, forward_recursion_samples - start)
        result = predictive_monte_carlo_beta(
            model=model,
            x_distribution=x_distribution,
            forward_recursion_steps=forward_recursion_steps,
            forward_recursion_samples=batch,
            sample_y=sample_y,
            temperature=temperature,
            init_x=init_x,
            init_y=init_y,
            save_y=save_y,
        )

        if save_y:
            betas_chunk, ys_chunk = result
            all_betas.append(betas_chunk)
            all_ys.append(ys_chunk)
        else:
            assert isinstance(result, np.ndarray)
            all_betas.append(result)

    # concatenate along samples dimension
    # single prompt: chunks are (chunk_samples, input_dim), concat axis=0
    # batched: chunks are (n_prompts, chunk_samples, input_dim), concat axis=1
    concat_axis = 0 if single_prompt else 1
    betas = np.concatenate(all_betas, axis=concat_axis)
    if save_y:
        ys = np.concatenate(all_ys, axis=concat_axis)
        return betas, ys
    return betas
