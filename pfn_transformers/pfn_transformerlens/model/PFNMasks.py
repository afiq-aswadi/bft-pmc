from typing import Callable

import torch
from jaxtyping import Float


def create_custom_mask_hook(
    mask_type: str, query_pos: int, key_pos: int
) -> Callable[
    [Float[torch.Tensor, "query key"], object],
    Float[torch.Tensor, "query key"],
]:
    """Create a custom attention mask hook for PFN models.

    Args:
        mask_type: Type of mask to create ("autoregressive-pfn").
        query_pos: Number of query positions.
        key_pos: Number of key positions.

    Returns:
        Hook function that applies the specified attention mask.
    """

    def autoregressive_pfn_mask(
        attn_scores: Float[torch.Tensor, "query key"], hook: object
    ) -> Float[torch.Tensor, "query key"]:
        device = attn_scores.device

        # Create base causal mask (lower triangular without diagonal)
        lower_no_diag = torch.tril(
            torch.ones(query_pos, key_pos, device=device), diagonal=-1
        )

        # Create mask for even key positions
        even_key_positions = (torch.arange(key_pos, device=device) % 2 - 1 == 0).float()

        # Combine masks - only allow attention to previous even positions
        allowed = lower_no_diag * even_key_positions

        # Allow each position to attend to itself
        allowed.diagonal().fill_(1.0)

        # Convert to boolean mask of positions to ignore
        mask = ~allowed.bool()

        # Apply the mask to attention scores
        attn_scores.masked_fill_(mask, float("-inf"))

        return attn_scores

    if mask_type == "autoregressive-pfn":
        return autoregressive_pfn_mask

    raise ValueError(
        f"Unsupported mask_type '{mask_type}'. Expected 'autoregressive-pfn'."
    )
