"""Positional-encoding variants for the LR and balls-and-urns transformers.

The three variants isolate what positional information the model can use:

- ``learned``: a trained absolute position embedding added to the residual
  stream (the historical default for both families);
- ``rope``: rotary embeddings applied inside attention, as in the Markov model;
- ``none``: no positional signal at all, leaving the model permutation
  invariant over the sequence.

Variants are stored in the model config that ships inside every checkpoint, so
:func:`positional_encoding_of` recovers the variant during analysis without
relying on run names or directory layout.
"""

from __future__ import annotations

from typing import Any, Literal


PositionalEncoding = Literal["learned", "rope", "none"]

POSITIONAL_ENCODINGS: tuple[PositionalEncoding, ...] = ("learned", "rope", "none")


def positional_encoding_kwargs(pos_encoding: str) -> dict[str, Any]:
    """Model-config keyword arguments implementing one positional-encoding variant.

    ``rotary_dim`` is left unset: HookedTransformerConfig defaults it to
    ``d_head`` when the embedding type is rotary.
    """
    if pos_encoding not in POSITIONAL_ENCODINGS:
        raise ValueError(
            f"Unknown pos_encoding {pos_encoding!r}; "
            f"expected one of {', '.join(POSITIONAL_ENCODINGS)}."
        )
    return {
        "positional_embedding_type": "rotary" if pos_encoding == "rope" else "standard",
        "use_pos_emb": pos_encoding != "none",
    }


def positional_encoding_of(model_config: Any) -> PositionalEncoding:
    """Recover the variant from a model config, e.g. one loaded from a checkpoint."""
    if not getattr(model_config, "use_pos_emb", True):
        return "none"
    if getattr(model_config, "positional_embedding_type", "standard") == "rotary":
        return "rope"
    return "learned"
