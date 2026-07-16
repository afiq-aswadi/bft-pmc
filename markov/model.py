"""Small decoder-only transformer for Markov-chain next-token prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rotary_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary embedding frequencies up to the max sequence length."""
    if head_dim % 2 != 0:
        raise ValueError("Rotary embeddings require an even head dimension.")

    inverse_frequency = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    frequencies = torch.outer(positions, inverse_frequency)
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return frequencies.cos(), frequencies.sin()


def apply_rotary_embedding(
    tensor: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to query or key tensors."""
    cos = (
        freqs_cos.unsqueeze(0)
        .unsqueeze(2)
        .to(
            device=tensor.device,
            dtype=tensor.dtype,
        )
    )
    sin = (
        freqs_sin.unsqueeze(0)
        .unsqueeze(2)
        .to(
            device=tensor.device,
            dtype=tensor.dtype,
        )
    )

    half_dim = tensor.shape[-1] // 2
    tensor_left = tensor[..., :half_dim]
    tensor_right = tensor[..., half_dim:]
    rotated = torch.cat((-tensor_right, tensor_left), dim=-1)
    return (tensor * cos) + (rotated * sin)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with rotary position embeddings."""

    freqs_cos: torch.Tensor
    freqs_sin: torch.Tensor

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        freqs_cos, freqs_sin = precompute_rotary_frequencies(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if seq_len > self.freqs_cos.shape[0]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured maximum "
                f"{self.freqs_cos.shape[0]}."
            )

        query = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        query = apply_rotary_embedding(
            query,
            self.freqs_cos[:seq_len],
            self.freqs_sin[:seq_len],
        ).transpose(1, 2)
        key = apply_rotary_embedding(
            key,
            self.freqs_cos[:seq_len],
            self.freqs_sin[:seq_len],
        ).transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)


class DecoderBlock(nn.Module):
    """Transformer decoder block with pre-norm attention and MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        expansion_factor: int,
        max_seq_len: int,
        theta: float,
    ) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, expansion_factor * d_model),
            nn.ReLU(),
            nn.Linear(expansion_factor * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MarkovTransformer(nn.Module):
    """Decoder-only transformer for discrete Markov sequences."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        seq_len: int,
        num_layers: int,
        num_heads: int,
        expansion_factor: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self._max_seq_len = seq_len
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    expansion_factor=expansion_factor,
                    max_seq_len=seq_len,
                    theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Linear(d_model, vocab_size)

    @property
    def max_seq_len(self) -> int:
        """Maximum input length supported by the transformer."""
        return self._max_seq_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
