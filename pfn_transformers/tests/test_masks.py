from __future__ import annotations

from collections.abc import Callable

import torch
from pytest import MonkeyPatch

from pfn_transformerlens.model.configs import SupervisedRegressionPFNConfig
from pfn_transformerlens.model.PFN import PFNModel
from pfn_transformerlens.model.PFNMasks import create_custom_mask_hook


def _make_test_config(seq_len: int, input_dim: int) -> SupervisedRegressionPFNConfig:
    return SupervisedRegressionPFNConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        d_mlp=64,
        d_vocab=32,
        bucket_type="uniform",
        y_min=-5.0,
        y_max=5.0,
        n_ctx=seq_len * 2,
        d_head=32,
        act_fn="gelu",
        normalization_type="LN",
        mask_type="autoregressive-pfn",
        input_dim=input_dim,
    )


def test_autoregressive_mask_allows_only_previous_y_tokens() -> None:
    seq_len = 6
    attn = torch.zeros(seq_len, seq_len)
    mask_hook = create_custom_mask_hook("autoregressive-pfn", seq_len, seq_len)
    masked_scores = mask_hook(attn, object())

    expected = torch.full_like(masked_scores, float("-inf"))
    for query in range(seq_len):
        expected[query, query] = 0.0
        for key in range(query):
            if key % 2 == 1:
                expected[query, key] = 0.0

    assert torch.equal(masked_scores, expected)


def test_model_forward_invokes_mask_hook(monkeypatch: MonkeyPatch) -> None:
    hook_calls: list[int] = []

    def tracking_hook(
        mask_type: str,
        query_pos: int,
        key_pos: int,
    ) -> Callable[[torch.Tensor, object], torch.Tensor]:
        inner_hook = create_custom_mask_hook(mask_type, query_pos, key_pos)

        def wrapped(attn_scores: torch.Tensor, hook: object) -> torch.Tensor:
            hook_calls.append(1)
            return inner_hook(attn_scores, hook)

        return wrapped

    monkeypatch.setattr(
        "pfn_transformerlens.model.PFN.create_custom_mask_hook", tracking_hook
    )

    cfg = _make_test_config(seq_len=4, input_dim=2)
    model = PFNModel(cfg)
    device = model.transformer.cfg.device

    batch = 2
    seq = 4
    x = torch.randn(batch, seq, cfg.input_dim, device=device)
    y = torch.randn(batch, seq, device=device)

    model.forward(x, y)

    assert hook_calls, (
        "Expected custom mask hook to be invoked during PFN forward pass."
    )


def test_attention_scores_follow_mask() -> None:
    batch = 1
    seq = 3
    cfg = _make_test_config(seq_len=seq, input_dim=2)
    model = PFNModel(cfg)
    device = model.transformer.cfg.device

    x = torch.randn(batch, seq, cfg.input_dim, device=device)
    y = torch.randn(batch, seq, device=device)

    _, cache = model._forward_autoregressive_pfn(x, y, return_cache=True)
    scores = cache["blocks.0.attn.hook_attn_scores"]
    score_mask = torch.isneginf(scores[0, 0]).cpu()

    tokens = 2 * seq
    allowed = torch.zeros(tokens, tokens, dtype=torch.bool)
    for query in range(tokens):
        allowed[query, query] = True
        for key in range(query):
            if key % 2 == 1:
                allowed[query, key] = True

    expected_mask = (~allowed).to(score_mask.device)
    assert torch.equal(
        score_mask[:tokens, :tokens],
        expected_mask,
    ), "Attention scores should be -inf exactly on disallowed positions."
