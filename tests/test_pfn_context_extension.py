from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.pos_encoding import positional_encoding_kwargs
from pfn_transformerlens import checkpointing
from pfn_transformerlens.model.PFN import PFNModel
from pfn_transformerlens.model.configs import (
    BasePFNConfig,
    SupervisedRegressionPFNConfig,
    UnsupervisedPFNConfig,
)


ENCODINGS = ("learned", "rope", "none")


def _common(pos_encoding: str) -> dict[str, object]:
    return {
        "n_layers": 1,
        "d_model": 8,
        "d_head": 4,
        "n_heads": 2,
        "d_mlp": 8,
        "n_ctx": 8,
        "act_fn": "gelu",
        "device": "cpu",
        **positional_encoding_kwargs(pos_encoding),
    }


def _sequence_config(pos_encoding: str) -> UnsupervisedPFNConfig:
    return UnsupervisedPFNConfig(
        **_common(pos_encoding),
        d_vocab=5,
        d_vocab_out=4,
        input_type="discrete",
        prediction_type="distribution",
    )


def _pair_config(pos_encoding: str) -> SupervisedRegressionPFNConfig:
    return SupervisedRegressionPFNConfig(
        **_common(pos_encoding),
        d_vocab=6,
        input_dim=2,
        prediction_type="distribution",
        mask_type="autoregressive-pfn",
        bucket_type="uniform",
        bucket_support="bounded",
        y_min=-1.0,
        y_max=1.0,
    )


def _save(config: BasePFNConfig, path: Path) -> Path:
    """Save a model whose weights, positions included, are far from initialisation."""
    torch.manual_seed(0)
    model = PFNModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
        if not config.use_pos_emb:
            model.transformer.W_pos.zero_()
    checkpointing.save_checkpoint(
        path,
        1,
        model.state_dict(),
        {},
        config,
        None,
        checkpointing.CheckpointMetadata("", None, None, None, None),
    )
    return path


@pytest.mark.parametrize("pos_encoding", ENCODINGS)
def test_a_sequence_model_keeps_its_outputs_and_runs_past_the_trained_context(
    pos_encoding: str, tmp_path: Path
) -> None:
    path = _save(_sequence_config(pos_encoding), tmp_path / "model.pt")
    original, _, _ = checkpointing.load_checkpoint(path, device="cpu")
    extended, _, _ = checkpointing.load_checkpoint(path, device="cpu", n_ctx_override=32)
    assert extended.transformer.cfg.n_ctx == 32

    tokens = torch.randint(0, 5, (3, 8)).float()
    with torch.no_grad():
        torch.testing.assert_close(original(tokens), extended(tokens))
        assert extended(torch.randint(0, 5, (3, 20)).float()).shape == (3, 20, 4)
    with pytest.raises(RuntimeError, match="size of tensor"):
        original(torch.randint(0, 5, (3, 20)).float())


@pytest.mark.parametrize("pos_encoding", ENCODINGS)
def test_a_pair_model_keeps_its_outputs_and_runs_past_the_trained_context(
    pos_encoding: str, tmp_path: Path
) -> None:
    path = _save(_pair_config(pos_encoding), tmp_path / "model.pt")
    original, _, _ = checkpointing.load_checkpoint(path, device="cpu")
    extended, _, _ = checkpointing.load_checkpoint(path, device="cpu", n_ctx_override=32)

    x, y = torch.randn(3, 4, 2), torch.randn(3, 4)
    with torch.no_grad():
        torch.testing.assert_close(original(x, y), extended(x, y))
        assert extended(torch.randn(3, 10, 2), torch.randn(3, 10)).shape == (3, 10, 6)


def test_a_learned_sequence_table_repeats_its_last_position(tmp_path: Path) -> None:
    path = _save(_sequence_config("learned"), tmp_path / "model.pt")
    trained = torch.load(path, weights_only=False)["model_state_dict"]["transformer.pos_embed.W_pos"]
    extended, _, _ = checkpointing.load_checkpoint(path, device="cpu", n_ctx_override=12)
    table = extended.transformer.W_pos.detach()
    torch.testing.assert_close(table[:8], trained)
    torch.testing.assert_close(table[8:], trained[7].expand(4, -1))


def test_a_learned_pair_table_keeps_the_x_and_y_slots(tmp_path: Path) -> None:
    """Interleaved models reuse the last x-slot row for x tokens and the last y-slot row for y."""
    path = _save(_pair_config("learned"), tmp_path / "model.pt")
    trained = torch.load(path, weights_only=False)["model_state_dict"]["transformer.pos_embed.W_pos"]
    extended, _, _ = checkpointing.load_checkpoint(path, device="cpu", n_ctx_override=14)
    table = extended.transformer.W_pos.detach()
    torch.testing.assert_close(table[:8], trained)
    torch.testing.assert_close(table[8::2], trained[6].expand(3, -1))
    torch.testing.assert_close(table[9::2], trained[7].expand(3, -1))


def test_extend_positional_table_follows_the_period() -> None:
    table = torch.arange(4.0).unsqueeze(-1)
    assert checkpointing.extend_positional_table(table, 7, 1).squeeze(-1).tolist() == [
        0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0,
    ]
    assert checkpointing.extend_positional_table(table, 8, 2).squeeze(-1).tolist() == [
        0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0,
    ]


def test_the_override_cannot_shrink_the_context(tmp_path: Path) -> None:
    path = _save(_sequence_config("rope"), tmp_path / "model.pt")
    with pytest.raises(ValueError, match="must exceed the trained context"):
        checkpointing.load_checkpoint(path, device="cpu", n_ctx_override=4)


def test_a_checkpoint_with_foreign_weights_is_refused(tmp_path: Path) -> None:
    path = _save(_sequence_config("none"), tmp_path / "model.pt")
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["model_state_dict"]["unexpected.weight"] = torch.zeros(1)
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="unexpected"):
        checkpointing.load_checkpoint(path, device="cpu", n_ctx_override=16)
