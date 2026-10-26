"""Running a trained Markov checkpoint past its context without retraining.

The rotary tables are non-persistent buffers, so they never reach the
checkpoint and a longer context is a construction-time choice, not a weight
change. What differs between the two ``rope_scaling`` modes is what the model
sees at each position, which these tests pin down in both directions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from markov.analysis_common import load_trained_markov_artifacts
from markov.config import MarkovConfig, dump_config
from markov.model import MarkovTransformer, precompute_rotary_frequencies


TRAINED_SEQ_LEN = 32
EXTENDED_SEQ_LEN = 64


def _model(seq_len: int, position_scale: float = 1.0) -> MarkovTransformer:
    torch.manual_seed(0)
    return MarkovTransformer(
        vocab_size=5,
        d_model=16,
        seq_len=seq_len,
        num_layers=2,
        num_heads=2,
        expansion_factor=2,
        rope_theta=10000.0,
        position_scale=position_scale,
    ).eval()


def test_the_rotary_tables_stay_out_of_the_checkpoint() -> None:
    """A longer context is not a different set of saved weights."""
    trained, extended = _model(TRAINED_SEQ_LEN), _model(EXTENDED_SEQ_LEN)
    assert not any("freqs" in key for key in trained.state_dict())
    assert list(trained.state_dict()) == list(extended.state_dict())
    extended.load_state_dict(trained.state_dict())


def test_extrapolation_keeps_the_trained_positions_and_runs_past_them() -> None:
    trained, extended = _model(TRAINED_SEQ_LEN), _model(EXTENDED_SEQ_LEN)
    extended.load_state_dict(trained.state_dict())

    tokens = torch.randint(0, 5, (2, TRAINED_SEQ_LEN))
    with torch.no_grad():
        torch.testing.assert_close(trained(tokens), extended(tokens))
        longer = torch.randint(0, 5, (2, EXTENDED_SEQ_LEN))
        assert extended(longer).shape == (2, EXTENDED_SEQ_LEN, 5)
    with pytest.raises(ValueError, match="exceeds configured maximum"):
        trained(longer)


def test_interpolation_keeps_every_angle_inside_the_trained_range() -> None:
    """Position interpolation is what buys safety at the unseen positions."""
    scale = TRAINED_SEQ_LEN / EXTENDED_SEQ_LEN
    trained_cos, _ = precompute_rotary_frequencies(8, TRAINED_SEQ_LEN, 10000.0)
    interpolated_cos, _ = precompute_rotary_frequencies(
        8, EXTENDED_SEQ_LEN, 10000.0, position_scale=scale
    )
    # Every interpolated position lands on a trained one, two rows apart here.
    torch.testing.assert_close(interpolated_cos[:: int(1 / scale)], trained_cos)

    extrapolated_cos, _ = precompute_rotary_frequencies(8, EXTENDED_SEQ_LEN, 10000.0)
    assert not torch.equal(extrapolated_cos[TRAINED_SEQ_LEN:], interpolated_cos[TRAINED_SEQ_LEN:])


def test_interpolation_also_moves_the_trained_positions() -> None:
    """The cost of interpolating: short rollouts no longer reproduce exactly."""
    trained = _model(TRAINED_SEQ_LEN)
    interpolated = _model(EXTENDED_SEQ_LEN, TRAINED_SEQ_LEN / EXTENDED_SEQ_LEN)
    interpolated.load_state_dict(trained.state_dict())

    tokens = torch.randint(0, 5, (2, TRAINED_SEQ_LEN))
    with torch.no_grad():
        assert not torch.allclose(trained(tokens), interpolated(tokens))


def _checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    """A small trained-looking Markov checkpoint with its resolved config."""
    config = MarkovConfig(
        k=4,
        seq_len=TRAINED_SEQ_LEN,
        n_chains=3,
        d_model=16,
        num_layers=2,
        num_heads=2,
        expansion_factor=2,
        context_len=8,
        seed=0,
    )
    config_path = tmp_path / "resolved_config.yaml"
    dump_config(config, config_path)
    model = _model(TRAINED_SEQ_LEN)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    return config_path, checkpoint_path


def test_the_loader_applies_the_requested_scaling(tmp_path: Path) -> None:
    config_path, checkpoint_path = _checkpoint(tmp_path)
    load = lambda **kwargs: load_trained_markov_artifacts(  # noqa: E731
        config_path, checkpoint_path, allow_seed_rehydration=True, **kwargs
    )

    trained = load()
    assert trained.model.max_seq_len == TRAINED_SEQ_LEN
    assert trained.model.position_scale == 1.0

    interpolated = load(seq_len_override=EXTENDED_SEQ_LEN)
    assert interpolated.model.max_seq_len == EXTENDED_SEQ_LEN
    assert interpolated.model.position_scale == TRAINED_SEQ_LEN / EXTENDED_SEQ_LEN
    # The dataset keeps the trained length, so prompt sampling is untouched.
    assert interpolated.dataset.seq_len == TRAINED_SEQ_LEN

    extrapolated = load(seq_len_override=EXTENDED_SEQ_LEN, rope_scaling="extrapolate")
    assert extrapolated.model.max_seq_len == EXTENDED_SEQ_LEN
    assert extrapolated.model.position_scale == 1.0


def test_the_override_cannot_shrink_the_context(tmp_path: Path) -> None:
    config_path, checkpoint_path = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="must exceed the trained context"):
        load_trained_markov_artifacts(
            config_path,
            checkpoint_path,
            allow_seed_rehydration=True,
            seq_len_override=TRAINED_SEQ_LEN // 2,
        )


def test_an_unknown_scaling_mode_is_refused(tmp_path: Path) -> None:
    config_path, checkpoint_path = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="Unknown rope_scaling"):
        load_trained_markov_artifacts(
            config_path,
            checkpoint_path,
            allow_seed_rehydration=True,
            seq_len_override=EXTENDED_SEQ_LEN,
            rope_scaling="yarn",
        )
