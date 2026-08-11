from __future__ import annotations

from dataclasses import dataclass

import pytest

from models.pos_encoding import (
    POSITIONAL_ENCODINGS,
    positional_encoding_kwargs,
    positional_encoding_of,
)


@dataclass
class FakeModelConfig:
    use_pos_emb: bool = True
    positional_embedding_type: str = "standard"


def test_positional_encoding_kwargs_cover_every_variant() -> None:
    assert POSITIONAL_ENCODINGS == ("learned", "rope", "none")
    assert positional_encoding_kwargs("learned") == {
        "positional_embedding_type": "standard",
        "use_pos_emb": True,
    }
    assert positional_encoding_kwargs("rope") == {
        "positional_embedding_type": "rotary",
        "use_pos_emb": True,
    }
    assert positional_encoding_kwargs("none") == {
        "positional_embedding_type": "standard",
        "use_pos_emb": False,
    }
    with pytest.raises(ValueError, match="Unknown pos_encoding"):
        positional_encoding_kwargs("alibi")


def test_positional_encoding_round_trips_through_a_model_config() -> None:
    for variant in POSITIONAL_ENCODINGS:
        config = FakeModelConfig(**positional_encoding_kwargs(variant))
        assert positional_encoding_of(config) == variant


def test_positional_encoding_of_defaults_to_learned() -> None:
    # configs predating the flag carry neither attribute
    assert positional_encoding_of(object()) == "learned"
