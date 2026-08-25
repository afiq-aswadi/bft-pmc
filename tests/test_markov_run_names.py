from __future__ import annotations

import pytest

from markov.run_names import parse_n_chains


def test_parses_both_naming_conventions() -> None:
    """Pre-2026-08 sweeps wrote chains<N>; the 2026-08 sweeps write M<N>."""
    assert parse_n_chains("markov_wd83ka_chains1024_seed0") == 1024
    assert parse_n_chains("run_chains2") == 2
    assert parse_n_chains("markov_M1024") == 1024
    assert parse_n_chains("markov_M4") == 4


def test_rejects_names_carrying_no_task_count() -> None:
    for name in ("unparseable", "markov_MX", "chains", "markov_"):
        with pytest.raises(ValueError, match="cannot parse n_chains"):
            parse_n_chains(name)
