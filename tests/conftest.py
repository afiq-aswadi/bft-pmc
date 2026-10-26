from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from bft_pmc.balls_and_urns import train as bau_train
from bft_pmc.markov.config import MarkovConfig, dump_config
from bft_pmc.markov.data import MarkovChainDataset
from bft_pmc.markov.model import MarkovTransformer


@pytest.fixture
def saved_figures(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Path]]:
    """Record figure destinations without rasterizing high-resolution test plots."""
    paths: list[Path] = []

    def record_save(
        self: Figure,
        filename: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        del self, args, kwargs
        paths.append(Path(filename))

    monkeypatch.setattr(Figure, "savefig", record_save)
    yield paths
    plt.close("all")


@pytest.fixture(scope="session")
def nope_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A three-step NoPE model with a 32-token context, saved by the trainer itself."""
    root = tmp_path_factory.mktemp("bau_nope")
    bau_train.main(
        bau_train.TrainConfig(
            num_tasks=4,
            vocab_size=4,
            d_model=8,
            d_mlp=16,
            n_layers=1,
            n_heads=1,
            d_head=8,
            n_ctx=32,
            pos_encoding="none",
            batch_size=4,
            seq_len=32,
            num_steps=3,
            warmup_steps=0,
            save_every=3,
            checkpoint_root=str(root),
            seed=42,
        )
    )
    return root / "checkpoint_step_3.pt"


@pytest.fixture
def tiny_run(tmp_path: Path) -> Path:
    """An untrained k=2 model with an 8-state context, saved as a training run."""
    config = MarkovConfig(
        k=2,
        seq_len=8,
        n_chains=2,
        batch_size=2,
        d_model=4,
        num_layers=1,
        num_heads=2,
        expansion_factor=2,
        context_len=3,
        num_eval_trials=1,
    )
    dataset = MarkovChainDataset(2, 8, 2, "cpu", seed=config.seed)
    model = MarkovTransformer(3, 4, 8, 1, 2, 2, config.rope_theta)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(model.state_dict(), run_dir / "model.pt")
    dump_config(config, run_dir / "resolved_config.yaml")
    np.save(run_dir / "transition_matrices.npy", dataset.transition_matrices.numpy())
    np.save(
        run_dir / "stationary_distributions.npy",
        dataset.stationary_distributions.numpy(),
    )
    return run_dir
