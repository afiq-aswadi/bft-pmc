"""Configuration helpers for Markov-chain experiments."""

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import yaml


@dataclass(slots=True)
class MarkovConfig:
    """Core configuration for Markov-chain training and evaluation."""

    # task
    k: int = 10
    seq_len: int = 512
    n_chains: int = 128

    # optimization
    batch_size: int = 128
    learning_rate: float = 0.0006
    eval_interval: int = 200
    max_steps: int = 1000

    # model
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    expansion_factor: int = 4
    rope_theta: float = 10000.0

    # evaluation
    context_len: int = 400
    num_eval_trials: int = 30
    # baseline-delta eval uses a separate (smaller) batch because bi_ret scales
    # its einsum with n_chains and can OOM at training batch_size when n_chains
    # is large. kept small by default; override in config.yaml if plenty of VRAM.
    delta_eval_batch_size: int = 16

    # reproducibility
    seed: int = 42

    def validate(self) -> None:
        """Validate constraints that would otherwise fail deep in training."""
        if self.k < 2:
            raise ValueError("k must be at least 2.")
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2.")
        if self.n_chains < 1:
            raise ValueError("n_chains must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.eval_interval < 1:
            raise ValueError("eval_interval must be at least 1.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.d_model < 1:
            raise ValueError("d_model must be positive.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if self.num_heads < 1:
            raise ValueError("num_heads must be at least 1.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.expansion_factor < 1:
            raise ValueError("expansion_factor must be at least 1.")
        if self.context_len < 2:
            raise ValueError("context_len must be at least 2.")
        if self.context_len + 1 > self.seq_len:
            raise ValueError(
                "context_len + 1 must be less than or equal to seq_len "
                "when using a BOS token."
            )
        if self.num_eval_trials < 1:
            raise ValueError("num_eval_trials must be at least 1.")
        if self.delta_eval_batch_size < 1:
            raise ValueError("delta_eval_batch_size must be at least 1.")


def load_config(path: str | Path) -> MarkovConfig:
    """Load a Markov experiment config from YAML."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise TypeError(
            f"Expected a mapping in {config_path}, got {type(raw_config)!r}."
        )

    valid_keys = {field.name for field in fields(MarkovConfig)}
    unknown_keys = sorted(set(raw_config) - valid_keys)
    if unknown_keys:
        raise KeyError(
            f"Unknown Markov config keys in {config_path}: {', '.join(unknown_keys)}"
        )

    config = MarkovConfig(**raw_config)
    config.validate()
    return config


def apply_overrides(
    config: MarkovConfig,
    **overrides: int | float | None,
) -> MarkovConfig:
    """Return a validated config with any non-None overrides applied."""
    cleaned_overrides = {
        key: value for key, value in overrides.items() if value is not None
    }
    updated = replace(config, **cleaned_overrides)
    updated.validate()
    return updated


def dump_config(config: MarkovConfig, path: str | Path) -> None:
    """Write a config snapshot to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
