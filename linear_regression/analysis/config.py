"""Configuration shared across the sweep-analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass


SOURCE_DISPLAY_LABELS = {
    "discrete": "In-distribution",
    "gaussian": "Out-of-distribution",
    "random": "Random",
}


@dataclass
class SweepConfig:
    """Configuration for sweep analysis experiment."""

    # Checkpoint location
    checkpoint_root: str = "checkpoints/lr/task_diversity"

    # Evaluation parameters
    eval_batch_size: int = 64
    seq_len: int = 64
    noise_std: float = 0.5
    seed: int = 42
    eval_position: int | None = (
        None  # if set, evaluate MSE at this position only (0-indexed); auto-shortens seq_len
    )

    # Distribution comparison parameters
    compute_distribution_metrics: bool = True
    prompt_sources: tuple[str, ...] = ("gaussian", "discrete", "random")
    prompt_lengths: tuple[int, ...] = (0, 8, 16, 32)  # 0 = prior mode
    n_prompts: tuple[int, ...] = (256,)  # number of prompts to average over (posterior)
    n_samples: tuple[int, ...] = (100,)  # samples per prompt (posterior mode)
    n_samples_prior: tuple[int, ...] = (10000,)  # samples for prior mode
    n_projections: int = 100  # for sliced Wasserstein

    # long rollout support (rollouts are clipped to the trained context)
    predictive_steps: int = 256  # number of autoregressive generation steps

    # Marginal CDF plotting
    plot_memorising_marginals: bool = True

    # Random eval
    include_random_eval: bool = True

    # Prompt-based eval: compute MSE on prompt-style sequences from each source
    separate_eval_prompts: bool = False  # draw separate prompt-style data for MSE eval
    eval_n_prompts: int = (
        50  # batch size for prompt-based eval (only when separate_eval_prompts=True)
    )
    eval_prompt_length: int = 8  # sequence length for prompt-based eval (only when separate_eval_prompts=True)

    # Output
    output_dir: str = "outputs/lr/sweep_analysis"
    eval_dataset_dir: str | None = None  # path to eval dataset directory

    @property
    def noise_variance(self) -> float:
        return self.noise_std**2

    def validate(self) -> None:
        """Validate dimensions and sampling settings before creating outputs."""
        if self.eval_batch_size < 1 or self.seq_len < 1:
            raise ValueError("eval_batch_size and seq_len must be positive.")
        if self.noise_std <= 0:
            raise ValueError("noise_std must be positive.")
        if (
            self.eval_position is not None
            and not 0 <= self.eval_position < self.seq_len
        ):
            raise ValueError("eval_position must lie within the evaluation sequence.")
        if not self.prompt_lengths or any(length < 0 for length in self.prompt_lengths):
            raise ValueError("prompt_lengths must contain non-negative lengths.")
        unsupported_sources = set(self.prompt_sources) - set(SOURCE_DISPLAY_LABELS)
        if unsupported_sources:
            raise ValueError(
                f"unsupported prompt sources: {sorted(unsupported_sources)}"
            )
        for name in ("n_prompts", "n_samples", "n_samples_prior"):
            values = getattr(self, name)
            if len(values) != 1 or values[0] < 1:
                raise ValueError(
                    f"{name} must contain one positive value; multiple values would "
                    "overwrite the corresponding sample bundle."
                )
        if self.n_projections < 1 or self.predictive_steps < 1:
            raise ValueError("n_projections and predictive_steps must be positive.")
        if self.eval_n_prompts < 1 or self.eval_prompt_length < 1:
            raise ValueError("eval_n_prompts and eval_prompt_length must be positive.")
