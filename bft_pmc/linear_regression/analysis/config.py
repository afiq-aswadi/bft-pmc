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
    """Evaluate the final checkpoint of every run in a linear-regression sweep.

    Prediction-space metrics (Delta MSE against both references) come from a
    fixed evaluation batch; latent-space metrics (energy distance and sliced
    Wasserstein between the PMC samples and both reference posteriors) from
    PMC rollouts after prompts of each source. Results go to a timestamped
    directory under output_dir.
    """

    checkpoint_root: str = "checkpoints/lr/task_diversity"
    """Directory holding one run directory per task diversity; the latest
    checkpoint of each is evaluated."""
    eval_batch_size: int = 64
    """Sequences in the prediction-space evaluation batch."""
    seq_len: int = 64
    """(x, y) pairs per prediction-space evaluation sequence."""
    noise_std: float = 0.5
    """Observation-noise standard deviation of the training data."""
    seed: int = 42
    """Seed for evaluation data, prompts and rollouts."""
    eval_position: int | None = None
    """Score the prediction-space MSE at this position only (0-indexed); the
    evaluation sequences are shortened to end there."""
    compute_distribution_metrics: bool = True
    """Run PMC and the latent-space comparison, writing sample bundles."""
    prompt_sources: tuple[str, ...] = ("gaussian", "discrete", "random")
    """Where posterior prompts come from: gaussian (a fresh task from N(0, I),
    out-of-distribution), discrete (a task from the training pool,
    in-distribution) or random (responses drawn independently of x)."""
    prompt_lengths: tuple[int, ...] = (0, 32)
    """Prompt lengths in pairs; 0 is the prior."""
    n_prompts: tuple[int, ...] = (128,)
    """Posterior prompts per source (a single value)."""
    n_samples: tuple[int, ...] = (100,)
    """Rollouts per posterior prompt (a single value)."""
    n_samples_prior: tuple[int, ...] = (1024,)
    """Rollouts for the prior (a single value)."""
    n_projections: int = 100
    """Random projections for sliced Wasserstein."""
    chunk_size: int = 100
    """Rollouts per prompt in one generation batch. Generation expands prompts
    x samples into one batch, so peak attention memory scales with
    prompt_chunk_size x chunk_size; 4 x 100 fits a 16 GB GPU at 510 positions."""
    prompt_chunk_size: int = 4
    """Prompts in one generation batch; see chunk_size."""
    predictive_steps: int = 223
    """Rollout length N in pairs. A 32-pair prompt plus 223 generated pairs
    fills the 256-pair (512-token) training context."""
    plot_memorising_marginals: bool = True
    """Draw the memorizing reference in the per-run marginal plots."""
    include_random_eval: bool = True
    """Also score prediction-space MSE on random-response sequences."""
    separate_eval_prompts: bool = False
    """Score prediction-space MSE on prompt-style sequences from each source
    (eval_n_prompts of length eval_prompt_length) instead of the shared
    evaluation batch."""
    eval_n_prompts: int = 50
    """Prompts per source when separate_eval_prompts is set."""
    eval_prompt_length: int = 8
    """Pairs per prompt when separate_eval_prompts is set."""
    output_dir: str = "outputs/lr/sweep_analysis"
    """Parent of the timestamped results directory."""
    eval_dataset_dir: str | None = None
    """Reuse fixed evaluation data from this directory (for the original paper
    checkpoints, paper_data/lr/eval_datasets); None generates it in memory."""
    save_rollouts: bool = True
    """Write the raw rollouts beside each sample bundle as
    <bundle>_rollouts.npz. They dominate a sweep's size on disk; turn off when
    only the estimates are needed."""

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
        if self.chunk_size < 1 or self.prompt_chunk_size < 1:
            raise ValueError("chunk_size and prompt_chunk_size must be positive.")
        if self.eval_n_prompts < 1 or self.eval_prompt_length < 1:
            raise ValueError("eval_n_prompts and eval_prompt_length must be positive.")
