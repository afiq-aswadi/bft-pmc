"""Run posterior-matching comparison analyses for a trained Markov model."""

from dataclasses import dataclass
from pathlib import Path

import tyro

from bft_pmc.markov.analysis_common import load_trained_markov_artifacts
from bft_pmc.markov.samples_saving import (
    PMCSamplingConfig,
    generate_and_save_pmc_artifacts,
)


@dataclass(slots=True)
class PMCConfig:
    """Write prior and posterior PMC samples for one trained Markov checkpoint."""

    checkpoint_path: Path
    """Checkpoint to sample from."""
    config_path: Path = Path("configs/markov/train.yaml")
    """The run's resolved_config.yaml (or the training YAML it was run with)."""
    output_dir: Path = Path("outputs/markov/pmc")
    """Directory for pmc_samples.npz and its rollouts."""
    num_samples: int = 100
    """Rollouts per prompt."""
    generation_length: int = 223
    """Rollout length N in states."""
    prompt_len: int = 32
    """States per posterior prompt."""
    seed: int = 0
    """Seed for prompts and rollouts."""
    save_rollouts: bool = True
    """Also write the raw rollouts behind the samples."""


def main(args: PMCConfig) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_trained_markov_artifacts(
        args.config_path,
        args.checkpoint_path,
    )
    generate_and_save_pmc_artifacts(
        model=artifacts.model,
        dataset=artifacts.dataset,
        sampling=PMCSamplingConfig(
            num_samples=args.num_samples,
            prompt_len=args.prompt_len,
            generation_length=args.generation_length,
            seed=args.seed,
            save_rollouts=args.save_rollouts,
        ),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main(tyro.cli(PMCConfig))
