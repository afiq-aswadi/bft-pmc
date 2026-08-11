"""Run posterior-matching comparison analyses for a trained Markov model."""

from dataclasses import dataclass
from pathlib import Path

import tyro

from markov.analysis_common import load_trained_markov_artifacts
from markov.samples_saving import (
    PMCSamplingConfig,
    generate_and_save_pmc_artifacts,
)


@dataclass(slots=True)
class PMCConfig:
    """CLI arguments for the Markov PMC analysis."""

    checkpoint_path: Path
    config_path: Path = Path("markov/train.yaml")
    output_dir: Path = Path("outputs/markov/pmc")
    num_samples: int = 100
    generation_length: int = 223
    prompt_len: int = 32
    seed: int = 0
    save_rollouts: bool = True


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
