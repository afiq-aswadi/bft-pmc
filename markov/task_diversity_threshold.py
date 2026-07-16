"""Task-diversity threshold experiment for Markov-chain training."""

from dataclasses import asdict, dataclass
from pathlib import Path
import csv

import tyro
import yaml

from markov.config import apply_overrides, dump_config, load_config
from markov.plotting import plot_task_diversity, plot_task_diversity_heatmap
from markov.samples_saving import PMCSamplingConfig
from markov.train import TrainingRunResult, run_training_job


@dataclass(slots=True)
class ThresholdExperimentConfig:
    """Configuration for sweeping over task diversity in Markov experiments."""

    n_chains_values: list[int]
    max_steps: int | None = None
    eval_interval: int | None = None
    gap_tolerance: float = 0.05

    def validate(self) -> None:
        if not self.n_chains_values:
            raise ValueError("n_chains_values must contain at least one value.")
        if any(value < 1 for value in self.n_chains_values):
            raise ValueError("All n_chains values must be at least 1.")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be at least 1 when provided.")
        if self.eval_interval is not None and self.eval_interval < 1:
            raise ValueError("eval_interval must be at least 1 when provided.")
        if self.gap_tolerance < 0:
            raise ValueError("gap_tolerance must be non-negative.")


@dataclass(slots=True)
class ThresholdConfig:
    """CLI arguments for the task-diversity threshold experiment."""

    config_path: str = "markov/train.yaml"
    threshold_config_path: str = "markov/task_diversity_threshold.yaml"

    checkpoint_root: str = "checkpoints/markov"
    output_root: str = "outputs/markov"
    experiment_name: str = "task_diversity_threshold"

    # optional base-config overrides applied to every run in the sweep
    seq_len: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    eval_interval: int | None = None
    max_steps: int | None = None
    seed: int | None = None

    use_wandb: bool = False
    wandb_project: str = "markov-task-diversity-threshold"

    pmc_num_samples: int = 128
    pmc_prompt_len: int = 8
    pmc_generation_length: int = 400

    def validate(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty.")
        if self.pmc_num_samples < 1:
            raise ValueError("pmc_num_samples must be positive.")
        if self.pmc_prompt_len < 0:
            raise ValueError("pmc_prompt_len must be non-negative.")
        if self.pmc_generation_length < 2:
            raise ValueError("pmc_generation_length must be at least 2.")


def load_threshold_config(path: str | Path) -> ThresholdExperimentConfig:
    """Load the threshold experiment YAML config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise TypeError(
            f"Expected a mapping in {config_path}, got {type(raw_config)!r}."
        )

    valid_keys = {"n_chains_values", "max_steps", "eval_interval", "gap_tolerance"}
    unknown_keys = sorted(set(raw_config) - valid_keys)
    if unknown_keys:
        raise KeyError(
            "Unknown threshold experiment keys in "
            f"{config_path}: {', '.join(unknown_keys)}"
        )

    config = ThresholdExperimentConfig(**raw_config)
    config.validate()
    return config


def estimate_threshold(
    rows: list[dict[str, str | int | float]],
    gap_tolerance: float,
) -> int | None:
    """Return the smallest task diversity whose final gap is below tolerance."""
    for row in rows:
        if float(row["generalization_gap"]) <= gap_tolerance:
            return int(row["n_chains"])
    return None


def write_summary_csv(
    rows: list[dict[str, str | int | float]],
    summary_csv_path: Path,
) -> None:
    """Write the threshold sweep summary to CSV."""
    fieldnames = [
        "n_chains",
        "run_name",
        "final_train_loss",
        "final_id_kl",
        "final_ood_kl",
        "generalization_gap",
        "checkpoint_dir",
        "output_dir",
        "latest_checkpoint_path",
        "final_model_path",
        "pmc_samples_path",
        "csv_path",
        "figure_path",
    ]

    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_from_result(
    n_chains: int, result: TrainingRunResult
) -> dict[str, str | int | float]:
    """Convert a single-run result into a summary CSV row."""
    return {
        "n_chains": n_chains,
        "run_name": result.run_name,
        "final_train_loss": result.final_train_loss,
        "final_id_kl": result.final_id_kl,
        "final_ood_kl": result.final_ood_kl,
        "generalization_gap": result.final_ood_kl - result.final_id_kl,
        "checkpoint_dir": str(result.checkpoint_dir),
        "output_dir": str(result.output_dir),
        "latest_checkpoint_path": (
            ""
            if result.latest_checkpoint_path is None
            else str(result.latest_checkpoint_path)
        ),
        "final_model_path": str(result.final_model_path),
        "pmc_samples_path": (
            "" if result.pmc_samples_path is None else str(result.pmc_samples_path)
        ),
        "csv_path": str(result.csv_path),
        "figure_path": str(result.figure_path),
    }


def main(args: ThresholdConfig) -> None:
    args.validate()
    base_config = load_config(args.config_path)
    threshold_config = load_threshold_config(args.threshold_config_path)
    effective_max_steps = (
        args.max_steps if args.max_steps is not None else threshold_config.max_steps
    )
    effective_eval_interval = (
        args.eval_interval
        if args.eval_interval is not None
        else threshold_config.eval_interval
    )
    base_config = apply_overrides(
        base_config,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_interval=effective_eval_interval,
        max_steps=effective_max_steps,
        seed=args.seed,
    )

    experiment_output_dir = Path(args.output_root) / args.experiment_name
    experiment_checkpoint_root = Path(args.checkpoint_root) / args.experiment_name
    runs_output_root = experiment_output_dir / "runs"

    experiment_output_dir.mkdir(parents=True, exist_ok=True)
    experiment_checkpoint_root.mkdir(parents=True, exist_ok=True)

    dump_config(base_config, experiment_output_dir / "base_train_config.yaml")
    with (experiment_output_dir / "threshold_config.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(asdict(threshold_config), handle, sort_keys=False)

    summary_rows: list[dict[str, str | int | float]] = []
    ordered_n_chains = sorted(set(threshold_config.n_chains_values))

    for n_chains in ordered_n_chains:
        run_name = f"{args.experiment_name}_nchains_{n_chains}"
        run_config = apply_overrides(base_config, n_chains=n_chains)
        result = run_training_job(
            config=run_config,
            checkpoint_root=experiment_checkpoint_root,
            output_root=runs_output_root,
            run_name=run_name,
            pmc_sampling=PMCSamplingConfig(
                num_samples=args.pmc_num_samples,
                prompt_len=args.pmc_prompt_len,
                generation_length=args.pmc_generation_length,
            ),
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
        )
        summary_rows.append(row_from_result(n_chains, result))

    threshold_n_chains = estimate_threshold(
        summary_rows,
        threshold_config.gap_tolerance,
    )

    summary_csv_path = experiment_output_dir / "threshold_summary.csv"
    write_summary_csv(summary_rows, summary_csv_path)

    summary_figure_path = experiment_output_dir / "threshold_summary.png"
    plot_task_diversity(
        summary_csv_path,
        summary_figure_path,
        max_steps=base_config.max_steps,
        context_len=base_config.context_len,
    )
    heatmap_figure_path = experiment_output_dir / "task_diversity_heatmap.png"
    plot_task_diversity_heatmap(
        summary_csv_path,
        heatmap_figure_path,
        gap_tolerance=threshold_config.gap_tolerance,
    )

    with (experiment_output_dir / "threshold_report.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        if threshold_n_chains is None:
            handle.write(
                "No threshold found within the configured generalization gap tolerance.\n"
            )
        else:
            handle.write(
                "Estimated threshold n_chains="
                f"{threshold_n_chains} "
                f"(gap_tolerance={threshold_config.gap_tolerance}).\n"
            )


if __name__ == "__main__":
    main(tyro.cli(ThresholdConfig))
