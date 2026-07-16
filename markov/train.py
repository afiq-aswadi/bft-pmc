"""Train one transient Markov PFN and record its dynamics."""

from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Protocol
import csv

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from tqdm import tqdm

from markov.config import MarkovConfig, apply_overrides, dump_config, load_config
from markov.data import MarkovChainDataset
from markov.evals import evaluate_baseline_deltas, evaluate_kl
from markov.model import MarkovTransformer
from markov.plotting import plot_transient
from markov.samples_saving import (
    PMCSamplingConfig,
    generate_and_save_pmc_artifacts,
    resolve_pmc_sampling_config,
)


class WandbRun(Protocol):
    """Subset of the W&B run interface used by training."""

    def log(self, data: dict[str, object]) -> None: ...

    def finish(self) -> None: ...


@dataclass(slots=True)
class TrainingRunResult:
    """Artifacts and final metrics produced by a single training run."""

    run_name: str
    checkpoint_dir: Path
    output_dir: Path
    csv_path: Path
    figure_path: Path
    final_model_path: Path
    pmc_samples_path: Path | None
    latest_checkpoint_path: Path | None
    final_train_loss: float
    final_id_kl: float
    final_ood_kl: float


@dataclass(slots=True)
class TrainConfig:
    """CLI arguments for the transient Markov experiment."""

    config_path: str = "markov/train.yaml"

    # sweep-friendly overrides for the YAML defaults
    k: int | None = None
    seq_len: int | None = None
    n_chains: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    eval_interval: int | None = None
    max_steps: int | None = None
    seed: int | None = None

    # output locations
    checkpoint_root: str = "checkpoints/markov/task_diversity"
    output_root: str = "outputs/markov/training"
    run_name: str | None = None
    # appended to the resolved run name (auto-generated or --run-name); useful for
    # retagging sweep retries so wandb treats them as fresh runs.
    run_name_suffix: str = ""

    # PMC sample saving
    pmc_num_samples: int = 128
    pmc_prompt_len: int = 8
    pmc_generation_length: int = 400

    # logging
    use_wandb: bool = False
    wandb_project: str = "markov-transformer-sweeps"

    def validate(self) -> None:
        if self.pmc_num_samples < 1:
            raise ValueError("pmc_num_samples must be positive.")
        if self.pmc_prompt_len < 0:
            raise ValueError("pmc_prompt_len must be non-negative.")
        if self.pmc_generation_length < 2:
            raise ValueError("pmc_generation_length must be at least 2.")
        if "/" in self.run_name_suffix:
            raise ValueError("run_name_suffix must not contain path separators.")


def build_run_name(config: MarkovConfig) -> str:
    """Create a readable run name for checkpoints and outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    learning_rate = f"{config.learning_rate:.0e}".replace("+", "")
    return (
        f"markov_k{config.k}_chains{config.n_chains}_"
        f"seq{config.seq_len}_lr{learning_rate}_{timestamp}"
    )


def resolve_config(args: TrainConfig) -> MarkovConfig:
    """Load YAML config and apply any non-None CLI overrides."""
    base_config = load_config(args.config_path)
    return apply_overrides(
        base_config,
        k=args.k,
        seq_len=args.seq_len,
        n_chains=args.n_chains,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        max_steps=args.max_steps,
        seed=args.seed,
    )


def initialize_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: str,
    config: MarkovConfig,
) -> WandbRun | None:
    """Initialize W&B when requested."""
    if not enabled:
        return None
    wandb_module = import_module("wandb")
    return wandb_module.init(
        project=project,
        name=run_name,
        config=asdict(config),
    )


def _wandb_image(path: Path) -> object:
    """Construct a W&B image without importing W&B for untracked runs."""
    return import_module("wandb").Image(str(path))


def save_checkpoint(
    *,
    model: MarkovTransformer,
    optimizer: optim.Optimizer,
    config: MarkovConfig,
    step: int,
    checkpoint_dir: Path,
) -> Path:
    """Save a resumable checkpoint with model, optimizer, and config state."""
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def run_training_job(
    *,
    config: MarkovConfig,
    checkpoint_root: str | Path,
    output_root: str | Path,
    run_name: str | None = None,
    run_name_suffix: str = "",
    pmc_sampling: PMCSamplingConfig | None = None,
    use_wandb: bool = False,
    wandb_project: str = "markov-transformer-sweeps",
    device: torch.device | str | None = None,
) -> TrainingRunResult:
    """Train one Markov model and return the produced artifacts."""
    config.validate()
    resolved_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    resolved_run_name = run_name or build_run_name(config)
    if run_name_suffix:
        resolved_run_name = f"{resolved_run_name}-{run_name_suffix}"
    checkpoint_dir = Path(checkpoint_root) / resolved_run_name
    output_dir = Path(output_root) / resolved_run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    dump_config(config, output_dir / "resolved_config.yaml")

    run = initialize_wandb(
        enabled=use_wandb,
        project=wandb_project,
        run_name=resolved_run_name,
        config=config,
    )

    dataset = MarkovChainDataset(
        num_states=config.k,
        seq_len=config.seq_len,
        num_chains=config.n_chains,
        device=resolved_device,
        seed=config.seed,
    )

    # persist the training task distribution alongside the config so analysis
    # scripts do not need to replay the seed inside MarkovChainDataset.__init__
    np.save(
        output_dir / "transition_matrices.npy",
        dataset.transition_matrices.detach().cpu().numpy(),
    )
    np.save(
        output_dir / "stationary_distributions.npy",
        dataset.stationary_distributions.detach().cpu().numpy(),
    )

    model = MarkovTransformer(
        vocab_size=dataset.vocab_size,
        d_model=config.d_model,
        seq_len=config.seq_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        expansion_factor=config.expansion_factor,
        rope_theta=config.rope_theta,
    ).to(resolved_device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    csv_path = output_dir / "training_log.csv"
    history_rows: list[dict[str, float | int]] = []
    last_checkpoint_step = 0
    latest_checkpoint_path: Path | None = None

    # baseline-delta columns: KL(model || baseline) for the four canonical
    # Markov predictors, split by ID vs OOD eval batches. The primary axis is
    # memorising vs generalising (which strategy is the model using); the
    # secondary axis is wellspec (order-1 bigram, matches the DGP) vs misspec
    # (order-0 unigram, ignores transitions). See markov/evals.py.
    _delta_keys = (
        "wellspec_generalising",
        "wellspec_memorising",
        "misspec_generalising",
        "misspec_memorising",
    )
    _delta_columns = [
        f"kl_{name}_{split}" for split in ("id", "ood") for name in _delta_keys
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["step", "n_chains", "train_loss", "id_kl", "ood_kl", *_delta_columns]
        )

        model.train()
        for step in tqdm(range(1, config.max_steps + 1), desc="Training"):
            batch = dataset.sample_batch(config.batch_size)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, dataset.vocab_size),
                targets.reshape(-1),
            )
            loss.backward()
            optimizer.step()

            should_evaluate = (
                step == 1
                or step % config.eval_interval == 0
                or step == config.max_steps
            )
            if not should_evaluate:
                continue

            id_kl = evaluate_kl(
                model=model,
                dataset=dataset,
                num_evals=config.num_eval_trials,
                context_len=config.context_len,
                is_ood=False,
            )
            ood_kl = evaluate_kl(
                model=model,
                dataset=dataset,
                num_evals=config.num_eval_trials,
                context_len=config.context_len,
                is_ood=True,
            )
            id_deltas = evaluate_baseline_deltas(
                model=model,
                dataset=dataset,
                batch_size=config.delta_eval_batch_size,
                is_ood=False,
            )
            ood_deltas = evaluate_baseline_deltas(
                model=model,
                dataset=dataset,
                batch_size=config.delta_eval_batch_size,
                is_ood=True,
            )

            row = {
                "step": step,
                "train_loss": float(loss.item()),
                "id_kl": id_kl,
                "ood_kl": ood_kl,
            }
            for name in _delta_keys:
                row[f"kl_{name}_id"] = id_deltas[name]
                row[f"kl_{name}_ood"] = ood_deltas[name]
            history_rows.append(row)
            writer.writerow(
                [step, config.n_chains, loss.item(), id_kl, ood_kl]
                + [id_deltas[name] for name in _delta_keys]
                + [ood_deltas[name] for name in _delta_keys]
            )
            handle.flush()

            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                checkpoint_dir=checkpoint_dir,
            )
            last_checkpoint_step = step
            latest_checkpoint_path = checkpoint_path

            if run is not None:
                wandb_log = {
                    "step": step,
                    "train_loss": loss.item(),
                    "id_kl": id_kl,
                    "ood_kl": ood_kl,
                }
                for name in _delta_keys:
                    spec, role = name.split("_", 1)
                    wandb_log[f"kl/id/{spec}/{role}"] = id_deltas[name]
                    wandb_log[f"kl/ood/{spec}/{role}"] = ood_deltas[name]
                run.log(wandb_log)

            model.train()

    assert last_checkpoint_step == config.max_steps
    assert latest_checkpoint_path is not None

    final_model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), final_model_path)

    resolved_pmc_sampling = resolve_pmc_sampling_config(
        pmc_sampling or PMCSamplingConfig(),
        seq_len=config.seq_len,
    )
    pmc_samples_path = output_dir / "pmc_samples.npz"
    generate_and_save_pmc_artifacts(
        model=model,
        dataset=dataset,
        sampling=resolved_pmc_sampling,
        output_dir=output_dir,
    )

    figure_path = output_dir / "transient.png"
    plot_transient(
        csv_path=csv_path,
        save_path=figure_path,
        n_chains=config.n_chains,
    )

    if run is not None:
        if not figure_path.exists():
            raise FileNotFoundError(f"Transient figure was not created: {figure_path}")
        run.log({"transient_plot": _wandb_image(figure_path)})
        run.finish()

    final_row = history_rows[-1]
    return TrainingRunResult(
        run_name=resolved_run_name,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        csv_path=csv_path,
        figure_path=figure_path,
        final_model_path=final_model_path,
        pmc_samples_path=pmc_samples_path,
        latest_checkpoint_path=latest_checkpoint_path,
        final_train_loss=float(final_row["train_loss"]),
        final_id_kl=float(final_row["id_kl"]),
        final_ood_kl=float(final_row["ood_kl"]),
    )


def main(args: TrainConfig) -> None:
    args.validate()
    config = resolve_config(args)
    run_training_job(
        config=config,
        checkpoint_root=args.checkpoint_root,
        output_root=args.output_root,
        run_name=args.run_name,
        run_name_suffix=args.run_name_suffix,
        pmc_sampling=PMCSamplingConfig(
            num_samples=args.pmc_num_samples,
            prompt_len=args.pmc_prompt_len,
            generation_length=args.pmc_generation_length,
        ),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
