"""Run orchestration for the LR sweep-analysis pipeline."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from linear_regression.likelihoods import linear_regression
from linear_regression.priors import DiscretePrior
from pfn_transformerlens import DeterministicGenerator, sample_batch
from pfn_transformerlens.model.PFN import SupervisedPFN

from linear_regression.analysis.config import SweepConfig
from linear_regression.analysis.data import (
    PromptData,
    create_generators,
    find_latest_checkpoint,
    generate_prompt_data,
    load_prior_dataset,
    load_run_info,
    load_shared_dataset,
    save_prior_dataset,
    save_shared_dataset,
)
from linear_regression.analysis.metrics import (
    compute_all_predictive_metrics,
    compute_distribution_metrics_single,
    merge_results,
)
from linear_regression.predictive_monte_carlo import prepare_model_for_long_rollout


class RunEvalInfo(TypedDict):
    """Checkpoint fields needed to construct fixed evaluation inputs."""

    task_size: int
    num_tasks: int


@dataclass
class SharedEvalContext:
    """Prior-independent evaluation data reused across runs."""

    data: dict[str, np.ndarray] | None
    generalising_batch: tuple[torch.Tensor, torch.Tensor] | None = None
    random_batch: tuple[torch.Tensor, torch.Tensor] | None = None


def load_or_create_shared_eval_context(
    config: SweepConfig,
    run_dirs: list[Path],
) -> SharedEvalContext:
    if config.eval_dataset_dir is None:
        return SharedEvalContext(data=None)

    dataset_dir = Path(config.eval_dataset_dir)
    shared_path = dataset_dir / "shared.npz"

    if shared_path.exists():
        shared_data = load_shared_dataset(shared_path)
    else:
        first_run_dir = next(
            (
                run_dir
                for run_dir in run_dirs
                if find_latest_checkpoint(run_dir) is not None
            ),
            None,
        )
        assert first_run_dir is not None, "No checkpoints found"
        first_ckpt = find_latest_checkpoint(first_run_dir)
        assert first_ckpt is not None
        first_info = load_run_info(first_ckpt)
        first_task_size = int(first_info["task_size"])
        del first_info
        torch.cuda.empty_cache()

        seq_len = config.seq_len
        max_prompt_length = max(max(config.prompt_lengths), 1)
        max_n_prompts = max(config.n_prompts)

        prompt_xs = np.random.randn(
            max_n_prompts, max_prompt_length, first_task_size
        ).astype(np.float32)
        noise = config.noise_std * np.random.randn(max_n_prompts, max_prompt_length)
        ws_gaussian = np.random.randn(max_n_prompts, first_task_size)
        prompt_ys_generalising = (
            np.einsum("npd,nd->np", prompt_xs, ws_gaussian) + noise
        ).astype(np.float32)
        prompt_ys_random = np.random.randn(max_n_prompts, max_prompt_length).astype(
            np.float32
        )

        gaussian_prior = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros(first_task_size), torch.ones(first_task_size)
            ),
            reinterpreted_batch_ndims=1,
        )
        generalising_gen = DeterministicGenerator(
            prior=gaussian_prior,
            function=linear_regression,
            input_dim=first_task_size,
            noise_std=config.noise_std,
            device="cpu",
        )
        generalising_xs, generalising_ys = sample_batch(
            generalising_gen, batch_size=config.eval_batch_size, seq_len=seq_len
        )
        assert generalising_xs is not None and generalising_ys is not None
        random_xs = torch.randn(config.eval_batch_size, seq_len, first_task_size)
        random_ys = torch.randn(config.eval_batch_size, seq_len)

        save_shared_dataset(
            shared_path,
            prompt_xs,
            prompt_ys_generalising,
            prompt_ys_random,
            generalising_xs.numpy(),
            generalising_ys.numpy(),
            random_xs.numpy(),
            random_ys.numpy(),
            first_task_size,
            config.noise_std,
            config.eval_batch_size,
            seq_len,
            max_prompt_length,
            max_n_prompts,
        )
        shared_data = load_shared_dataset(shared_path)

    if not np.isclose(float(shared_data["noise_std"]), config.noise_std):
        raise ValueError("Cached shared dataset noise_std does not match the config.")
    if int(shared_data["eval_batch_size"]) != config.eval_batch_size:
        raise ValueError(
            "Cached shared dataset eval_batch_size does not match the config."
        )
    if int(shared_data["seq_len"]) != config.seq_len:
        raise ValueError("Cached shared dataset seq_len does not match the config.")
    required_prompt_length = max(
        max(config.prompt_lengths),
        config.eval_prompt_length if config.separate_eval_prompts else 0,
    )
    required_prompt_count = max(
        max(config.n_prompts),
        config.eval_n_prompts if config.separate_eval_prompts else 0,
    )
    if int(shared_data["max_prompt_length"]) < required_prompt_length:
        raise ValueError("Cached shared dataset does not contain long enough prompts.")
    if int(shared_data["max_n_prompts"]) < required_prompt_count:
        raise ValueError("Cached shared dataset does not contain enough prompts.")

    generalising_batch = (
        torch.from_numpy(shared_data["generalising_xs"]),
        torch.from_numpy(shared_data["generalising_ys"]),
    )
    random_batch = (
        torch.from_numpy(shared_data["random_xs"]),
        torch.from_numpy(shared_data["random_ys"]),
    )
    return SharedEvalContext(
        data=shared_data,
        generalising_batch=generalising_batch,
        random_batch=random_batch,
    )


def build_run_eval_inputs(
    config: SweepConfig,
    run_id: str,
    run_info: RunEvalInfo,
    prior: DiscretePrior,
    shared_context: SharedEvalContext,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]] | None, PromptData | None]:
    shared_data = shared_context.data
    eval_data_for_run: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
    prompt_data: PromptData | None = None

    if shared_data is not None:
        assert int(shared_data["task_size"]) == int(run_info["task_size"])
        if config.eval_position is not None:
            assert int(shared_data["seq_len"]) >= config.eval_position + 1

        assert config.eval_dataset_dir is not None
        dataset_dir = Path(config.eval_dataset_dir)
        run_path = dataset_dir / f"{run_id}.npz"

        if run_path.exists():
            run_dataset = load_prior_dataset(run_path)
            if int(run_dataset["num_tasks"]) != int(run_info["num_tasks"]):
                raise ValueError(
                    f"Cached run dataset task count does not match run {run_id}."
                )
            if str(run_dataset["run_id"]) != run_id:
                raise ValueError(
                    f"Cached run dataset belongs to {run_dataset['run_id']!r}."
                )
            memorising_xs = torch.from_numpy(run_dataset["memorising_xs"])
            memorising_ys = torch.from_numpy(run_dataset["memorising_ys"])
            prompt_ys_memorising = run_dataset["prompt_ys_memorising"]
        else:
            memorising_gen, _ = create_generators(
                prior, int(run_info["task_size"]), config.noise_std
            )
            gen_seq_len = int(shared_data["seq_len"])
            gen_batch_size = int(shared_data["eval_batch_size"])
            memorising_xs, memorising_ys = sample_batch(
                memorising_gen,
                batch_size=gen_batch_size,
                seq_len=gen_seq_len,
            )
            assert memorising_xs is not None and memorising_ys is not None

            n_prompts = int(shared_data["max_n_prompts"])
            prompt_length = int(shared_data["max_prompt_length"])
            noise = config.noise_std * np.random.randn(n_prompts, prompt_length)
            tasks_np = prior.tasks.cpu().numpy()
            task_indices = np.arange(n_prompts) % prior.num_tasks
            ws_discrete = tasks_np[task_indices]
            prompt_ys_memorising = (
                np.einsum("npd,nd->np", shared_data["prompt_xs"], ws_discrete) + noise
            ).astype(np.float32)

            save_prior_dataset(
                run_path,
                prompt_ys_memorising,
                memorising_xs.numpy(),
                memorising_ys.numpy(),
                int(run_info["num_tasks"]),
                run_id,
            )

        if config.separate_eval_prompts:
            n_eval = config.eval_n_prompts
            prompt_length = config.eval_prompt_length
            prompt_xs = torch.from_numpy(
                shared_data["prompt_xs"][:n_eval, :prompt_length]
            )
            eval_data_for_run = {
                "data_memorising": (
                    prompt_xs,
                    torch.from_numpy(prompt_ys_memorising[:n_eval, :prompt_length]),
                ),
                "data_generalising": (
                    prompt_xs,
                    torch.from_numpy(
                        shared_data["prompt_ys_generalising"][:n_eval, :prompt_length]
                    ),
                ),
            }
            if config.include_random_eval:
                eval_data_for_run["random"] = (
                    prompt_xs,
                    torch.from_numpy(
                        shared_data["prompt_ys_random"][:n_eval, :prompt_length]
                    ),
                )
        else:
            assert shared_context.generalising_batch is not None
            eval_data_for_run = {
                "data_memorising": (memorising_xs, memorising_ys),
                "data_generalising": shared_context.generalising_batch,
            }
            if config.include_random_eval:
                assert shared_context.random_batch is not None
                eval_data_for_run["random"] = shared_context.random_batch

        if config.compute_distribution_metrics:
            prompt_data = PromptData(
                xs=shared_data["prompt_xs"],
                ys_gaussian=shared_data["prompt_ys_generalising"],
                ys_discrete=prompt_ys_memorising,
                ys_random=shared_data["prompt_ys_random"],
            )
        return eval_data_for_run, prompt_data

    if config.compute_distribution_metrics or config.separate_eval_prompts:
        max_prompt_length = (
            max(config.prompt_lengths)
            if config.prompt_lengths
            else config.eval_prompt_length
        )
        max_n_prompts = (
            max(config.n_prompts) if config.n_prompts else config.eval_n_prompts
        )
        if config.separate_eval_prompts:
            max_prompt_length = max(max_prompt_length, config.eval_prompt_length)
            max_n_prompts = max(max_n_prompts, config.eval_n_prompts)
        prompt_data = generate_prompt_data(
            prior=prior,
            max_prompt_length=max_prompt_length,
            n_prompts=max_n_prompts,
            noise_std=config.noise_std,
        )

    if config.separate_eval_prompts and prompt_data is not None:
        n_eval = config.eval_n_prompts
        prompt_length = config.eval_prompt_length
        prompt_xs = torch.from_numpy(prompt_data.xs[:n_eval, :prompt_length])
        eval_data_for_run = {
            "data_memorising": (
                prompt_xs,
                torch.from_numpy(prompt_data.ys_discrete[:n_eval, :prompt_length]),
            ),
            "data_generalising": (
                prompt_xs,
                torch.from_numpy(prompt_data.ys_gaussian[:n_eval, :prompt_length]),
            ),
        }
        if config.include_random_eval:
            eval_data_for_run["random"] = (
                prompt_xs,
                torch.from_numpy(prompt_data.ys_random[:n_eval, :prompt_length]),
            )

    return eval_data_for_run, prompt_data


def run_analysis(
    config: SweepConfig, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run analysis on all checkpoints in the sweep."""
    config.validate()
    checkpoint_root = Path(config.checkpoint_root)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        run_dir for run_dir in checkpoint_root.iterdir() if run_dir.is_dir()
    )
    predictive_results: list[dict] = []
    distribution_results: list[dict] = []
    per_prompt_results: list[dict] = []

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    shared_context = load_or_create_shared_eval_context(config, run_dirs)

    for run_dir in tqdm(run_dirs, desc="Processing runs"):
        checkpoint_path = find_latest_checkpoint(run_dir)
        if checkpoint_path is None:
            continue

        run_info = load_run_info(checkpoint_path)
        model = run_info["model"]
        assert isinstance(model, SupervisedPFN)
        prior = DiscretePrior(
            task_size=run_info["task_size"], tasks=run_info["tasks"], device="cpu"
        )

        run_id = run_dir.name
        num_tasks = int(run_info["num_tasks"])
        checkpoint_step = int(checkpoint_path.stem.split("_")[-1])

        eval_data_for_run, prompt_data = build_run_eval_inputs(
            config, run_id, run_info, prior, shared_context
        )

        pred_metrics = compute_all_predictive_metrics(
            model, prior, config, eval_data=eval_data_for_run
        )
        predictive_results.append(
            {
                "run_id": run_id,
                "num_tasks": num_tasks,
                "checkpoint_step": checkpoint_step,
                **pred_metrics,
            }
        )

        if config.compute_distribution_metrics:
            max_prompt_length = max(config.prompt_lengths)
            effective_steps = prepare_model_for_long_rollout(
                model,
                rollout_length=config.predictive_steps,
                prompt_length=max_prompt_length,
            )

            if 0 in config.prompt_lengths:
                for n_samples_prior in config.n_samples_prior:
                    dist_metrics, dist_samples, dist_per_prompt = (
                        compute_distribution_metrics_single(
                            model,
                            prior,
                            noise_std=config.noise_std,
                            noise_variance=config.noise_variance,
                            n_projections=config.n_projections,
                            prompt_source="N/A",
                            prompt_length=0,
                            predictive_steps=effective_steps,
                            n_samples=0,
                            n_samples_prior=n_samples_prior,
                            n_prompts=0,
                            model_prepared=True,
                        )
                    )
                    np.savez(samples_dir / f"T{num_tasks}_prior.npz", **dist_samples)
                    row_base = {
                        "run_id": run_id,
                        "num_tasks": num_tasks,
                        "checkpoint_step": checkpoint_step,
                        "prompt_source": "N/A",
                        "prompt_length": 0,
                        "n_samples": 0,
                        "n_samples_prior": n_samples_prior,
                        "n_prompts": 0,
                    }
                    distribution_results.append({**row_base, **dist_metrics})
                    for prompt_idx, prompt_metrics in enumerate(dist_per_prompt):
                        per_prompt_results.append(
                            {**row_base, "prompt_idx": prompt_idx, **prompt_metrics}
                        )

            posterior_lengths = [
                prompt_length
                for prompt_length in config.prompt_lengths
                if prompt_length > 0
            ]
            for source, length, n_samples, n_prompts in itertools.product(
                config.prompt_sources,
                posterior_lengths,
                config.n_samples,
                config.n_prompts,
            ):
                dist_metrics, dist_samples, dist_per_prompt = (
                    compute_distribution_metrics_single(
                        model,
                        prior,
                        noise_std=config.noise_std,
                        noise_variance=config.noise_variance,
                        n_projections=config.n_projections,
                        prompt_source=source,
                        prompt_length=length,
                        predictive_steps=effective_steps,
                        n_samples=n_samples,
                        n_samples_prior=0,
                        n_prompts=n_prompts,
                        model_prepared=True,
                        prompt_data=prompt_data,
                    )
                )
                np.savez(
                    samples_dir / f"T{num_tasks}_{source}_L{length}.npz", **dist_samples
                )
                source_for_csv = {
                    "discrete": "memorising",
                    "gaussian": "generalising",
                }.get(source, source)
                row_base = {
                    "run_id": run_id,
                    "num_tasks": num_tasks,
                    "checkpoint_step": checkpoint_step,
                    "prompt_source": source_for_csv,
                    "prompt_length": length,
                    "n_samples": n_samples,
                    "n_samples_prior": 0,
                    "n_prompts": n_prompts,
                }
                distribution_results.append({**row_base, **dist_metrics})
                for prompt_idx, prompt_metrics in enumerate(dist_per_prompt):
                    per_prompt_results.append(
                        {**row_base, "prompt_idx": prompt_idx, **prompt_metrics}
                    )

        del run_info, prior
        torch.cuda.empty_cache()

    if not predictive_results:
        raise RuntimeError("No results collected - check checkpoint directory")

    per_prompt_df = (
        pd.DataFrame(per_prompt_results) if per_prompt_results else pd.DataFrame()
    )
    return merge_results(predictive_results, distribution_results), per_prompt_df
