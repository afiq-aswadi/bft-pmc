"""Metric computation for the LR sweep-analysis pipeline."""

from __future__ import annotations

from jaxtyping import Float

import numpy as np
import pandas as pd
import torch

from linear_regression.baselines import (
    generalising_predictor,
    memorising_predictor,
)
from linear_regression.evals import mse
from linear_regression.predictive_monte_carlo import (
    predictive_monte_carlo_beta_chunked,
    prepare_model_for_long_rollout,
)
from linear_regression.priors import DiscretePrior
from metrics import energy_distance_multidim, sliced_wasserstein
from pfn_transformerlens import sample_batch
from pfn_transformerlens.model.PFN import DistributionPrediction, SupervisedPFN

from linear_regression.analysis.config import SweepConfig
from linear_regression.analysis.data import (
    PromptData,
    create_generators,
    get_prompts_for_config,
    sample_prompt,
)


def compute_predictive_metrics(
    model: SupervisedPFN,
    xs: Float[torch.Tensor, "batch seq dim"],
    ys: Float[torch.Tensor, "batch seq"] | Float[torch.Tensor, "batch seq 1"],
    prior: DiscretePrior,
    noise_variance: float,
    config: SweepConfig,
) -> dict[str, float]:
    """Compute model-vs-baseline predictive metrics on a batch."""
    device = next(model.parameters()).device
    xs = xs.to(device)
    ys = ys.to(device)
    if ys.dim() == 2:
        ys = ys.unsqueeze(-1)

    memorising_preds = memorising_predictor(xs, ys, prior, noise_variance)
    generalising_preds = generalising_predictor(xs, ys, noise_variance)

    model.eval()
    with torch.no_grad():
        pred = model.predict_on_prompt(xs, ys.squeeze(-1))
        if isinstance(pred, tuple):
            pred = pred[0]
        assert isinstance(pred, DistributionPrediction)
        model_preds = (pred.probs * pred.y_grid).sum(dim=-1).unsqueeze(-1)

    ys_cpu = ys.cpu()
    model_preds_cpu = model_preds.cpu()
    memorising_preds_cpu = memorising_preds.cpu()
    generalising_preds_cpu = generalising_preds.cpu()

    if config.eval_position is not None:
        position = config.eval_position
        ys_cpu = ys_cpu[:, position : position + 1, :]
        model_preds_cpu = model_preds_cpu[:, position : position + 1, :]
        memorising_preds_cpu = memorising_preds_cpu[:, position : position + 1, :]
        generalising_preds_cpu = generalising_preds_cpu[:, position : position + 1, :]

    task_size = xs.shape[-1]
    results = {
        "model_mse": mse(ys_cpu, model_preds_cpu).item() / task_size,
        "baseline_memorising_mse": mse(ys_cpu, memorising_preds_cpu).item() / task_size,
        "baseline_generalising_mse": mse(ys_cpu, generalising_preds_cpu).item()
        / task_size,
        "delta_vs_baseline_memorising": mse(
            model_preds_cpu, memorising_preds_cpu
        ).item()
        / task_size,
        "delta_vs_baseline_generalising": mse(
            model_preds_cpu, generalising_preds_cpu
        ).item()
        / task_size,
    }

    return results


def compute_dmmse_weights(
    xs: Float[torch.Tensor, "seq dim"],
    ys: Float[torch.Tensor, " seq"],
    prior: DiscretePrior,
    noise_variance: float,
) -> np.ndarray:
    """Closed-form discrete posterior weights over `prior.tasks`. Shape (M,), sums to 1."""
    tasks = prior.tasks
    xs, ys = xs.to(tasks.device), ys.to(tasks.device)
    residuals = ys.unsqueeze(0) - torch.einsum("kd,sd->ks", tasks, xs)
    log_likelihoods = -0.5 * (residuals**2).sum(dim=1) / noise_variance
    log_probs = log_likelihoods - log_likelihoods.logsumexp(dim=0)
    return log_probs.exp().cpu().numpy()


def sample_dmmse_posterior(
    xs: Float[torch.Tensor, "seq dim"],
    ys: Float[torch.Tensor, " seq"],
    prior: DiscretePrior,
    noise_variance: float,
    n_samples: int,
) -> np.ndarray:
    """Sample from the dMMSE posterior (categorical over discrete tasks)."""
    probs = compute_dmmse_weights(xs, ys, prior, noise_variance)
    indices = np.random.choice(len(probs), size=n_samples, p=probs)
    return prior.tasks.cpu().numpy()[indices]


def sample_ridge_posterior(
    xs: Float[torch.Tensor, "seq dim"],
    ys: Float[torch.Tensor, " seq"],
    noise_variance: float,
    n_samples: int,
) -> np.ndarray:
    """Sample from the ridge posterior (Gaussian)."""
    mean, cov = gaussian_posterior_params(xs, ys, noise_variance)
    return np.random.multivariate_normal(mean, cov, size=n_samples)


def gaussian_posterior_params(
    xs: Float[torch.Tensor, "seq dim"],
    ys: Float[torch.Tensor, " seq"],
    noise_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return analytic Gaussian posterior mean and covariance."""
    x_np = xs.cpu().numpy()
    y_np = ys.cpu().numpy()
    task_size = x_np.shape[1]
    precision = x_np.T @ x_np / noise_variance + np.eye(task_size)
    cov = np.linalg.inv(precision)
    mean = cov @ x_np.T @ y_np / noise_variance
    return mean, cov


def compute_distribution_metrics_single(
    model: SupervisedPFN,
    prior: DiscretePrior,
    noise_std: float,
    noise_variance: float,
    n_projections: int,
    prompt_source: str,
    prompt_length: int,
    predictive_steps: int,
    n_samples: int,
    n_samples_prior: int,
    n_prompts: int,
    model_prepared: bool = False,
    prompt_data: PromptData | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray], list[dict[str, float]]]:
    """Compute ED and SW between the model's distribution and baselines."""
    if not model_prepared:
        predictive_steps = prepare_model_for_long_rollout(
            model,
            rollout_length=predictive_steps,
            prompt_length=prompt_length,
        )

    task_size = prior.task_size
    x_dist = torch.distributions.Normal(0.0, 1.0)

    if prompt_length == 0:
        pt_samples = predictive_monte_carlo_beta_chunked(
            model=model,
            x_distribution=x_dist,
            forward_recursion_steps=predictive_steps,
            forward_recursion_samples=n_samples_prior,
            chunk_size=100,
            init_x=None,
            init_y=None,
        )
        assert isinstance(pt_samples, np.ndarray)
        discrete_indices = np.random.randint(0, prior.num_tasks, size=n_samples_prior)
        theta_pool = prior.tasks.cpu().numpy()
        dmmse_samples = theta_pool[discrete_indices]
        ridge_samples = np.random.randn(n_samples_prior, task_size)

        metrics: dict[str, float] = {
            "dist/ed_vs_baseline_memorising": energy_distance_multidim(
                pt_samples, dmmse_samples
            ),
            "dist/ed_vs_baseline_generalising": energy_distance_multidim(
                pt_samples, ridge_samples
            ),
            "dist/sw_vs_baseline_memorising": sliced_wasserstein(
                pt_samples, dmmse_samples, n_projections=n_projections
            ),
            "dist/sw_vs_baseline_generalising": sliced_wasserstein(
                pt_samples, ridge_samples, n_projections=n_projections
            ),
        }
        # theta_pool and dmmse_weights together describe the analytic memorising
        # marginal (uniform 1/M for the prior); plotters use these instead of
        # baseline_memorising_samples so the rendered density has no MC error.
        samples: dict[str, np.ndarray] = {
            "pt": pt_samples,
            "baseline_memorising_samples": dmmse_samples,
            "baseline_generalising_samples": ridge_samples,
            "theta_pool": theta_pool.astype(np.float32),
            "dmmse_weights": np.full(
                prior.num_tasks, 1.0 / prior.num_tasks, dtype=np.float64
            ),
            "is_prior": np.array(True),
        }
        return metrics, samples, [metrics]

    model_device = next(model.parameters()).device

    if prompt_data is not None:
        all_xs, all_ys = get_prompts_for_config(
            prompt_data, prompt_source, prompt_length, n_prompts
        )
        all_xs_device = all_xs.to(model_device)
        all_ys_device = all_ys.to(model_device)

        pt_samples = predictive_monte_carlo_beta_chunked(
            model=model,
            x_distribution=x_dist,
            forward_recursion_steps=predictive_steps,
            forward_recursion_samples=n_samples,
            chunk_size=100,
            init_x=all_xs_device,
            init_y=all_ys_device,
        )
        assert isinstance(pt_samples, np.ndarray)

        metrics_list: list[dict[str, float]] = []
        all_memorising = []
        all_memorising_weights = []
        all_generalising = []
        all_generalising_means = []
        all_generalising_covs = []
        for prompt_idx in range(n_prompts):
            pt_samples_prompt = pt_samples[prompt_idx]
            context_xs = all_xs[prompt_idx]
            context_ys = all_ys[prompt_idx]

            memorising_samples = sample_dmmse_posterior(
                context_xs, context_ys, prior, noise_variance, n_samples
            )
            dmmse_weights_i = compute_dmmse_weights(
                context_xs, context_ys, prior, noise_variance
            )
            all_memorising_weights.append(dmmse_weights_i)
            generalising_samples = sample_ridge_posterior(
                context_xs, context_ys, noise_variance, n_samples
            )
            mean, cov = gaussian_posterior_params(
                context_xs, context_ys, noise_variance
            )
            all_generalising_means.append(mean)
            all_generalising_covs.append(cov)

            all_memorising.append(memorising_samples)
            all_generalising.append(generalising_samples)
            prompt_metrics: dict[str, float] = {
                "dist/ed_vs_baseline_memorising": energy_distance_multidim(
                    pt_samples_prompt, memorising_samples
                ),
                "dist/ed_vs_baseline_generalising": energy_distance_multidim(
                    pt_samples_prompt, generalising_samples
                ),
                "dist/sw_vs_baseline_memorising": sliced_wasserstein(
                    pt_samples_prompt, memorising_samples, n_projections=n_projections
                ),
                "dist/sw_vs_baseline_generalising": sliced_wasserstein(
                    pt_samples_prompt, generalising_samples, n_projections=n_projections
                ),
            }
            metrics_list.append(prompt_metrics)

        metrics = {
            key: float(np.mean([metric[key] for metric in metrics_list]))
            for key in metrics_list[0]
        }
        samples: dict[str, np.ndarray] = {
            "pt": pt_samples,
            "baseline_memorising_samples": np.stack(all_memorising, axis=0),
            "baseline_generalising_samples": np.stack(all_generalising, axis=0),
            "baseline_generalising_posterior_means": np.stack(
                all_generalising_means, axis=0
            ),
            "baseline_generalising_posterior_covs": np.stack(
                all_generalising_covs, axis=0
            ),
            # theta_pool + dmmse_weights describe the analytic memorising posterior
            # per prompt; plotters use these for MC-error-free density/CDF rendering.
            # prompt_xs/prompt_ys are saved so plotters can recompute weights if needed.
            "theta_pool": prior.tasks.cpu().numpy().astype(np.float32),
            "dmmse_weights": np.stack(all_memorising_weights, axis=0).astype(
                np.float64
            ),
            "prompt_xs": all_xs.cpu().numpy().astype(np.float32),
            "prompt_ys": all_ys.cpu().numpy().astype(np.float32),
            "is_prior": np.array(False),
        }
        return metrics, samples, metrics_list

    metrics_list = []
    all_pt = []
    all_memorising = []
    all_memorising_weights = []
    all_generalising = []
    all_generalising_means = []
    all_generalising_covs = []
    all_prompt_xs = []
    all_prompt_ys = []
    for _ in range(n_prompts):
        context_xs, context_ys = sample_prompt(
            prior, prompt_source, prompt_length, noise_std
        )
        context_xs_device = context_xs.to(model_device)
        context_ys_device = context_ys.to(model_device)
        pt_samples = predictive_monte_carlo_beta_chunked(
            model=model,
            x_distribution=x_dist,
            forward_recursion_steps=predictive_steps,
            forward_recursion_samples=n_samples,
            chunk_size=100,
            init_x=context_xs_device,
            init_y=context_ys_device,
        )
        assert isinstance(pt_samples, np.ndarray)
        memorising_samples = sample_dmmse_posterior(
            context_xs, context_ys, prior, noise_variance, n_samples
        )
        dmmse_weights_i = compute_dmmse_weights(
            context_xs,
            context_ys,
            prior,
            noise_variance,
        )
        generalising_samples = sample_ridge_posterior(
            context_xs, context_ys, noise_variance, n_samples
        )
        mean, cov = gaussian_posterior_params(context_xs, context_ys, noise_variance)
        all_generalising_means.append(mean)
        all_generalising_covs.append(cov)

        all_pt.append(pt_samples)
        all_memorising.append(memorising_samples)
        all_memorising_weights.append(dmmse_weights_i)
        all_generalising.append(generalising_samples)
        all_prompt_xs.append(context_xs.cpu().numpy())
        all_prompt_ys.append(context_ys.cpu().numpy())
        prompt_metrics: dict[str, float] = {
            "dist/ed_vs_baseline_memorising": energy_distance_multidim(
                pt_samples, memorising_samples
            ),
            "dist/ed_vs_baseline_generalising": energy_distance_multidim(
                pt_samples, generalising_samples
            ),
            "dist/sw_vs_baseline_memorising": sliced_wasserstein(
                pt_samples, memorising_samples, n_projections=n_projections
            ),
            "dist/sw_vs_baseline_generalising": sliced_wasserstein(
                pt_samples, generalising_samples, n_projections=n_projections
            ),
        }
        metrics_list.append(prompt_metrics)

    metrics = {
        key: float(np.mean([metric[key] for metric in metrics_list]))
        for key in metrics_list[0]
    }
    samples = {
        "pt": np.stack(all_pt, axis=0),
        "baseline_memorising_samples": np.stack(all_memorising, axis=0),
        "baseline_generalising_samples": np.stack(all_generalising, axis=0),
        "baseline_generalising_posterior_means": np.stack(
            all_generalising_means, axis=0
        ),
        "baseline_generalising_posterior_covs": np.stack(all_generalising_covs, axis=0),
        "theta_pool": prior.tasks.cpu().numpy().astype(np.float32),
        "dmmse_weights": np.stack(all_memorising_weights, axis=0).astype(np.float64),
        "prompt_xs": np.stack(all_prompt_xs, axis=0).astype(np.float32),
        "prompt_ys": np.stack(all_prompt_ys, axis=0).astype(np.float32),
        "is_prior": np.array(False),
    }
    return metrics, samples, metrics_list


def compute_all_predictive_metrics(
    model: SupervisedPFN,
    prior: DiscretePrior,
    config: SweepConfig,
    eval_data: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, float]:
    """Compute predictive metrics on memorising/generalising/random batches."""
    assert isinstance(model, SupervisedPFN)

    if eval_data is not None:
        memorising_xs, memorising_ys = eval_data["data_memorising"]
        generalising_xs, generalising_ys = eval_data["data_generalising"]
    else:
        task_size = prior.task_size
        memorising_gen, generalising_gen = create_generators(
            prior, task_size, config.noise_std
        )
        effective_seq_len = (
            (config.eval_position + 1)
            if config.eval_position is not None
            else config.seq_len
        )
        torch.manual_seed(config.seed)
        memorising_xs, memorising_ys = sample_batch(
            memorising_gen, batch_size=config.eval_batch_size, seq_len=effective_seq_len
        )
        assert memorising_xs is not None
        assert memorising_ys is not None
        generalising_xs, generalising_ys = sample_batch(
            generalising_gen,
            batch_size=config.eval_batch_size,
            seq_len=effective_seq_len,
        )
        assert generalising_xs is not None
        assert generalising_ys is not None

    memorising_metrics = compute_predictive_metrics(
        model,
        memorising_xs,
        memorising_ys,
        prior,
        config.noise_variance,
        config,
    )
    generalising_metrics = compute_predictive_metrics(
        model,
        generalising_xs,
        generalising_ys,
        prior,
        config.noise_variance,
        config,
    )

    results = {}
    for key, value in memorising_metrics.items():
        results[f"data_memorising/{key}"] = value
    for key, value in generalising_metrics.items():
        results[f"data_generalising/{key}"] = value

    if config.include_random_eval:
        if eval_data is not None and "random" in eval_data:
            random_xs, random_ys = eval_data["random"]
        else:
            random_xs = torch.randn_like(memorising_xs)
            random_ys = torch.randn_like(memorising_ys)
        random_metrics = compute_predictive_metrics(
            model, random_xs, random_ys, prior, config.noise_variance, config
        )
        for key, value in random_metrics.items():
            results[f"data_random/{key}"] = value

    return results


def merge_results(
    predictive_results: list[dict],
    distribution_results: list[dict],
) -> pd.DataFrame:
    """Merge predictive and distribution results into a single DataFrame."""
    pred_df = pd.DataFrame(predictive_results)

    if not distribution_results:
        return pred_df.sort_values("num_tasks")

    dist_df = pd.DataFrame(distribution_results)
    merged = dist_df.merge(
        pred_df,
        on=["run_id", "num_tasks", "checkpoint_step"],
        how="left",
    )

    return merged.sort_values(
        [
            "num_tasks",
            "prompt_source",
            "prompt_length",
            "n_samples",
            "n_samples_prior",
            "n_prompts",
        ]
    ).reset_index(drop=True)
