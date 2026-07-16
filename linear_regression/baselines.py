"""Bayes-optimal baselines for in-context linear regression."""

from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Float

from linear_regression.priors import DiscretePrior


def memorising_predictor(
    xs: Float[torch.Tensor, "batch seq d_x"],
    ys: Float[torch.Tensor, "batch seq 1"],
    prior: DiscretePrior,
    noise_variance: float,
    chunk_size: int | None = None,
    batch_chunk_size: int | None = None,
    compute_device: str | torch.device | None = None,
) -> Float[torch.Tensor, "batch seq 1"]:
    """Predict with the exact posterior over a finite task pool.

    The sequence dimension is streamed so memory scales with batch size and
    task diversity rather than batch size, sequence length, and task diversity.
    """
    assert xs.ndim == 3, xs.shape
    assert ys.shape == (*xs.shape[:2], 1), (xs.shape, ys.shape)
    assert noise_variance > 0, noise_variance

    batch_size, seq_len, task_size = xs.shape
    output_device = xs.device
    work_device = (
        torch.device(compute_device) if compute_device is not None else output_device
    )
    assert prior.task_size == task_size, (
        f"dimension mismatch: data {task_size} != prior {prior.task_size}"
    )

    task_pool = prior.tasks.to(work_device)
    num_tasks = prior.num_tasks
    task_chunk_size = num_tasks if chunk_size is None or chunk_size <= 0 else chunk_size

    if batch_chunk_size is None:
        target_elements = 2_000_000
        batch_chunk_size = (
            max(1, target_elements // num_tasks)
            if batch_size * num_tasks > target_elements
            else batch_size
        )
    batch_chunk_size = min(batch_chunk_size, batch_size)

    def run_batch_chunk(x_chunk: torch.Tensor, y_chunk: torch.Tensor) -> torch.Tensor:
        chunk_batch_size = x_chunk.shape[0]
        log_weights = torch.zeros(
            chunk_batch_size,
            num_tasks,
            device=work_device,
            dtype=x_chunk.dtype,
        )
        predictions = torch.empty(
            chunk_batch_size,
            seq_len,
            1,
            device=work_device,
            dtype=x_chunk.dtype,
        )
        targets = y_chunk.squeeze(-1)

        for position in range(seq_len):
            if position > 0:
                previous_x = x_chunk[:, position - 1]
                previous_y = targets[:, position - 1]
                likelihood_scale = -0.5 / noise_variance

                for start in range(0, num_tasks, task_chunk_size):
                    end = start + task_chunk_size
                    task_chunk = task_pool[start:end]
                    residual = previous_y.unsqueeze(1) - previous_x @ task_chunk.T
                    log_weights[:, start:end] += likelihood_scale * residual.square()

            log_normalizer = torch.logsumexp(log_weights, dim=-1, keepdim=True)
            posterior_mean = torch.zeros(
                chunk_batch_size,
                task_size,
                device=work_device,
                dtype=x_chunk.dtype,
            )
            for start in range(0, num_tasks, task_chunk_size):
                end = start + task_chunk_size
                weights = torch.exp(log_weights[:, start:end] - log_normalizer)
                posterior_mean += weights @ task_pool[start:end]

            predictions[:, position] = (posterior_mean * x_chunk[:, position]).sum(
                dim=-1, keepdim=True
            )

        return predictions

    xs_work = xs.to(work_device)
    ys_work = ys.to(work_device)
    predictions = [
        run_batch_chunk(
            xs_work[start : start + batch_chunk_size],
            ys_work[start : start + batch_chunk_size],
        )
        for start in range(0, batch_size, batch_chunk_size)
    ]
    result = torch.cat(predictions, dim=0)
    return result.to(output_device) if result.device != output_device else result


def generalising_predictor(
    xs: Float[torch.Tensor, "batch seq d_x"],
    ys: Float[torch.Tensor, "batch seq 1"],
    noise_variance: float,
) -> Float[torch.Tensor, "batch seq 1"]:
    """Predict with the exact posterior under a standard Gaussian task prior."""
    assert xs.ndim == 3, xs.shape
    assert ys.shape == (*xs.shape[:2], 1), (xs.shape, ys.shape)
    assert noise_variance > 0, noise_variance

    batch_size, seq_len, task_size = xs.shape
    xtx = torch.empty(batch_size, seq_len, task_size, task_size, device=xs.device)
    rhs = torch.empty(batch_size, seq_len, task_size, 1, device=xs.device)

    for position in range(seq_len):
        context_x = xs[:, :position]
        context_y = ys[:, :position]
        context_x_t = context_x.transpose(-2, -1)
        xtx[:, position] = context_x_t @ context_x
        rhs[:, position] = context_x_t @ context_y

    regularizer = noise_variance * torch.eye(task_size, device=xs.device)
    posterior_means = torch.linalg.solve(xtx + regularizer, rhs).view(
        batch_size,
        seq_len,
        task_size,
    )
    return (xs * posterior_means).sum(dim=-1, keepdim=True)


def compute_generalising_posterior(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma2: float,
    tau2: float = 1.0,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact Gaussian posterior mean and covariance for one prompt."""
    assert x.ndim in {2, 3}, x.shape
    assert y.ndim in {2, 3}, y.shape
    assert sigma2 > 0, sigma2
    assert tau2 > 0, tau2

    work_device = x.device if device is None else torch.device(device)
    x_matrix = x.to(work_device).squeeze(0) if x.ndim == 3 else x.to(work_device)
    y_vector = y.to(work_device).squeeze(0) if y.ndim == 3 else y.to(work_device)
    assert x_matrix.ndim == 2, x_matrix.shape
    assert y_vector.shape == (x_matrix.shape[0], 1), (x_matrix.shape, y_vector.shape)

    task_size = x_matrix.shape[1]
    precision = (
        x_matrix.T @ x_matrix / sigma2 + torch.eye(task_size, device=work_device) / tau2
    )
    covariance = torch.linalg.inv(precision)
    mean = covariance @ (x_matrix.T @ y_vector / sigma2)
    return (
        mean.squeeze(-1).detach().cpu().numpy(),
        covariance.detach().cpu().numpy(),
    )


def compute_memorising_posterior(
    x: torch.Tensor,
    y: torch.Tensor,
    task_pool: torch.Tensor,
    sigma2: float,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Return exact posterior weights over a finite pool for one prompt."""
    assert x.ndim in {2, 3}, x.shape
    assert y.ndim in {2, 3}, y.shape
    assert task_pool.ndim == 2, task_pool.shape
    assert sigma2 > 0, sigma2

    work_device = x.device if device is None else torch.device(device)
    x_matrix = x.to(work_device).squeeze(0) if x.ndim == 3 else x.to(work_device)
    y_vector = y.to(work_device).squeeze(0) if y.ndim == 3 else y.to(work_device)
    task_pool = task_pool.to(work_device)
    assert x_matrix.shape[1] == task_pool.shape[1], (x_matrix.shape, task_pool.shape)
    assert y_vector.shape == (x_matrix.shape[0], 1), (x_matrix.shape, y_vector.shape)

    residuals = y_vector - x_matrix @ task_pool.T
    posterior_weights = torch.softmax(
        -0.5 / sigma2 * residuals.square().sum(dim=0),
        dim=0,
    )
    values = task_pool.detach().cpu().numpy()
    weights = posterior_weights.detach().cpu().numpy()

    masses = []
    for dimension in range(values.shape[1]):
        unique_values, inverse_indices = np.unique(
            values[:, dimension],
            return_inverse=True,
        )
        probability_mass = np.zeros_like(unique_values, dtype=np.float64)
        np.add.at(probability_mass, inverse_indices, weights)
        masses.append((unique_values, probability_mass))

    return values, weights, masses
