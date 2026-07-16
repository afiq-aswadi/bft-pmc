"""Distance metrics for comparing model and reference sample distributions."""

import numpy as np
import torch
from scipy.stats import wasserstein_distance


def _validate_sample_matrices(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> None:
    """Validate two non-empty finite sample matrices with matching dimensions."""
    if samples_a.ndim != 2 or samples_b.ndim != 2:
        raise ValueError(
            f"samples must be 2D, got {samples_a.shape} and {samples_b.shape}."
        )
    if samples_a.shape[1] != samples_b.shape[1]:
        raise ValueError(
            "sample dimensions must match, got "
            f"{samples_a.shape[1]} and {samples_b.shape[1]}."
        )
    if len(samples_a) == 0 or len(samples_b) == 0:
        raise ValueError("sample matrices must be non-empty.")
    if not np.isfinite(samples_a).all() or not np.isfinite(samples_b).all():
        raise ValueError("samples must be finite.")


def symmetrised_kl(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-10,
) -> float:
    """Symmetrised KL divergence averaged over batch and sequence."""
    if p.shape != q.shape:
        raise ValueError(f"probability shapes must match, got {p.shape} and {q.shape}.")
    if not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError("probabilities must be finite.")
    if (p < 0).any() or (q < 0).any():
        raise ValueError("probabilities must be non-negative.")
    if not torch.allclose(p.sum(dim=-1), torch.ones_like(p.sum(dim=-1)), atol=1e-5):
        raise ValueError("p must sum to one along its final dimension.")
    if not torch.allclose(q.sum(dim=-1), torch.ones_like(q.sum(dim=-1)), atol=1e-5):
        raise ValueError("q must sum to one along its final dimension.")
    log_p = torch.log(p + eps)
    log_q = torch.log(q + eps)
    kl_pq = (p * (log_p - log_q)).sum(dim=-1).mean()
    kl_qp = (q * (log_q - log_p)).sum(dim=-1).mean()
    return (0.5 * (kl_pq + kl_qp)).item()


def sliced_wasserstein(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    n_projections: int = 100,
    seed: int = 42,
) -> float:
    """Compute sliced Wasserstein distance between two sample sets.

    Projects samples onto random 1D directions and averages the 1D Wasserstein
    distances. This is a computationally efficient approximation to the full
    Wasserstein distance in high dimensions.

    Args:
        samples_a: Samples from first distribution, shape [N, D]
        samples_b: Samples from second distribution, shape [M, D]
        n_projections: Number of random 1D projections to average over
        seed: Random seed for reproducibility

    Returns:
        Average 1D Wasserstein distance across random projections
    """
    _validate_sample_matrices(samples_a, samples_b)
    if n_projections < 1:
        raise ValueError("n_projections must be positive.")
    rng = np.random.default_rng(seed)
    d = samples_a.shape[1]

    # Generate random unit vectors for projection
    projections = rng.standard_normal((n_projections, d))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    # Project samples onto each direction
    proj_a = samples_a @ projections.T  # [N, n_projections]
    proj_b = samples_b @ projections.T  # [M, n_projections]

    # Compute 1D Wasserstein for each projection
    distances = [
        wasserstein_distance(proj_a[:, i], proj_b[:, i]) for i in range(n_projections)
    ]

    return float(np.mean(distances))


def energy_distance_multidim(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> float:
    """Compute the unbiased sample estimator of multivariate energy distance.

    Uses the formula:
        ED = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]

    The within-sample terms exclude self-pairs, producing the U-statistic
    estimator. Unlike the population distance, this finite-sample estimator can
    be slightly negative when the distributions are close.

    Note: scipy.stats.energy_distance only works for 1D, so we implement
    the multivariate version directly.

    Args:
        samples_a: Samples from first distribution, shape [N, D]
        samples_b: Samples from second distribution, shape [M, D]

    Returns:
        Unbiased energy-distance estimate, which may be negative.
    """
    _validate_sample_matrices(samples_a, samples_b)
    n, m = len(samples_a), len(samples_b)

    # E[||X-Y||] - cross term
    # Use broadcasting: samples_a[:, None] - samples_b[None, :] gives [N, M, D]
    cross_diffs = samples_a[:, None, :] - samples_b[None, :, :]
    cross_norms = np.linalg.norm(cross_diffs, axis=2)  # [N, M]
    e_xy = np.mean(cross_norms)

    # E[||X-X'||] over distinct sample pairs
    if n > 1:
        a_diffs = samples_a[:, None, :] - samples_a[None, :, :]
        a_norms = np.linalg.norm(a_diffs, axis=2)  # [N, N]
        e_xx = np.sum(a_norms) / (n * (n - 1))
    else:
        e_xx = 0.0

    # E[||Y-Y'||] over distinct sample pairs
    if m > 1:
        b_diffs = samples_b[:, None, :] - samples_b[None, :, :]
        b_norms = np.linalg.norm(b_diffs, axis=2)  # [M, M]
        e_yy = np.sum(b_norms) / (m * (m - 1))
    else:
        e_yy = 0.0

    return float(2 * e_xy - e_xx - e_yy)


def marginal_wasserstein(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> np.ndarray:
    """Compute 1D Wasserstein distance for each dimension separately.

    Useful for interpretability - shows which dimensions have the largest
    distributional differences.

    Args:
        samples_a: Samples from first distribution, shape [N, D]
        samples_b: Samples from second distribution, shape [M, D]

    Returns:
        Array of Wasserstein distances, one per dimension, shape [D]
    """
    _validate_sample_matrices(samples_a, samples_b)
    d = samples_a.shape[1]
    distances = np.array(
        [wasserstein_distance(samples_a[:, i], samples_b[:, i]) for i in range(d)]
    )
    return distances
