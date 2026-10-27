"""Bayes-optimal baselines for the Dirichlet-Multinomial (BAU) setting.

Predictive baselines:
- generalising_predictive: closed-form posterior predictive under Dirichlet prior
- memorising_predictive: Bayes-optimal for a discrete set of fixed simplexes

Posterior sampling (for distribution metrics):
- sample_generalising_posterior: sample theta from Dirichlet posterior given observed tokens
- sample_memorising_posterior: sample theta from discrete posterior given observed tokens
"""

import torch
from jaxtyping import Float


def generalising_predictive(
    tokens: Float[torch.Tensor, "batch seq"],
    alpha: Float[torch.Tensor, " V"],
) -> Float[torch.Tensor, "batch seq V"]:
    """Bayes-optimal predictive under Dirichlet(alpha) prior.

    P(x_{k+1}=j | x_{1:k}) = (alpha_j + n_j(x_{1:k})) / (sum(alpha) + k)

    Fully vectorized via cumulative one-hot counts.

    Args:
        tokens: Integer token sequences, shape (batch, seq).
        alpha: Dirichlet concentration parameters, shape (V,).

    Returns:
        Predictive probabilities, shape (batch, seq, V).
        Position k gives P(x_{k+1} | x_{1:k}), i.e. prediction AFTER seeing x_k.
    """
    V = alpha.shape[0]
    B, S = tokens.shape

    one_hot = torch.nn.functional.one_hot(tokens.long(), V).float()
    # cumulative counts up to (but not including) position k
    cum_counts = one_hot.cumsum(dim=1) - one_hot  # (B, S, V)

    # positions: k=0 has seen 0 tokens, k=1 has seen 1, etc.
    k = torch.arange(S, device=tokens.device, dtype=torch.float32).view(1, -1, 1)

    numerator = alpha.view(1, 1, V) + cum_counts
    denominator = alpha.sum() + k
    return numerator / denominator


def memorising_predictive(
    tokens: Float[torch.Tensor, "batch seq"],
    thetas: Float[torch.Tensor, "M V"],
) -> Float[torch.Tensor, "batch seq V"]:
    """Bayes-optimal predictive under a discrete uniform prior over M simplexes.

    Streaming log-posterior update:
        log P(theta_m | x_{1:k}) += log theta_m[x_k]
    Then predictive:
        P(x_{k+1} | x_{1:k}) = sum_m P(theta_m | x_{1:k}) * theta_m

    Args:
        tokens: Integer token sequences, shape (batch, seq).
        thetas: Pool of probability simplexes, shape (M, V).

    Returns:
        Predictive probabilities, shape (batch, seq, V).
        Position k gives P(x_{k+1} | x_{1:k}).
    """
    B, S = tokens.shape
    M, V = thetas.shape

    preds = torch.empty(B, S, V, device=tokens.device, dtype=thetas.dtype)
    log_weights = torch.zeros(B, M, device=tokens.device, dtype=thetas.dtype)

    for k in range(S):
        if k > 0:
            # update log-posterior with observation at position k-1
            prev_token = tokens[:, k - 1].long()  # (B,)
            log_lik = torch.log(thetas.T[prev_token])  # (B, M)
            log_weights = log_weights + log_lik

        log_norm = torch.logsumexp(log_weights, dim=-1, keepdim=True)
        weights = torch.exp(log_weights - log_norm)  # (B, M)

        preds[:, k, :] = weights @ thetas  # (B, V)

    return preds


def dirichlet_posterior_alpha(
    tokens: Float[torch.Tensor, " seq"],
    alpha: Float[torch.Tensor, " V"],
) -> Float[torch.Tensor, " V"]:
    """Concentration of the Dirichlet posterior: alpha + token counts."""
    counts = torch.bincount(tokens.long(), minlength=alpha.shape[0]).to(alpha.dtype)
    return alpha + counts


def discrete_posterior_weights(
    tokens: Float[torch.Tensor, " seq"],
    thetas: Float[torch.Tensor, "M V"],
) -> Float[torch.Tensor, " M"]:
    """Posterior weights over the discrete theta pool given observed tokens."""
    log_weights = torch.log(thetas[:, tokens.long()]).sum(dim=1)
    log_norm = torch.logsumexp(log_weights, dim=0)
    return torch.exp(log_weights - log_norm)


def sample_generalising_posterior(
    tokens: Float[torch.Tensor, " seq"],
    alpha: Float[torch.Tensor, " V"],
    n_samples: int,
) -> Float[torch.Tensor, "n_samples V"]:
    """Sample from Dirichlet posterior: Dirichlet(alpha + counts)."""
    posterior_alpha = dirichlet_posterior_alpha(tokens, alpha)
    return torch.distributions.Dirichlet(posterior_alpha).sample((n_samples,))


def sample_memorising_posterior(
    tokens: Float[torch.Tensor, " seq"],
    thetas: Float[torch.Tensor, "M V"],
    n_samples: int,
) -> Float[torch.Tensor, "n_samples V"]:
    """Sample theta vectors from the discrete posterior given observed tokens."""
    weights = discrete_posterior_weights(tokens, thetas)
    indices = torch.multinomial(weights, n_samples, replacement=True)
    return thetas[indices]
