"""Bayes-optimal baselines for Markov-chain pretraining.

Four predictives for symmetrised-KL comparisons against the transformer:
  - uni_inf_predictive: Dirichlet-multinomial over stationary distribution (order-0).
  - uni_ret_predictive: discrete prior over training stationaries {pi_m} (order-0).
  - bi_inf_predictive: row-wise Dirichlet over transition rows (order-1).
  - bi_ret_predictive: discrete prior over training transition matrices {T_m} (order-1).

Closed-form posterior samplers (bi-* only; used downstream for distribution metrics):
  - sample_bi_inf_posterior: Dirichlet(alpha + counts) per row -> (n_samples, k, k).
  - sample_bi_ret_posterior: weighted discrete sample from {T_m} -> (n_samples, k, k).

Posterior for uni-* deliberately not implemented here — the right order-0 framing is
under discussion.

Convention: post-observation. preds[:, t, :] predicts the token at position t+1
given tokens[:, 0..t]. This matches the transformer's causal-LM output indexing.
Position t = S-1 still produces a value but corresponds to a non-existent "next"
token; downstream code slices [:-1] for loss/KL alignment.

All functions expect tokens in the state range [0, k) — BOS must be stripped by
the caller.
"""

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


def uni_inf_predictive(
    tokens: Int[torch.Tensor, "batch seq"],
    alpha: Float[torch.Tensor, " k"],
) -> Float[torch.Tensor, "batch seq k"]:
    """Posterior predictive under Dirichlet(alpha) prior over the stationary.

    preds[b, t, :] = (alpha + counts(tokens[b, 0..t])) / (sum(alpha) + t + 1)
    """
    k = alpha.shape[0]
    B, S = tokens.shape

    one_hot = F.one_hot(tokens.long(), k).to(alpha.dtype)
    cum_counts = one_hot.cumsum(dim=1)

    # number of observations after seeing tokens[:, 0..t] is t + 1
    n_obs = torch.arange(1, S + 1, device=tokens.device, dtype=alpha.dtype).view(
        1, -1, 1
    )
    numerator = einops.rearrange(alpha, "k -> 1 1 k") + cum_counts
    denominator = alpha.sum() + n_obs
    return numerator / denominator


def uni_ret_predictive(
    tokens: Int[torch.Tensor, "batch seq"],
    stationaries: Float[torch.Tensor, "M k"],
) -> Float[torch.Tensor, "batch seq k"]:
    """Posterior predictive under a uniform discrete prior over {pi_m}.

    log w_m(t) = sum_{s=0..t} log pi_m[tokens[b, s]]; weights = softmax(log_w);
    preds[b, t, :] = sum_m w_m(t) * pi_m.
    """
    M, k = stationaries.shape
    B, S = tokens.shape

    log_pi = torch.log(stationaries)

    # log pi_m[tokens[b, s]] for all (m, b, s); rearrange to (B, S, M).
    per_step = log_pi[:, tokens.long()]
    per_step = einops.rearrange(per_step, "M B S -> B S M")

    cum_log_lik = per_step.cumsum(dim=1)
    weights = torch.softmax(cum_log_lik, dim=-1)

    return einops.einsum(weights, stationaries, "b s m, m k -> b s k")


def bi_inf_predictive(
    tokens: Int[torch.Tensor, "batch seq"],
    alpha: Float[torch.Tensor, "k k"],
) -> Float[torch.Tensor, "batch seq k"]:
    """Posterior predictive under row-wise Dirichlet(alpha[i]) priors on T.

    At position t, the current state is tokens[:, t] (observed); predict tokens[:, t+1]
    using transitions in tokens[:, 0..t] (t transitions).

    preds[b, t, j] = (alpha[i, j] + n[i, j]) / (sum_l alpha[i, l] + sum_l n[i, l])
    where i = tokens[b, t] and n = transition counts over tokens[b, 0..t].
    """
    k = alpha.shape[0]
    B, S = tokens.shape
    assert alpha.shape == (k, k), f"alpha must be (k, k), got {tuple(alpha.shape)}"

    prev = tokens[:, :-1].long()
    curr = tokens[:, 1:].long()
    prev_oh = F.one_hot(prev, k).to(alpha.dtype)
    curr_oh = F.one_hot(curr, k).to(alpha.dtype)

    # per-step outer product: pair_oh[b, s, i, j] = 1[(tokens[b,s], tokens[b,s+1]) = (i, j)]
    pair_oh = einops.einsum(prev_oh, curr_oh, "b s i, b s j -> b s i j")

    cum_pair = pair_oh.cumsum(dim=1)  # (B, S-1, k, k)
    # position t=0 has zero transitions; prepend a zero slab
    zero_slab = torch.zeros_like(cum_pair[:, :1])
    all_counts = torch.cat([zero_slab, cum_pair], dim=1)  # (B, S, k, k)

    # Dirichlet posterior concentration per (b, t, i, j)
    posterior = einops.rearrange(alpha, "i j -> 1 1 i j") + all_counts

    # select row i = tokens[b, t]
    tokens_oh = F.one_hot(tokens.long(), k).to(alpha.dtype)
    rows = einops.einsum(tokens_oh, posterior, "b s i, b s i j -> b s j")

    return rows / rows.sum(dim=-1, keepdim=True)


def bi_ret_predictive(
    tokens: Int[torch.Tensor, "batch seq"],
    training_matrices: Float[torch.Tensor, "M k k"],
) -> Float[torch.Tensor, "batch seq k"]:
    """Posterior predictive under a uniform discrete prior over {T_m}.

    log w_m(t) = sum_{s=1..t} log T_m[tokens[b, s-1], tokens[b, s]]; weights = softmax(log_w);
    preds[b, t, :] = sum_m w_m(t) * T_m[tokens[b, t], :].

    At t=0 no transitions are observed, so weights are uniform.
    """
    M, k, _ = training_matrices.shape
    B, S = tokens.shape

    log_T = torch.log(training_matrices)  # (M, k, k)

    prev = tokens[:, :-1].long()
    curr = tokens[:, 1:].long()
    # log_T[m, prev[b,s], curr[b,s]] -> (M, B, S-1) via advanced indexing
    per_step = log_T[:, prev, curr]
    per_step = einops.rearrange(per_step, "M B S -> B S M")  # S here is S-1

    cum_log_lik = per_step.cumsum(dim=1)  # (B, S-1, M)
    # position t=0 has zero transitions -> log_w = 0; prepend zero slab
    zero_slab = torch.zeros_like(cum_log_lik[:, :1])
    all_log_w = torch.cat([zero_slab, cum_log_lik], dim=1)  # (B, S, M)

    weights = torch.softmax(all_log_w, dim=-1)  # (B, S, M)

    # select row i = tokens[b, t] from each T_m -> (B, S, M, k)
    rows_per_chain = training_matrices[:, tokens.long()]  # (M, B, S, k)
    rows_per_chain = einops.rearrange(rows_per_chain, "M B S k -> B S M k")

    return einops.einsum(weights, rows_per_chain, "b s m, b s m k -> b s k")


def bi_inf_posterior_alpha(
    tokens: Int[torch.Tensor, " seq"],
    alpha: Float[torch.Tensor, "k k"],
) -> Float[torch.Tensor, "k k"]:
    """Row-wise Dirichlet posterior concentration: alpha[i, j] + transition count n[i, j]."""
    k = alpha.shape[0]
    assert tokens.dim() == 1, f"tokens must be 1-D, got shape {tuple(tokens.shape)}"

    if tokens.numel() < 2:
        return alpha.clone()

    prev = tokens[:-1].long()
    curr = tokens[1:].long()
    counts = torch.zeros(k, k, dtype=alpha.dtype, device=alpha.device)
    counts.index_put_(
        (prev, curr),
        torch.ones_like(prev, dtype=alpha.dtype),
        accumulate=True,
    )
    return alpha + counts


def bi_ret_posterior_weights(
    tokens: Int[torch.Tensor, " seq"],
    training_matrices: Float[torch.Tensor, "M k k"],
) -> Float[torch.Tensor, " M"]:
    """Softmax posterior weights over {T_m} given observed transitions."""
    M = training_matrices.shape[0]
    assert tokens.dim() == 1, f"tokens must be 1-D, got shape {tuple(tokens.shape)}"

    log_T = torch.log(training_matrices)
    if tokens.numel() < 2:
        return torch.full(
            (M,),
            1.0 / M,
            dtype=training_matrices.dtype,
            device=training_matrices.device,
        )

    prev = tokens[:-1].long()
    curr = tokens[1:].long()
    log_weights = log_T[:, prev, curr].sum(dim=-1)  # (M,)
    return torch.softmax(log_weights, dim=0)


def sample_bi_inf_posterior(
    tokens: Int[torch.Tensor, " seq"],
    alpha: Float[torch.Tensor, "k k"],
    n_samples: int,
) -> Float[torch.Tensor, "n_samples k k"]:
    """Sample T ~ row-wise Dirichlet posterior -> (n_samples, k, k)."""
    posterior = bi_inf_posterior_alpha(tokens, alpha)
    return torch.distributions.Dirichlet(posterior).sample((n_samples,))


def sample_bi_ret_posterior(
    tokens: Int[torch.Tensor, " seq"],
    training_matrices: Float[torch.Tensor, "M k k"],
    n_samples: int,
) -> Float[torch.Tensor, "n_samples k k"]:
    """Sample T from the discrete posterior over {T_m} -> (n_samples, k, k)."""
    weights = bi_ret_posterior_weights(tokens, training_matrices)
    indices = torch.multinomial(weights, n_samples, replacement=True)
    return training_matrices[indices]
