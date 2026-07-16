"""Evaluation helpers for Markov-chain experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet

from markov.baselines import (
    bi_inf_predictive,
    bi_ret_predictive,
    uni_inf_predictive,
    uni_ret_predictive,
)
from markov.data import MarkovChainDataset
from markov.model import MarkovTransformer
from metrics import symmetrised_kl


@torch.inference_mode()
def estimate_transition_matrix(
    model: MarkovTransformer,
    dataset: MarkovChainDataset,
    transition_matrix: torch.Tensor,
    context_len: int,
) -> torch.Tensor:
    """Estimate a transition matrix using model next-token probabilities."""
    base_chain = dataset.sample_eval_chains(transition_matrix, context_len)[:-1]

    k = dataset.k
    contexts = base_chain.expand(k, -1)
    last_tokens = torch.arange(k, device=dataset.device).unsqueeze(1)
    batch_tokens = torch.cat([contexts, last_tokens], dim=1)
    batch_tokens = dataset.prepend_bos(batch_tokens)

    logits = model(batch_tokens)
    return F.softmax(logits[:, -1, : dataset.k], dim=-1)


def evaluate_kl(
    model: MarkovTransformer,
    dataset: MarkovChainDataset,
    num_evals: int,
    context_len: int,
    is_ood: bool,
) -> float:
    """Evaluate weighted state-wise KL divergence for ID or OOD chains."""
    was_training = model.training
    model.eval()

    total_kl = 0.0
    eps = 1e-9

    for _ in range(num_evals):
        if is_ood:
            transition_matrix = dataset.sample_ood_matrix()
        else:
            idx = torch.randint(0, dataset.n_chains, (1,), device=dataset.device)
            transition_matrix = dataset.transition_matrices[idx].squeeze(0)

        estimated = estimate_transition_matrix(
            model=model,
            dataset=dataset,
            transition_matrix=transition_matrix,
            context_len=context_len,
        )
        stationary = dataset._compute_stationary_batch(
            transition_matrix.unsqueeze(0)
        ).squeeze(0)

        estimated_safe = torch.clamp(estimated, min=eps, max=1.0)
        transition_safe = torch.clamp(transition_matrix, min=eps, max=1.0)
        kl_per_state = torch.sum(
            estimated_safe * torch.log(estimated_safe / transition_safe),
            dim=1,
        )
        total_kl += torch.sum(stationary * kl_per_state).item()

    if was_training:
        model.train()

    return total_kl / num_evals


@torch.inference_mode()
def _sample_ood_tokens_bos(
    dataset: MarkovChainDataset,
    batch_size: int,
) -> torch.Tensor:
    """Sample BOS-prefixed chains from freshly-drawn OOD transition matrices."""
    k = dataset.k
    device = dataset.device
    dirichlet = Dirichlet(torch.ones(k, device=device))
    trans_mats = dirichlet.sample((batch_size, k))
    trans_mats = torch.clamp(torch.nan_to_num(trans_mats, nan=1e-9), min=1e-9)
    trans_mats = trans_mats / trans_mats.sum(dim=-1, keepdim=True)
    stationary = dataset._compute_stationary_batch(trans_mats)
    chains = dataset._generate_chains(trans_mats, stationary, dataset.seq_len)
    return dataset.prepend_bos(chains)


@torch.inference_mode()
def evaluate_baseline_deltas(
    model: MarkovTransformer,
    dataset: MarkovChainDataset,
    batch_size: int,
    is_ood: bool,
) -> dict[str, float]:
    """Symmetrised KL(model, baseline) per position for the four canonical baselines.

    Matches the BAU paper pipeline (metrics.symmetrised_kl) so deltas are
    cross-comparable between the two settings.

    Keys in the returned dict pair the model's computational strategy
    (generalising vs memorising) with how well the baseline's order matches the
    data (wellspec = order-1 bigram, matches the Markov DGP; misspec = order-0
    unigram, ignores transitions):

      - wellspec_generalising : bi_inf_predictive (Dirichlet(1) row prior)
      - wellspec_memorising   : bi_ret_predictive (discrete prior over {T_m})
      - misspec_generalising  : uni_inf_predictive (Dirichlet(1) stationary prior)
      - misspec_memorising    : uni_ret_predictive (discrete prior over {pi_m})

    Tokens come from either ID (training-pool chain) or OOD (fresh Dirichlet task)
    sequences. The symmetrised KL is averaged over all batch elements and
    predictable positions.
    """
    was_training = model.training
    model.eval()

    if is_ood:
        tokens_bos = _sample_ood_tokens_bos(dataset, batch_size)
    else:
        tokens_bos = dataset.sample_batch(batch_size)

    # training uses inputs=tokens_bos[:, :-1]; mirror that so we stay within
    # the model's trained context length. tokens_no_bos is the state sequence
    # the baselines operate on (no BOS). model logits at positions [1:] predict
    # tok_1..tok_{S-1}; baseline preds at positions [:-1] predict the same
    # tokens under the same conditioning.
    model_inputs = tokens_bos[:, :-1]
    tokens_no_bos = tokens_bos[:, 1:]

    logits = model(model_inputs)
    k = dataset.k
    model_probs = F.softmax(logits[:, 1:, :k], dim=-1)

    alpha_row = torch.ones(k, device=dataset.device)
    alpha_mat = torch.ones(k, k, device=dataset.device)

    baselines = {
        "wellspec_generalising": bi_inf_predictive(tokens_no_bos, alpha_mat),
        "wellspec_memorising": bi_ret_predictive(
            tokens_no_bos, dataset.transition_matrices
        ),
        "misspec_generalising": uni_inf_predictive(tokens_no_bos, alpha_row),
        "misspec_memorising": uni_ret_predictive(
            tokens_no_bos, dataset.stationary_distributions
        ),
    }

    result: dict[str, float] = {}
    for name, preds in baselines.items():
        result[name] = symmetrised_kl(model_probs, preds[:, :-1, :])

    if was_training:
        model.train()
    return result
