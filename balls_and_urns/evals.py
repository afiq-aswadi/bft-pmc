"""Evaluation for the Dirichlet-Multinomial (balls and urns) setting.

Compares model predictions against:
- generalising_predictive: Bayes-optimal under Dirichlet prior (analog of ridge)
- memorising_predictive: Bayes-optimal under discrete prior (analog of dMMSE)

Fields follow preds_on_data_{eval_dist}_vs_baseline_{baseline} — e.g.
preds_on_data_memorising_vs_baseline_generalising is the generalising (Dirichlet)
baseline applied to the memorising eval batch.
"""

import torch
from jaxtyping import Float

from pfn_transformerlens import sample_batch
from pfn_transformerlens.model.PFN import DistributionPrediction, UnsupervisedPFN
from pfn_transformerlens.sampler.prior_likelihood import DiscreteTaskDistribution

from balls_and_urns.baselines import generalising_predictive, memorising_predictive
from balls_and_urns.data import BOSGenerator


class BAUEvaluator:
    """Evaluator that compares model against Bayes-optimal baselines.

    Stores fixed evaluation batches computed once at initialization.
    Handles BOS-prepended sequences: baselines are computed on data tokens only.
    Model uses d_vocab_out=vocab_size so output excludes BOS logits.
    """

    def __init__(
        self,
        memorising_gen: BOSGenerator,
        generalising_gen: BOSGenerator,
        alpha: Float[torch.Tensor, " V"],
        eval_batch_size: int,
        seq_len: int,
    ):
        self.alpha = alpha
        _, self.memorising_tokens = sample_batch(
            memorising_gen, batch_size=eval_batch_size, seq_len=seq_len
        )
        _, self.generalising_tokens = sample_batch(
            generalising_gen, batch_size=eval_batch_size, seq_len=seq_len
        )
        assert self.memorising_tokens is not None
        assert self.generalising_tokens is not None

        # extract discrete theta pool from the underlying generator
        base_gen = memorising_gen.base
        assert isinstance(base_gen.prior.base_distribution, DiscreteTaskDistribution)
        thetas = base_gen.prior.base_distribution.tasks

        # baselines operate on data tokens (strip BOS at position 0)
        memorising_data = self.memorising_tokens[:, 1:]
        generalising_data = self.generalising_tokens[:, 1:]

        self.preds_on_data_memorising_vs_baseline_generalising = (
            generalising_predictive(memorising_data, alpha)
        )
        self.preds_on_data_memorising_vs_baseline_memorising = memorising_predictive(
            memorising_data, thetas
        )
        self.preds_on_data_generalising_vs_baseline_generalising = (
            generalising_predictive(generalising_data, alpha)
        )
        self.preds_on_data_generalising_vs_baseline_memorising = memorising_predictive(
            generalising_data, thetas
        )

    def __call__(self, model: UnsupervisedPFN) -> dict[str, float]:
        device = next(model.parameters()).device

        memorising_tokens = self.memorising_tokens.to(device)
        generalising_tokens = self.generalising_tokens.to(device)

        with torch.no_grad():
            memorising_pred = model.predict_on_prompt(memorising_tokens)
            if isinstance(memorising_pred, tuple):
                memorising_pred = memorising_pred[0]
            assert isinstance(memorising_pred, DistributionPrediction)
            memorising_probs = memorising_pred.probs.cpu()

            generalising_pred = model.predict_on_prompt(generalising_tokens)
            if isinstance(generalising_pred, tuple):
                generalising_pred = generalising_pred[0]
            assert isinstance(generalising_pred, DistributionPrediction)
            generalising_probs = generalising_pred.probs.cpu()

        # drop last position (predicts beyond sequence)
        memorising_probs = memorising_probs[:, :-1]
        generalising_probs = generalising_probs[:, :-1]

        memorising_ce = _cross_entropy(
            self.preds_on_data_memorising_vs_baseline_generalising, memorising_probs
        )
        generalising_ce = _cross_entropy(
            self.preds_on_data_generalising_vs_baseline_generalising, generalising_probs
        )

        memorising_delta_generalising = _mse(
            memorising_probs, self.preds_on_data_memorising_vs_baseline_generalising
        )
        memorising_delta_memorising = _mse(
            memorising_probs, self.preds_on_data_memorising_vs_baseline_memorising
        )
        generalising_delta_generalising = _mse(
            generalising_probs, self.preds_on_data_generalising_vs_baseline_generalising
        )
        generalising_delta_memorising = _mse(
            generalising_probs, self.preds_on_data_generalising_vs_baseline_memorising
        )

        memorising_per_pos = _per_position_ce(
            self.preds_on_data_memorising_vs_baseline_generalising, memorising_probs
        )
        generalising_per_pos = _per_position_ce(
            self.preds_on_data_generalising_vs_baseline_generalising, generalising_probs
        )
        k = memorising_per_pos.shape[0]

        return {
            "ce/data_memorising": memorising_ce,
            "ce/data_generalising": generalising_ce,
            "deltas/data_memorising/delta_vs_baseline_generalising": memorising_delta_generalising,
            "deltas/data_memorising/delta_vs_baseline_memorising": memorising_delta_memorising,
            "deltas/data_generalising/delta_vs_baseline_generalising": generalising_delta_generalising,
            "deltas/data_generalising/delta_vs_baseline_memorising": generalising_delta_memorising,
            "pertoken/data_memorising/0": memorising_per_pos[0].item(),
            f"pertoken/data_memorising/{k // 2}": memorising_per_pos[k // 2].item(),
            f"pertoken/data_memorising/{k - 1}": memorising_per_pos[k - 1].item(),
            "pertoken/data_generalising/0": generalising_per_pos[0].item(),
            f"pertoken/data_generalising/{k // 2}": generalising_per_pos[k // 2].item(),
            f"pertoken/data_generalising/{k - 1}": generalising_per_pos[k - 1].item(),
        }


def _mse(
    a: Float[torch.Tensor, "batch seq V"],
    b: Float[torch.Tensor, "batch seq V"],
) -> float:
    return (a - b).square().mean().item()


def _cross_entropy(
    target_probs: Float[torch.Tensor, "batch seq V"],
    model_probs: Float[torch.Tensor, "batch seq V"],
    eps: float = 1e-10,
) -> float:
    """Cross-entropy: -sum_v p_target(v) * log p_model(v), averaged over batch and seq."""
    log_model = torch.log(model_probs + eps)
    return -(target_probs * log_model).sum(dim=-1).mean().item()


def _per_position_ce(
    target_probs: Float[torch.Tensor, "batch seq V"],
    model_probs: Float[torch.Tensor, "batch seq V"],
    eps: float = 1e-10,
) -> Float[torch.Tensor, " seq"]:
    """Per-position cross-entropy, averaged over batch."""
    log_model = torch.log(model_probs + eps)
    return -(target_probs * log_model).sum(dim=-1).mean(dim=0)
