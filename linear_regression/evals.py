"""Evaluation metrics for in-context linear regression experiments.

Compares model predictions against Bayes-optimal baselines (memorising and
generalising) on both the memorising eval distribution (discrete tasks) and
the generalising eval distribution (Gaussian prior).
"""

import torch
from jaxtyping import Float

from linear_regression.baselines import generalising_predictor, memorising_predictor
from linear_regression.priors import DiscretePrior
from pfn_transformerlens import DeterministicGenerator, sample_batch
from pfn_transformerlens.model.PFN import DistributionPrediction, SupervisedPFN


def mse(
    y1: Float[torch.Tensor, "..."],
    y2: Float[torch.Tensor, "..."],
    axis: int | tuple[int, ...] | None = None,
) -> Float[torch.Tensor, "..."]:
    """Mean squared error between two tensors.

    Args:
        y1: First tensor
        y2: Second tensor
        axis: Dimensions to average over (None = all dimensions)

    Returns:
        MSE scalar (if axis=None) or tensor of per-axis MSEs
    """
    return (y1 - y2).square().mean(dim=axis)


class ICLEvaluator:
    """Evaluator that compares model against Bayes-optimal baselines.

    Stores fixed evaluation batches computed once at initialization.
    Computes metrics by comparing model predictions to:
    - Memorising: Bayes-optimal for the discrete task pool (dMMSE)
    - Generalising: Bayes-optimal for the Gaussian prior (ridge)

    Fields follow preds_on_data_{eval_dist}_vs_baseline_{baseline} — e.g.
    preds_on_data_memorising_vs_baseline_generalising is the generalising (ridge)
    baseline applied to the memorising eval batch.
    """

    def __init__(
        self,
        memorising_gen: DeterministicGenerator,
        generalising_gen: DeterministicGenerator,
        max_examples: int,
        eval_batch_size: int,
        noise_variance: float,
    ):
        """Initialize evaluator with fixed evaluation batches.

        Args:
            memorising_gen: Data generator with discrete prior (finite tasks)
            generalising_gen: Data generator with Gaussian prior (infinite tasks)
            max_examples: Sequence length for evaluation
            eval_batch_size: Number of sequences in each batch
            noise_variance: Known noise variance for baselines
        """
        self.noise_variance = noise_variance

        self.memorising_xs, self.memorising_ys = sample_batch(
            memorising_gen, batch_size=eval_batch_size, seq_len=max_examples
        )
        assert self.memorising_xs is not None
        assert self.memorising_ys is not None
        self.generalising_xs, self.generalising_ys = sample_batch(
            generalising_gen, batch_size=eval_batch_size, seq_len=max_examples
        )
        assert self.generalising_xs is not None
        assert self.generalising_ys is not None

        if self.memorising_ys.dim() == 2:
            self.memorising_ys = self.memorising_ys.unsqueeze(-1)
        if self.generalising_ys.dim() == 2:
            self.generalising_ys = self.generalising_ys.unsqueeze(-1)

        assert isinstance(memorising_gen.prior, DiscretePrior), (
            "memorising_gen must use DiscretePrior"
        )
        discrete_prior = memorising_gen.prior

        self.preds_on_data_memorising_vs_baseline_memorising = memorising_predictor(
            self.memorising_xs, self.memorising_ys, discrete_prior, noise_variance
        )
        self.preds_on_data_memorising_vs_baseline_generalising = generalising_predictor(
            self.memorising_xs, self.memorising_ys, noise_variance
        )
        self.preds_on_data_generalising_vs_baseline_memorising = memorising_predictor(
            self.generalising_xs, self.generalising_ys, discrete_prior, noise_variance
        )
        self.preds_on_data_generalising_vs_baseline_generalising = (
            generalising_predictor(
                self.generalising_xs, self.generalising_ys, noise_variance
            )
        )

    def __call__(self, model: SupervisedPFN) -> dict[str, float]:
        """Evaluate model on fixed batches and return metrics.

        Args:
            model: Trained PFN model with predict_on_prompt method

        Returns:
            Dictionary of metrics:
            - mse/data_memorising, mse/data_generalising
            - deltas/data_{eval_dist}/delta_vs_baseline_{baseline} for eval_dist, baseline in
              {memorising, generalising}
            - pertoken/data_{eval_dist}/{pos} at positions 0, k//2, k-1
        """
        device = next(model.parameters()).device
        assert self.memorising_xs is not None
        assert self.memorising_ys is not None
        assert self.generalising_xs is not None
        assert self.generalising_ys is not None

        memorising_xs = self.memorising_xs.to(device)
        memorising_ys = self.memorising_ys.to(device)
        generalising_xs = self.generalising_xs.to(device)
        generalising_ys = self.generalising_ys.to(device)

        with torch.no_grad():
            memorising_pred = model.predict_on_prompt(
                memorising_xs, memorising_ys.squeeze(-1)
            )
            if isinstance(memorising_pred, tuple):
                memorising_pred = memorising_pred[0]
            assert isinstance(memorising_pred, DistributionPrediction)
            memorising_model_preds = (
                (memorising_pred.probs * memorising_pred.y_grid)
                .sum(dim=-1)
                .unsqueeze(-1)
            )

        memorising_model_losses = mse(
            self.memorising_ys, memorising_model_preds.cpu(), axis=(0, 2)
        )

        with torch.no_grad():
            generalising_pred = model.predict_on_prompt(
                generalising_xs, generalising_ys.squeeze(-1)
            )
            if isinstance(generalising_pred, tuple):
                generalising_pred = generalising_pred[0]
            assert isinstance(generalising_pred, DistributionPrediction)
            generalising_model_preds = (
                (generalising_pred.probs * generalising_pred.y_grid)
                .sum(dim=-1)
                .unsqueeze(-1)
            )

        generalising_model_losses = mse(
            self.generalising_ys, generalising_model_preds.cpu(), axis=(0, 2)
        )

        k = len(memorising_model_losses)

        return {
            "mse/data_memorising": memorising_model_losses.mean().item(),
            "mse/data_generalising": generalising_model_losses.mean().item(),
            "deltas/data_memorising/delta_vs_baseline_memorising": mse(
                memorising_model_preds.cpu(),
                self.preds_on_data_memorising_vs_baseline_memorising,
            ).item(),
            "deltas/data_memorising/delta_vs_baseline_generalising": mse(
                memorising_model_preds.cpu(),
                self.preds_on_data_memorising_vs_baseline_generalising,
            ).item(),
            "deltas/data_generalising/delta_vs_baseline_memorising": mse(
                generalising_model_preds.cpu(),
                self.preds_on_data_generalising_vs_baseline_memorising,
            ).item(),
            "deltas/data_generalising/delta_vs_baseline_generalising": mse(
                generalising_model_preds.cpu(),
                self.preds_on_data_generalising_vs_baseline_generalising,
            ).item(),
            "pertoken/data_memorising/0": memorising_model_losses[0].item(),
            f"pertoken/data_memorising/{k // 2}": memorising_model_losses[
                k // 2
            ].item(),
            f"pertoken/data_memorising/{k - 1}": memorising_model_losses[k - 1].item(),
            "pertoken/data_generalising/0": generalising_model_losses[0].item(),
            f"pertoken/data_generalising/{k // 2}": generalising_model_losses[
                k // 2
            ].item(),
            f"pertoken/data_generalising/{k - 1}": generalising_model_losses[
                k - 1
            ].item(),
        }
