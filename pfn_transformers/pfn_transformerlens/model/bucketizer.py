from __future__ import annotations

import math
from typing import Literal
from warnings import warn

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from pfn_transformerlens.model.configs.regression import SupervisedRegressionPFNConfig
from pfn_transformerlens.model.configs.unsupervised import UnsupervisedPFNConfig

BucketType = Literal["uniform", "riemann"]
BucketSupport = Literal["bounded", "unbounded"]

HALF_NORMAL_TARGET_CDF = 0.5
HALF_NORMAL_ICDF_AT_TARGET = float(
    torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(
        torch.tensor(HALF_NORMAL_TARGET_CDF),
    ),
)
SQRT_TWO_OVER_PI = math.sqrt(2.0 / math.pi)
LOG_SQRT_TWO_OVER_PI = math.log(SQRT_TWO_OVER_PI)
LOG_HALF = math.log(0.5)


def estimate_riemann_borders(
    ys: Float[Tensor, "..."],
    *,
    num_buckets: int,
    widen_borders_factor: float = 1.0,
) -> Float[Tensor, " num_buckets+1"]:
    """
    Estimate quantile-based bucket borders for Riemann bucketization.
    As implemented in https://arxiv.org/abs/2112.10510
    """
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive.")
    flat = ys.reshape(-1)
    if flat.numel() == 0:
        raise ValueError("ys must contain at least one value.")
    clean = flat[~torch.isnan(flat)]
    if clean.numel() <= num_buckets:
        raise ValueError("Need more samples than buckets to estimate borders.")
    remainder = clean.numel() % num_buckets
    if remainder:
        clean = clean[:-remainder]
    ys_sorted = clean.sort().values
    bucket_size = clean.numel() // num_buckets
    upper_candidates = ys_sorted[bucket_size::bucket_size]
    lower_candidates = ys_sorted[bucket_size - 1 :: bucket_size][:-1]
    interior = 0.5 * (lower_candidates + upper_candidates)
    dtype = clean.dtype
    device = clean.device
    borders = torch.cat(
        (
            ys_sorted[:1],
            interior,
            ys_sorted[-1:].to(dtype=dtype, device=device),
        ),
    )
    if not math.isclose(widen_borders_factor, 1.0):
        borders = borders * float(widen_borders_factor)
    unique_borders = torch.unique_consecutive(borders)
    if unique_borders.numel() != borders.numel():
        warn("Bucket borders were not unique; removed duplicates.")
    if unique_borders.numel() - 1 != num_buckets:
        raise ValueError(
            "Estimated borders do not define the requested number of buckets.",
        )
    return unique_borders


class Bucketizer(nn.Module):
    """Utility that maps continuous targets to discrete buckets (and back)."""

    borders: Tensor
    bucket_widths: Tensor

    def __init__(
        self,
        *,
        bucket_type: BucketType,
        bucket_support: BucketSupport,
        d_vocab: int,
        y_min: float | None,
        y_max: float | None,
        borders: Tensor | None = None,
    ) -> None:
        super().__init__()
        if d_vocab < 2:
            raise ValueError("d_vocab must be >= 2 for bucketization.")
        if bucket_support not in {"bounded", "unbounded"}:
            raise ValueError(f"Unknown bucket support: {bucket_support}")

        self.bucket_type = bucket_type
        self.bucket_support = bucket_support
        self.d_vocab = d_vocab

        if bucket_type == "uniform":
            if y_min is None or y_max is None:
                raise ValueError(
                    "y_min/y_max must be provided for uniform bucketization."
                )
            if y_min >= y_max:
                raise ValueError(f"Require y_min < y_max (got {y_min} >= {y_max}).")
            borders = torch.linspace(float(y_min), float(y_max), steps=d_vocab + 1)
        elif bucket_type == "riemann":
            if borders is None:
                raise ValueError(
                    "Riemann bucketization requires precomputed borders.",
                )
        else:
            raise ValueError(f"Unknown bucket type: {bucket_type}")

        assert borders.ndim == 1, "Bucket borders must be 1D."
        assert len(borders) == d_vocab + 1, (
            f"Expected {d_vocab + 1} borders for {d_vocab} buckets (got {len(borders)})."
        )

        widths = borders[1:] - borders[:-1]
        assert torch.all(widths > 0), "Bucket borders must be strictly increasing."

        self.register_buffer("borders", borders.contiguous(), persistent=False)
        self.register_buffer("bucket_widths", widths, persistent=False)

    @classmethod
    def from_config(
        cls,
        config: SupervisedRegressionPFNConfig | UnsupervisedPFNConfig,
    ) -> "Bucketizer":
        assert config.bucket_type is not None
        return cls(
            bucket_type=config.bucket_type,
            bucket_support=config.bucket_support,
            d_vocab=config.d_vocab,
            y_min=config.y_min,
            y_max=config.y_max,
            borders=config.riemann_borders,
        )

    @property
    def num_buckets(self) -> int:
        return self.d_vocab

    def bucketize(self, y: Float[Tensor, "..."]) -> Int[Tensor, "..."]:
        """Map continuous values to discrete bucket indices.

        Args:
            y: Continuous values to bucketize.

        Returns:
            Bucket indices in range [0, num_buckets-1].
        """
        borders = self.borders.to(device=y.device, dtype=y.dtype)
        indices = torch.searchsorted(borders, y.contiguous(), side="left") - 1
        return indices.clamp(min=0, max=self.num_buckets - 1)

    def decode(
        self,
        bucket_indices: Int[Tensor, "..."],
        *,
        dtype: torch.dtype | None = None,
    ) -> Float[Tensor, "..."]:
        """Convert bucket indices back to continuous values.

        Returns bucket representative values (midpoints for bounded support
        and interior buckets; modes for unbounded edge buckets).

        Args:
            bucket_indices: Bucket indices to decode.
            dtype: Optional dtype for output tensor.

        Returns:
            Continuous values corresponding to bucket representatives.
            For unbounded support, edge bucket values are modes (at inner boundary).
        """
        reps = self.bucket_representatives()
        reps = reps.to(device=bucket_indices.device)
        if dtype is not None:
            reps = reps.to(dtype=dtype)
        bucket_indices = bucket_indices.clamp(min=0, max=self.num_buckets - 1)
        return reps[bucket_indices]

    def bucket_representatives(self) -> Float[Tensor, "num_buckets"]:
        """Get representative values (means) for each bucket.

        For bounded support, returns bucket midpoints. For unbounded support,
        interior buckets use midpoints and edge buckets use half-normal means.

        Returns:
            Bucket means for each bucket.
            - Bounded: All buckets use midpoints
            - Unbounded: Interior buckets use midpoints, edge buckets use
              half-normal means (shifted outward from the inner boundary)
        """
        mids = self.borders[:-1] + 0.5 * self.bucket_widths

        if self.bucket_support == "unbounded":
            means = mids.clone()
            means[0] = self._left_edge_bucket_mean()
            means[-1] = self._right_edge_bucket_mean()
            return means

        return mids

    @torch.no_grad()
    def sample(
        self,
        logits: Float[Tensor, "... num_buckets"],
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "..."]:
        """Sample continuous values from piecewise-uniform distribution.

        Uses inverse CDF with linear interpolation within buckets. For unbounded
        support, edge buckets are modeled as full half-normal distributions
        anchored at the inner boundary, matching the density used in training loss.

        Parameters:
            logits: Logits over buckets. Supports arbitrary batch dimensions.
            temperature: Sampling temperature applied before softmax.
            generator: Optional RNG generator for deterministic sampling.

        Returns:
            Continuous values sampled from the distribution.
            Returns values on same device/dtype as logits.

        Notes:
            - For bounded support: samples uniformly within bucket boundaries
            - For unbounded support: edge buckets follow half-normal distributions
            - Numerically stable at extreme temperatures via epsilon flooring
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        device = logits.device
        orig_dtype = logits.dtype
        batch_shape = logits.shape[:-1]

        # use higher-precision math for low-precision logits to avoid CPU restrictions
        calc_dtype = (
            torch.float32
            if orig_dtype in (torch.float16, torch.bfloat16)
            else orig_dtype
        )
        logits_calc = logits.to(calc_dtype)

        # apply temperature and softmax
        probs = torch.softmax(logits_calc / float(temperature), dim=-1)

        # sample uniform p ~ U(0,1) for inverse CDF in calculation dtype, then cast back
        p_cdf = torch.rand(
            batch_shape,
            device=device,
            dtype=calc_dtype,
            generator=generator,
        )

        if self.bucket_support == "unbounded":
            samples = self._sample_unbounded(probs, p_cdf, calc_dtype)
        else:
            samples = self._sample_bounded(probs, p_cdf, calc_dtype)

        return samples.to(orig_dtype)

    def _sample_bounded(
        self,
        probs: Float[Tensor, "... num_buckets"],
        p_cdf: Float[Tensor, "..."],
        calc_dtype: torch.dtype,
    ) -> Float[Tensor, "..."]:
        """Sample from bounded piecewise-uniform distribution."""
        # standard inverse CDF: find bucket via cumulative probabilities
        cumprobs = torch.cumsum(probs, dim=-1)

        # find which bucket each p falls into
        bucket_idx = torch.searchsorted(
            cumprobs.contiguous(),
            p_cdf.unsqueeze(-1).contiguous(),
        ).squeeze(-1)
        bucket_idx = bucket_idx.clamp(max=self.num_buckets - 1)

        # get remaining probability within the bucket
        cumprobs_before = torch.cat(
            [torch.zeros_like(cumprobs[..., :1]), cumprobs[..., :-1]],
            dim=-1,
        )
        rest_prob = p_cdf - cumprobs_before.gather(
            -1, bucket_idx.unsqueeze(-1)
        ).squeeze(-1)

        # get bucket boundaries and probabilities
        borders = self.borders.to(device=probs.device, dtype=calc_dtype)
        left_border = borders[bucket_idx]
        right_border = borders[bucket_idx + 1]
        bucket_probs = probs.gather(-1, bucket_idx.unsqueeze(-1)).squeeze(-1)

        # numerical guard: avoid division by near-zero probabilities
        bucket_probs = bucket_probs.clamp_min(1e-12)

        # linear interpolation within bucket, clamped to [0, 1] in same dtype
        interp_factor = (rest_prob / bucket_probs).clamp(0.0, 1.0).to(probs.dtype)
        samples = left_border + (right_border - left_border) * interp_factor

        # sanity check
        assert torch.all(torch.isfinite(samples)), "Samples must be finite"

        return samples

    def _sample_unbounded(
        self,
        probs: Float[Tensor, "... num_buckets"],
        p_cdf: Float[Tensor, "..."],
        calc_dtype: torch.dtype,
    ) -> Float[Tensor, "..."]:
        """Sample from unbounded distribution with half-normal edge buckets.

        Edge buckets use half-normal distributions throughout:
        - Bucket 0: Y ~ borders[1] - Z where Z ~ HalfNormal(scale)
        - Bucket n-1: Y ~ borders[-2] + Z where Z ~ HalfNormal(scale)
        - Interior buckets: uniform within boundaries
        """
        device = probs.device
        borders = self.borders.to(device=device, dtype=calc_dtype)

        # cumulative probabilities for finding which bucket to sample from
        cumprobs = torch.cumsum(probs, dim=-1)

        # find which bucket each p falls into
        bucket_idx = torch.searchsorted(
            cumprobs.contiguous(),
            p_cdf.unsqueeze(-1).contiguous(),
        ).squeeze(-1)
        bucket_idx = bucket_idx.clamp(max=self.num_buckets - 1)

        # get remaining probability within the bucket
        cumprobs_before = torch.cat(
            [torch.zeros_like(cumprobs[..., :1]), cumprobs[..., :-1]],
            dim=-1,
        )
        rest_prob = p_cdf - cumprobs_before.gather(
            -1, bucket_idx.unsqueeze(-1)
        ).squeeze(-1)

        # get bucket probabilities
        bucket_probs = probs.gather(-1, bucket_idx.unsqueeze(-1)).squeeze(-1)
        bucket_probs = bucket_probs.clamp_min(1e-12)

        # renormalize to [0, 1] within bucket
        p_in_bucket = (rest_prob / bucket_probs).clamp(0.0, 1.0)

        # handle scalar case
        is_scalar = p_cdf.dim() == 0

        if is_scalar:
            if bucket_idx.item() == 0:
                # left edge bucket: Y ~ borders[1] - Z
                scale = self._halfnormal_scale(self.bucket_widths[0]).to(
                    device=device, dtype=calc_dtype
                )
                # inverse CDF: y = borders[1] - scale * sqrt(2) * erfinv(1 - p)
                distance = (
                    scale
                    * (2.0**0.5)
                    * torch.erfinv(
                        (1.0 - p_in_bucket)
                        .clamp(min=1e-7, max=0.9999999)
                        .to(calc_dtype)
                    )
                )
                return borders[1] - distance
            elif bucket_idx.item() == self.num_buckets - 1:
                # right edge bucket: Y ~ borders[-2] + Z
                scale = self._halfnormal_scale(self.bucket_widths[-1]).to(
                    device=device, dtype=calc_dtype
                )
                # inverse CDF: y = borders[-2] + scale * sqrt(2) * erfinv(p)
                distance = (
                    scale
                    * (2.0**0.5)
                    * torch.erfinv(
                        p_in_bucket.clamp(min=1e-7, max=0.9999999).to(calc_dtype)
                    )
                )
                return borders[-2] + distance
            else:
                # interior bucket: uniform interpolation
                left_border = borders[bucket_idx]
                right_border = borders[bucket_idx + 1]
                return left_border + (right_border - left_border) * p_in_bucket

        # multi-dimensional case
        samples = torch.empty_like(p_cdf, dtype=calc_dtype)

        # left edge bucket mask
        left_bucket_mask = bucket_idx == 0
        if left_bucket_mask.any():
            scale = self._halfnormal_scale(self.bucket_widths[0]).to(
                device=device, dtype=calc_dtype
            )
            distances = (
                scale
                * (2.0**0.5)
                * torch.erfinv(
                    (1.0 - p_in_bucket[left_bucket_mask])
                    .clamp(min=1e-7, max=0.9999999)
                    .to(calc_dtype)
                )
            )
            samples[left_bucket_mask] = borders[1] - distances

        # right edge bucket mask
        right_bucket_mask = bucket_idx == self.num_buckets - 1
        if right_bucket_mask.any():
            scale = self._halfnormal_scale(self.bucket_widths[-1]).to(
                device=device, dtype=calc_dtype
            )
            distances = (
                scale
                * (2.0**0.5)
                * torch.erfinv(
                    p_in_bucket[right_bucket_mask]
                    .clamp(min=1e-7, max=0.9999999)
                    .to(calc_dtype)
                )
            )
            samples[right_bucket_mask] = borders[-2] + distances

        # interior buckets: uniform interpolation
        interior_mask = ~(left_bucket_mask | right_bucket_mask)
        if interior_mask.any():
            left_border = borders[bucket_idx[interior_mask]]
            right_border = borders[bucket_idx[interior_mask] + 1]
            samples[interior_mask] = (
                left_border + (right_border - left_border) * p_in_bucket[interior_mask]
            )

        # sanity check
        assert torch.all(torch.isfinite(samples)), "Samples must be finite"

        return samples

    def log_bucket_densities(
        self,
        logits: Float[Tensor, "... num_buckets"],
    ) -> Float[Tensor, "... num_buckets"]:
        """Convert logits into log densities by accounting for bucket widths."""
        log_probs = torch.log_softmax(logits, dim=-1)
        log_widths = torch.log(self.bucket_widths).to(
            device=logits.device,
            dtype=logits.dtype,
        )
        return log_probs - log_widths

    def log_density_at_values(
        self,
        logits: Float[Tensor, "... num_buckets"],
        values: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        """Return log density f(y) evaluated at arbitrary continuous targets."""
        y = values.to(device=logits.device, dtype=logits.dtype)
        log_probs = torch.log_softmax(logits, dim=-1)
        bucket_indices = self.bucketize(y).long()
        gathered = log_probs.gather(-1, bucket_indices.unsqueeze(-1)).squeeze(-1)
        widths = self.bucket_widths.to(device=y.device, dtype=y.dtype)

        # Assert widths are positive
        assert torch.all(widths > 0), "Bucket widths must be positive"

        if self.bucket_support == "bounded":
            selected_widths = widths[bucket_indices]
            return gathered - torch.log(selected_widths)

        borders = self.borders.to(device=y.device, dtype=y.dtype)
        selected_widths = widths[bucket_indices]
        result = torch.empty_like(gathered)
        num_buckets = self.num_buckets

        interior_mask = (bucket_indices > 0) & (bucket_indices < num_buckets - 1)
        if interior_mask.any():
            result[interior_mask] = gathered[interior_mask] - torch.log(
                selected_widths[interior_mask],
            )

        # Left bucket: use half-normal throughout
        # Model: Y ~ borders[1] - Z where Z ~ HalfNormal(scale)
        left_bucket_mask = bucket_indices == 0
        if left_bucket_mask.any():
            left_scale = self._halfnormal_scale(self.bucket_widths[0]).to(
                device=y.device,
                dtype=y.dtype,
            )
            distances = (borders[1] - y[left_bucket_mask]).clamp_min(0.0)
            log_pdf = self._half_normal_log_pdf(distances, left_scale)
            result[left_bucket_mask] = gathered[left_bucket_mask] + log_pdf

        # Right bucket: use half-normal throughout
        # Model: Y ~ borders[-2] + Z where Z ~ HalfNormal(scale)
        right_bucket_mask = bucket_indices == num_buckets - 1
        if right_bucket_mask.any():
            right_scale = self._halfnormal_scale(self.bucket_widths[-1]).to(
                device=y.device,
                dtype=y.dtype,
            )
            distances = (y[right_bucket_mask] - borders[-2]).clamp_min(0.0)
            log_pdf = self._half_normal_log_pdf(distances, right_scale)
            result[right_bucket_mask] = gathered[right_bucket_mask] + log_pdf

        # Assert all values are finite
        assert torch.all(torch.isfinite(result)), "Log densities must be finite"

        return result

    def _left_edge_bucket_mean(self) -> Tensor:
        """Compute mean of left edge bucket for unbounded support.

        Model: Y ~ borders[1] - Z where Z ~ HalfNormal(scale)
        E[Y] = borders[1] - E[Z] = borders[1] - scale * sqrt(2/pi)
        """
        scale = self._halfnormal_scale(self.bucket_widths[0])
        mean_offset = scale * SQRT_TWO_OVER_PI
        return self.borders[1] - mean_offset

    def _right_edge_bucket_mean(self) -> Tensor:
        """Compute mean of right edge bucket for unbounded support.

        Model: Y ~ borders[-2] + Z where Z ~ HalfNormal(scale)
        E[Y] = borders[-2] + E[Z] = borders[-2] + scale * sqrt(2/pi)
        """
        scale = self._halfnormal_scale(self.bucket_widths[-1])
        mean_offset = scale * SQRT_TWO_OVER_PI
        return self.borders[-2] + mean_offset

    @staticmethod
    def _halfnormal_scale(width: Tensor) -> Tensor:
        """Compute scale parameter for half-normal distribution from bucket width."""
        return width / HALF_NORMAL_ICDF_AT_TARGET

    @staticmethod
    def _half_normal_log_pdf(distances: Tensor, scale: Tensor) -> Tensor:
        """Compute log PDF of half-normal distribution."""
        return (
            LOG_SQRT_TWO_OVER_PI
            - torch.log(scale)
            - 0.5 * torch.square(distances / scale)
        )
