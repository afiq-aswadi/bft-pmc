from __future__ import annotations

import math
import pytest
import torch

from pfn_transformerlens.model.bucketizer import (
    Bucketizer,
    BucketSupport,
)


def build_uniform_bucketizer(bucket_support: BucketSupport) -> Bucketizer:
    return Bucketizer(
        bucket_type="uniform",
        bucket_support=bucket_support,
        d_vocab=4,
        y_min=-2.0,
        y_max=2.0,
    )


def build_riemann_bucketizer(
    borders: list[float],
    bucket_support: BucketSupport = "bounded",
) -> Bucketizer:
    return Bucketizer(
        bucket_type="riemann",
        bucket_support=bucket_support,
        d_vocab=len(borders) - 1,
        y_min=None,
        y_max=None,
        borders=torch.tensor(borders),
    )


def test_uniform_bucket_mapping_midpoints() -> None:
    bucketizer = build_uniform_bucketizer("bounded")
    y = torch.tensor([-2.0, -0.9, 0.0, 1.7])
    indices = bucketizer.bucketize(y)
    assert indices.tolist() == [0, 1, 1, 3]

    reps = bucketizer.bucket_representatives()
    expected = torch.tensor([-1.5, -0.5, 0.5, 1.5])
    assert torch.allclose(reps, expected)

    decoded = bucketizer.decode(indices)
    assert torch.allclose(decoded, expected[[0, 1, 1, 3]])


def test_unbounded_bucket_representatives_are_means() -> None:
    # For unbounded support, edge buckets use half-normal means
    # while interior buckets use midpoints
    bucketizer = build_uniform_bucketizer("unbounded")
    reps = bucketizer.bucket_representatives()

    # edge buckets should be half-normal means (shifted outward from inner boundary)
    assert reps[0] < bucketizer.borders[1]
    assert reps[-1] > bucketizer.borders[-2]
    assert torch.isclose(reps[0], bucketizer._left_edge_bucket_mean())
    assert torch.isclose(reps[-1], bucketizer._right_edge_bucket_mean())

    # interior buckets should be midpoints
    expected_mids = bucketizer.borders[1:-2] + 0.5 * bucketizer.bucket_widths[1:-1]
    assert torch.allclose(reps[1:-1], expected_mids)


def test_bucketize_clamps_out_of_range_values() -> None:
    bucketizer = build_uniform_bucketizer("bounded")
    y = torch.tensor([-10.0, -2.001, -1.0, 0.0, 2.0, 10.0])
    indices = bucketizer.bucketize(y)
    assert indices.tolist() == [0, 0, 0, 1, 3, 3]


def test_riemann_bucketizer_respects_custom_borders() -> None:
    borders = [-3.0, -1.5, -0.2, 1.0, 4.0, 6.0]
    bucketizer = build_riemann_bucketizer(borders)
    widths = bucketizer.bucket_widths
    assert torch.allclose(
        widths,
        torch.tensor([1.5, 1.3, 1.2, 3.0, 2.0]),
    )

    y = torch.tensor([-2.0, -0.2, 0.0, 10.0])
    indices = bucketizer.bucketize(y)
    assert indices.tolist() == [0, 1, 2, 4]

    decoded = bucketizer.decode(indices)
    mids = bucketizer.bucket_representatives()
    assert torch.allclose(decoded, mids[indices])


def test_riemann_bucketizer_requires_borders() -> None:
    with pytest.raises(ValueError):
        Bucketizer(
            bucket_type="riemann",
            bucket_support="bounded",
            d_vocab=3,
            y_min=None,
            y_max=None,
            borders=None,
        )


def test_log_bucket_densities_scale_softmax() -> None:
    bucketizer = build_uniform_bucketizer("bounded")
    logits = torch.zeros(2, bucketizer.num_buckets)
    log_densities = bucketizer.log_bucket_densities(logits)
    expected = torch.log_softmax(logits, dim=-1) - torch.log(bucketizer.bucket_widths)
    assert torch.allclose(log_densities, expected)


def test_log_density_at_values_matches_bucket_lookup_for_bounded() -> None:
    bucketizer = build_uniform_bucketizer("bounded")
    logits = torch.randn(4, bucketizer.num_buckets)
    y = torch.tensor([-1.7, -0.2, 0.4, 1.6])

    log_density = bucketizer.log_density_at_values(logits, y)
    log_bucket = bucketizer.log_bucket_densities(logits)
    gathered = log_bucket.gather(-1, bucketizer.bucketize(y).unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(log_density, gathered)


def test_log_density_at_values_uses_half_normal_tails_for_unbounded() -> None:
    # Edge buckets use half-normal throughout (no 50/50 split)
    # Model: bucket 0 uses Y ~ borders[1] - Z, bucket n-1 uses Y ~ borders[-2] + Z
    bucketizer = build_uniform_bucketizer("unbounded")
    logits = torch.zeros(4, bucketizer.num_buckets)
    y = torch.tensor(
        [
            bucketizer.borders[0].item() - 0.5,  # Left edge bucket, beyond borders[0]
            bucketizer.borders[0].item()
            + 0.25,  # Left edge bucket, within [borders[0], borders[1])
            0.5,  # Interior bucket
            bucketizer.borders[-1].item()
            + 0.75,  # Right edge bucket, beyond borders[-1]
        ],
        dtype=torch.float32,
    )

    log_density = bucketizer.log_density_at_values(logits, y)
    log_probs = torch.log_softmax(logits, dim=-1)

    left_scale = bucketizer._halfnormal_scale(bucketizer.bucket_widths[0]).to(
        log_probs.dtype
    )
    right_scale = bucketizer._halfnormal_scale(bucketizer.bucket_widths[-1]).to(
        log_probs.dtype
    )

    # Left edge bucket: distance from borders[1] for ALL values
    left_distance_tail = (bucketizer.borders[1] - y[0]).to(log_probs.dtype)
    left_distance_interior = (bucketizer.borders[1] - y[1]).to(log_probs.dtype)

    # Right edge bucket: distance from borders[-2] for ALL values
    right_distance = (y[3] - bucketizer.borders[-2]).to(log_probs.dtype)

    mid_bucket = int(bucketizer.bucketize(y[2:3]).item())
    mid_width = bucketizer.bucket_widths[mid_bucket].to(log_probs.dtype)

    expected = torch.stack(
        [
            # Left edge bucket (tail region): use half-normal from borders[1]
            log_probs[0, 0]
            + Bucketizer._half_normal_log_pdf(
                left_distance_tail,
                left_scale,
            ),
            # Left edge bucket (interior region): use half-normal from borders[1]
            log_probs[1, 0]
            + Bucketizer._half_normal_log_pdf(
                left_distance_interior,
                left_scale,
            ),
            # Interior bucket: use piecewise constant
            log_probs[2, mid_bucket] - torch.log(mid_width),
            # Right edge bucket: use half-normal from borders[-2]
            log_probs[3, bucketizer.num_buckets - 1]
            + Bucketizer._half_normal_log_pdf(
                right_distance,
                right_scale,
            ),
        ],
    )

    assert torch.allclose(log_density, expected)


def test_sample_bounded_histogram() -> None:
    """Verify sampling produces correct bucket probabilities for bounded support."""
    bucketizer = build_uniform_bucketizer("bounded")
    num_samples = 50000

    # uniform logits should give equal probability to all buckets
    logits = torch.zeros(bucketizer.num_buckets)

    # test with multiple seeds to avoid flaky failures
    for seed in [42, 123, 456]:
        gen = torch.Generator().manual_seed(seed)
        samples = torch.stack(
            [
                bucketizer.sample(logits, temperature=1.0, generator=gen)
                for _ in range(num_samples)
            ]
        )

        # verify all samples within bounds
        assert torch.all(samples >= bucketizer.borders[0])
        assert torch.all(samples <= bucketizer.borders[-1])

        # build histogram
        bucket_indices = bucketizer.bucketize(samples)
        counts = torch.bincount(bucket_indices.long(), minlength=bucketizer.num_buckets)

        # expected: uniform across 4 buckets
        expected_probs = torch.ones(4) * 0.25

        # chi-squared test: sum((obs - exp)^2 / exp)
        chi_squared = torch.sum(
            (counts.float() - num_samples * expected_probs) ** 2
            / (num_samples * expected_probs)
        )

        # for 3 degrees of freedom (4 buckets - 1), critical value at p=0.001 is ~16.3
        # with 50k samples, we should pass comfortably
        assert chi_squared < 20.0, (
            f"Chi-squared test failed with {chi_squared} (seed={seed})"
        )


def test_sample_unbounded_tails() -> None:
    """Verify sampling from half-normal tails for unbounded support."""
    bucketizer = build_uniform_bucketizer("unbounded")
    num_samples = 50000

    # heavily skewed logits favoring edge buckets
    logits = torch.tensor([2.0, 0.0, 0.0, 2.0])
    gen = torch.Generator().manual_seed(42)

    samples = torch.stack(
        [
            bucketizer.sample(logits, temperature=1.0, generator=gen)
            for _ in range(num_samples)
        ]
    )

    # verify all samples are finite
    assert torch.all(torch.isfinite(samples))

    # for edge buckets, approximately 50% of mass should be in tails
    left_bucket_samples = (samples >= bucketizer.borders[0]) & (
        samples < bucketizer.borders[1]
    )
    right_bucket_samples = (samples >= bucketizer.borders[-2]) & (
        samples < bucketizer.borders[-1]
    )
    left_tail_samples = samples < bucketizer.borders[0]
    right_tail_samples = samples >= bucketizer.borders[-1]

    # total samples in bucket 0 region (interior + tail)
    total_left = left_bucket_samples.sum() + left_tail_samples.sum()
    # approximately 50% should be in tail
    left_tail_ratio = left_tail_samples.sum().float() / total_left
    assert 0.45 < left_tail_ratio < 0.55, (
        f"Left tail ratio {left_tail_ratio} not near 0.5"
    )

    total_right = right_bucket_samples.sum() + right_tail_samples.sum()
    right_tail_ratio = right_tail_samples.sum().float() / total_right
    assert 0.45 < right_tail_ratio < 0.55, (
        f"Right tail ratio {right_tail_ratio} not near 0.5"
    )

    # verify tail samples are reasonable distances from borders (not at exact boundaries)
    if left_tail_samples.any():
        left_tail_distances = bucketizer.borders[0] - samples[left_tail_samples]
        assert torch.all(left_tail_distances > 0), (
            "Left tail samples should be below border"
        )
        # verify distances are reasonable (not too extreme)
        assert left_tail_distances.max() < 10.0, "Left tail distances too large"

    if right_tail_samples.any():
        right_tail_distances = samples[right_tail_samples] - bucketizer.borders[-1]
        assert torch.all(right_tail_distances > 0), (
            "Right tail samples should be above border"
        )
        # verify distances are reasonable (not too extreme)
        assert right_tail_distances.max() < 10.0, "Right tail distances too large"


def test_sample_temperature_scaling() -> None:
    """Verify temperature affects concentration of samples."""
    bucketizer = build_uniform_bucketizer("bounded")
    logits = torch.tensor([0.0, 3.0, 1.0, 0.0])  # strongly favors bucket 1
    num_samples = 10000

    # low temperature: concentrated
    gen_cold = torch.Generator().manual_seed(42)
    samples_cold = torch.stack(
        [
            bucketizer.sample(logits, temperature=0.1, generator=gen_cold)
            for _ in range(num_samples)
        ]
    )
    bucket_indices_cold = bucketizer.bucketize(samples_cold)
    counts_cold = torch.bincount(
        bucket_indices_cold.long(), minlength=bucketizer.num_buckets
    )
    prob_bucket1_cold = counts_cold[1].float() / num_samples

    # high temperature: more diffuse
    gen_hot = torch.Generator().manual_seed(42)
    samples_hot = torch.stack(
        [
            bucketizer.sample(logits, temperature=2.0, generator=gen_hot)
            for _ in range(num_samples)
        ]
    )
    bucket_indices_hot = bucketizer.bucketize(samples_hot)
    counts_hot = torch.bincount(
        bucket_indices_hot.long(), minlength=bucketizer.num_buckets
    )
    prob_bucket1_hot = counts_hot[1].float() / num_samples

    # cold should have >90% in bucket 1, hot should be more spread
    assert prob_bucket1_cold > 0.9, (
        f"Cold temperature gave {prob_bucket1_cold} in favored bucket"
    )
    assert prob_bucket1_hot < 0.8, (
        f"Hot temperature gave {prob_bucket1_hot} in favored bucket"
    )

    # entropy should be higher for hot
    def entropy(probs):
        p_nonzero = probs[probs > 0]
        return -torch.sum(p_nonzero * torch.log(p_nonzero))

    entropy_cold = entropy(counts_cold.float() / num_samples)
    entropy_hot = entropy(counts_hot.float() / num_samples)
    assert entropy_hot > entropy_cold, "Hot temperature should have higher entropy"


def test_sample_deterministic_mode_limit() -> None:
    """Verify that very low temperature concentrates samples in mode bucket."""
    bucketizer = build_uniform_bucketizer("bounded")
    logits = torch.tensor([0.0, 3.0, 1.0, 0.0])  # mode is bucket 1
    num_samples = 1000

    gen = torch.Generator().manual_seed(42)
    samples = torch.stack(
        [
            bucketizer.sample(logits, temperature=1e-6, generator=gen)
            for _ in range(num_samples)
        ]
    )

    # all samples should be in bucket 1
    bucket_indices = bucketizer.bucketize(samples)
    mode_bucket = torch.argmax(torch.softmax(logits, dim=-1))
    assert torch.all(bucket_indices == mode_bucket), (
        "All samples should be in mode bucket"
    )

    # samples should be distributed within the bucket, near midpoint
    assert torch.all(samples >= bucketizer.borders[mode_bucket])
    assert torch.all(samples < bucketizer.borders[mode_bucket + 1])


def test_sample_multidimensional_logits() -> None:
    """Verify sampling works with multidimensional logits."""
    bucketizer = build_uniform_bucketizer("bounded")
    batch_size, seq_len = 2, 3

    # different logits for each batch/sequence position
    logits = torch.randn(batch_size, seq_len, bucketizer.num_buckets)

    gen = torch.Generator().manual_seed(42)
    samples = bucketizer.sample(logits, temperature=1.0, generator=gen)

    # verify output shape
    assert samples.shape == (batch_size, seq_len)

    # verify all samples are finite and within bounds
    assert torch.all(torch.isfinite(samples))
    assert torch.all(samples >= bucketizer.borders[0])
    assert torch.all(samples <= bucketizer.borders[-1])

    # verify positions sample independently (should have different values)
    # with random logits, very unlikely all positions give same sample
    unique_samples = len(torch.unique(samples))
    assert unique_samples > 1, "Positions should sample independently"


def test_sample_uniform_within_buckets() -> None:
    """Verify samples are uniformly distributed within a single bucket."""
    bucketizer = build_uniform_bucketizer("bounded")
    # give >99.9% probability to bucket 1
    logits = torch.tensor([-100.0, 10.0, -100.0, -100.0])
    num_samples = 10000

    gen = torch.Generator().manual_seed(42)
    samples = torch.stack(
        [
            bucketizer.sample(logits, temperature=1.0, generator=gen)
            for _ in range(num_samples)
        ]
    )

    # almost all samples should be in bucket 1
    bucket_indices = bucketizer.bucketize(samples)
    bucket1_samples = samples[bucket_indices == 1]
    assert len(bucket1_samples) > 9900, "Most samples should be in bucket 1"

    # samples should be uniformly distributed within the bucket
    # Kolmogorov-Smirnov test against uniform distribution
    left_border = bucketizer.borders[1].item()
    right_border = bucketizer.borders[2].item()

    # normalize to [0, 1]
    normalized = (bucket1_samples - left_border) / (right_border - left_border)

    # KS statistic: max difference between empirical and theoretical CDF
    sorted_norm = torch.sort(normalized)[0]
    n = len(sorted_norm)
    empirical_cdf = torch.arange(1, n + 1, dtype=torch.float32) / n
    theoretical_cdf = sorted_norm  # for uniform on [0, 1]
    ks_stat = torch.max(torch.abs(empirical_cdf - theoretical_cdf))

    # critical value for KS test at p=0.01 is approximately 1.63 / sqrt(n)
    critical_value = 1.63 / math.sqrt(n)
    assert ks_stat < critical_value, f"KS test failed: {ks_stat} > {critical_value}"

    # verify standard deviation matches uniform: width / sqrt(12)
    width = right_border - left_border
    expected_std = width / math.sqrt(12.0)
    actual_std = bucket1_samples.std().item()
    assert 0.9 * expected_std < actual_std < 1.1 * expected_std


def test_sample_zero_mass_bucket_stability() -> None:
    """Verify numerical stability when some buckets have near-zero mass."""
    bucketizer = build_uniform_bucketizer("bounded")
    # buckets 0 and 3 have ~0 mass
    logits = torch.tensor([-100.0, 2.0, 1.0, -100.0])
    num_samples = 1000

    gen = torch.Generator().manual_seed(42)
    samples = torch.stack(
        [
            bucketizer.sample(logits, temperature=1.0, generator=gen)
            for _ in range(num_samples)
        ]
    )

    # verify no NaN or inf
    assert torch.all(torch.isfinite(samples)), (
        "Samples should be finite despite zero-mass buckets"
    )

    # all samples should fall in buckets 1-2 (high-probability buckets)
    bucket_indices = bucketizer.bucketize(samples)
    assert torch.all((bucket_indices == 1) | (bucket_indices == 2))

    # verify samples stay within high-probability bucket bounds
    assert torch.all(samples >= bucketizer.borders[1])
    assert torch.all(samples <= bucketizer.borders[3])

    # verify tail logic doesn't activate accidentally
    # (no samples outside the borders for bounded support)
    assert torch.all(samples >= bucketizer.borders[0])
    assert torch.all(samples <= bucketizer.borders[-1])


def test_sample_riemann_bounded_histogram() -> None:
    """Riemann bucket sampling should respect bucket probabilities."""
    borders = [-3.0, -1.0, -0.2, 1.0, 3.0]
    bucketizer = build_riemann_bucketizer(borders, bucket_support="bounded")
    num_samples = 50000

    logits = torch.zeros(bucketizer.num_buckets)
    logits_batch = logits.repeat(num_samples, 1)
    gen = torch.Generator().manual_seed(123)

    samples = bucketizer.sample(logits_batch, temperature=1.0, generator=gen)

    assert torch.all(samples >= bucketizer.borders[0])
    assert torch.all(samples <= bucketizer.borders[-1])

    bucket_indices = bucketizer.bucketize(samples)
    counts = torch.bincount(bucket_indices.long(), minlength=bucketizer.num_buckets)

    expected_probs = torch.full((bucketizer.num_buckets,), 1.0 / bucketizer.num_buckets)
    chi_squared = torch.sum(
        (counts.float() - num_samples * expected_probs) ** 2
        / (num_samples * expected_probs)
    )
    assert chi_squared < 20.0


def test_sample_riemann_unbounded_tails() -> None:
    """Riemann buckets with unbounded support should sample half-normal tails."""
    borders = [-3.0, -1.5, -0.5, 1.0, 2.5]
    bucketizer = build_riemann_bucketizer(borders, bucket_support="unbounded")
    num_samples = 50000

    logits = torch.tensor([2.0, -0.5, -0.5, 2.0])
    logits_batch = logits.repeat(num_samples, 1)
    gen = torch.Generator().manual_seed(321)

    samples = bucketizer.sample(logits_batch, temperature=1.0, generator=gen)
    assert torch.all(torch.isfinite(samples))

    left_bucket = (samples >= bucketizer.borders[0]) & (samples < bucketizer.borders[1])
    right_bucket = (samples >= bucketizer.borders[-2]) & (
        samples < bucketizer.borders[-1]
    )
    left_tail = samples < bucketizer.borders[0]
    right_tail = samples >= bucketizer.borders[-1]

    total_left = left_bucket.sum() + left_tail.sum()
    total_right = right_bucket.sum() + right_tail.sum()
    assert total_left.item() > 0
    assert total_right.item() > 0

    left_tail_ratio = left_tail.sum().float() / total_left.float()
    right_tail_ratio = right_tail.sum().float() / total_right.float()
    assert 0.45 < left_tail_ratio < 0.55
    assert 0.45 < right_tail_ratio < 0.55

    if left_tail.any():
        distances = bucketizer.borders[0] - samples[left_tail]
        assert torch.all(distances > 0)
    if right_tail.any():
        distances = samples[right_tail] - bucketizer.borders[-1]
        assert torch.all(distances > 0)
