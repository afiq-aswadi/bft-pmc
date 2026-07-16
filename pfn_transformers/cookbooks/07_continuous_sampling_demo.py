"""
Cookbook 7: Continuous Sampling Demo

Illustrates how the bucketizer draws continuous samples for bounded vs unbounded
support, and how sampling differs from simply decoding bucket midpoints.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from pfn_transformerlens.model.bucketizer import Bucketizer, BucketSupport


def make_bucketizer(bucket_support: BucketSupport) -> Bucketizer:
    return Bucketizer(
        bucket_type="uniform",
        bucket_support=bucket_support,
        d_vocab=6,
        y_min=-3.0,
        y_max=3.0,
    )


def sample_distribution(
    bucketizer: Bucketizer,
    logits: torch.Tensor,
    *,
    num_samples: int,
    temperature: float,
    seed: int,
) -> torch.Tensor:
    logits_batch = logits.repeat(num_samples, 1)
    generator = torch.Generator().manual_seed(seed)
    samples = bucketizer.sample(
        logits_batch,
        temperature=temperature,
        generator=generator,
    )
    return samples.cpu()


def decode_representatives(
    bucketizer: Bucketizer, logits: torch.Tensor
) -> torch.Tensor:
    """Get representative value for most likely bucket.

    For bounded support and interior buckets, returns midpoint.
    For unbounded edge buckets, returns mode (at inner boundary).
    """
    bucket_indices = torch.argmax(logits, dim=-1)
    return bucketizer.decode(bucket_indices).cpu()


def plot_kde(
    ax: plt.Axes,
    samples: torch.Tensor,
    borders: torch.Tensor,
    *,
    title: str,
) -> None:
    """Plot kernel density estimate of samples with bucket borders."""
    samples_np = samples.numpy()

    # plot KDE using seaborn
    sns.kdeplot(
        samples_np,
        ax=ax,
        fill=True,
        color="steelblue",
        alpha=0.3,
        linewidth=2,
    )

    # plot bucket borders
    for border in borders.numpy():
        ax.axvline(border, color="black", linewidth=0.6, linestyle="--", alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("sampled value")
    ax.set_ylabel("density")


def describe_distribution(
    label: str,
    samples: torch.Tensor,
    borders: torch.Tensor,
) -> None:
    tail_left = (samples < borders[0]).sum().item()
    tail_right = (samples > borders[-1]).sum().item()
    fraction_left = tail_left / samples.numel()
    fraction_right = tail_right / samples.numel()
    print(f"{label}: mean={samples.mean():.3f}, std={samples.std():.3f}")
    print(
        f"{label}: tail mass left={fraction_left:.3%}, right={fraction_right:.3%}",
    )


def main() -> None:
    output_dir = Path(__file__).parent / "outputs" / "07_sampling"
    output_dir.mkdir(parents=True, exist_ok=True)

    # logits = torch.tensor([1.5, 0.2, -0.5, -0.5, 0.2, 1.5])
    logits = torch.ones(6)

    bounded = make_bucketizer("bounded")
    bounded_samples = sample_distribution(
        bounded,
        logits,
        num_samples=50_000,
        temperature=1.0,
        seed=0,
    )

    unbounded = make_bucketizer("unbounded")
    unbounded_samples = sample_distribution(
        unbounded,
        logits,
        num_samples=50_000,
        temperature=1.0,
        seed=1,
    )

    describe_distribution("bounded", bounded_samples, bounded.borders)
    describe_distribution("unbounded", unbounded_samples, unbounded.borders)

    mean_decode = decode_representatives(bounded, logits.unsqueeze(0))
    print("decoded representative (most likely bucket):", mean_decode.item())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_kde(
        axes[0],
        bounded_samples,
        bounded.borders.cpu(),
        title="Bounded support (uniform buckets)",
    )
    plot_kde(
        axes[1],
        unbounded_samples,
        unbounded.borders.cpu(),
        title="Unbounded support with half-normal tails",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "bucketizer_sampling.png", dpi=200)
    plt.close(fig)

    print(f"Saved KDE plot to {output_dir / 'bucketizer_sampling.png'}")


if __name__ == "__main__":
    main()
