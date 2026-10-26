"""Can PMC tell a Gaussian prior from a variance-matched Laplace prior?

The rebuttal to Reviewer 9Er4 claims a capability gap. In linear regression the
prediction-space comparison depends on a prior only through its mean, so a
zero-mean Gaussian and a zero-mean Laplace give *identical* prior point
predictions and Delta_MSE cannot separate them at all. The latent comparison
comes from samples, so it can in principle see the difference in shape.

This script measures whether it does, at the budget we actually have. It reads
the saved prior rollouts (no checkpoint, no generation) and, for each task
diversity:

  1. recomputes the PMC prior samples w_hat by least squares on each rollout;
  2. draws a Gaussian reference N(0, I) -- the training prior Pi_infinity --
     and a Laplace reference with the SAME mean and variance (b = 1/sqrt(2),
     since Var[Laplace(0, b)] = 2 b^2). Matching the variance is what makes
     this a test of shape rather than of scale;
  3. reports energy distance and sliced Wasserstein-1 to each reference, with
     a bootstrap spread, and the paired difference (laplace - gaussian) which
     is positive when the Gaussian is the closer reference;
  4. calibrates against an oracle: a fresh Gaussian sample of the same size,
     which is the best any method could do here. If the oracle separation is
     itself small, the budget is the limit, not PMC.

Run from anywhere:
    python3 scripts/prior_shape_discrimination.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from metrics import energy_distance_multidim, sliced_wasserstein
from plotting.paper_style import apply_paper_style

ROOT = REPO / "outputs/pmc_diagnostics"
# Var[Laplace(0, b)] = 2 b^2, so this matches the unit-variance Gaussian.
LAPLACE_SCALE = 1.0 / np.sqrt(2.0)


def pmc_prior_samples(run: Path, length: int) -> np.ndarray:
    """Least-squares weights from the first `length` pairs of each prior rollout."""
    with np.load(run / "prior_trajectories.npz") as bundle:
        x = bundle["rollout_x"][:, :length, :]
        y = bundle["rollout_y"][:, :length]
    gram = np.einsum("rni,rnj->rij", x, x)
    rhs = np.einsum("rni,rn->ri", x, y)
    # pinv, matching analysis/path_stability.py: a short prefix can leave the
    # Gram matrix singular, and lstsq would raise rather than fall back.
    return np.einsum("rij,rj->ri", np.linalg.pinv(gram), rhs)


def references(rng: np.random.Generator, n: int, dim: int) -> dict[str, np.ndarray]:
    return {
        "gaussian": rng.standard_normal((n, dim)),
        "laplace": rng.laplace(0.0, LAPLACE_SCALE, (n, dim)),
    }


def distances(samples: np.ndarray, refs: dict[str, np.ndarray], max_ed: int) -> dict[str, float]:
    """Distances to both references.

    Energy distance is a U-statistic over all pairs, so it costs O(n^2) time
    and memory: 8192 samples is ~67M pairwise distances and several hundred MB
    per call. It is therefore capped at `max_ed` samples, while sliced
    Wasserstein, which only sorts projections, always uses the full set.
    """
    out = {}
    for name, ref in refs.items():
        a, b = samples[:max_ed], ref[:max_ed]
        out[f"ed_{name}"] = energy_distance_multidim(a, b)
        out[f"sw_{name}"] = sliced_wasserstein(samples, ref)
    for metric in ("ed", "sw"):
        # Positive => the Gaussian is the closer reference, which is the claim.
        out[f"{metric}_margin"] = out[f"{metric}_laplace"] - out[f"{metric}_gaussian"]
    return out


def evaluate(samples: np.ndarray, seed: int, n_blocks: int, max_ed: int) -> dict[str, float]:
    """Full-sample margins, with a standard error from disjoint blocks.

    A bootstrap would recompute an O(n^2) statistic hundreds of times. Instead
    the point estimate uses every sample once, and the error comes from
    splitting the samples into `n_blocks` disjoint blocks, each scored against
    its own fresh references. Block margins are independent, so the standard
    error of their mean estimates the standard error of the full-sample margin:
    both scale as 1/n.
    """
    rng = np.random.default_rng(seed)
    n, dim = samples.shape
    out = dict(distances(samples, references(rng, n, dim), max_ed))

    # Each metric's blocks must partition exactly the samples its own point
    # estimate used. Energy distance is capped at `max_ed` while sliced
    # Wasserstein uses all of them, so blocking both over the full set would
    # give energy distance a standard error describing an estimator at a
    # larger sample size than the one reported, understating it by roughly
    # sqrt(n / max_ed).
    for metric, used in (("ed", min(n, max_ed)), ("sw", n)):
        margins: list[float] = []
        for block in np.array_split(rng.permutation(used), n_blocks):
            subset, refs = samples[block], references(rng, len(block), dim)
            if metric == "ed":
                margin = (energy_distance_multidim(subset, refs["laplace"])
                          - energy_distance_multidim(subset, refs["gaussian"]))
            else:
                margin = (sliced_wasserstein(subset, refs["laplace"])
                          - sliced_wasserstein(subset, refs["gaussian"]))
            margins.append(margin)
        out[f"{metric}_margin_se"] = float(np.std(margins, ddof=1) / np.sqrt(len(margins)))
    return out


def prediction_space_gap(rng: np.random.Generator, n: int, dim: int) -> dict[str, float]:
    """The prediction-space comparison cannot separate the two priors.

    With an empty prompt the Bayes point prediction of the response is
    E[w]^T x, and both priors have E[w] = 0, so the prediction is 0 for every
    query and the population gap is exactly zero. What a finite sample shows is
    Monte Carlo error, reported here for scale.
    """
    refs = references(rng, n, dim)
    queries = rng.standard_normal((512, dim))
    preds = {k: queries @ v.mean(axis=0) for k, v in refs.items()}
    return {
        "population_gap": 0.0,
        "sampled_rms_gap": float(np.sqrt(np.mean((preds["gaussian"] - preds["laplace"]) ** 2))),
        "sampled_rms_prediction": float(np.sqrt(np.mean(preds["gaussian"] ** 2))),
    }


def figure(rows: list[dict], marginals: dict[int, np.ndarray], destination: Path, dim: int) -> Path:
    apply_paper_style(5.5)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))
    fig.subplots_adjust(left=.11, right=.985, bottom=.21, top=.88, wspace=.30)

    # (a) pooled one-dimensional marginal: the prior is isotropic, so pooling
    # the coordinates gives dim times the samples and resolves the tails.
    ax = axes[0]
    grid = np.linspace(-4, 4, 400)
    ax.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), color="#1f77b4", lw=1.2, label="Gaussian")
    ax.plot(grid, np.exp(-np.abs(grid) / LAPLACE_SCALE) / (2 * LAPLACE_SCALE),
            color="#d62728", lw=1.2, ls="--", label="Laplace")
    largest = max(marginals)
    pooled = marginals[largest].ravel()
    ax.hist(pooled, bins=80, range=(-4, 4), density=True, histtype="step",
            color="black", lw=.9, label=f"PMC prior ($M={largest}$)")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("Coefficient value")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=6, loc="lower center")
    ax.set_title("(a) Pooled marginal", fontsize=7)

    # (b) paired margin with bootstrap spread: above zero means the Gaussian
    # reference is closer, which is what the claim requires.
    ax = axes[1]
    xs = np.arange(len(rows))
    for offset, metric, color in ((-.12, "ed", "#1f77b4"), (.12, "sw", "#2ca02c")):
        # Normalise by the oracle margin, not by the distance to the Gaussian:
        # energy distance is an unbiased U-statistic and goes negative once the
        # samples match the reference, which would flip the error bars. The
        # oracle margin is positive by construction and gives both metrics a
        # common scale, where 1.0 is the separation a true Gaussian sample of
        # this size achieves.
        denom = [max(abs(r[f"{metric}_margin_oracle"]), 1e-12) for r in rows]
        mean = [r[f"{metric}_margin"] / d for r, d in zip(rows, denom)]
        err = [r[f"{metric}_margin_se"] / d for r, d in zip(rows, denom)]
        ax.errorbar(xs + offset, mean, yerr=err, fmt="o", ms=3, lw=1,
                    capsize=2, color=color,
                    label="Energy distance" if metric == "ed" else "Sliced Wasserstein-1")
    ax.axhline(0, color=".65", lw=.6)
    # Bare numbers: the sweep can run to M=65536, where "$M=65536$" per tick
    # would collide.
    ax.set_xticks(xs, [str(r["task_diversity"]) for r in rows], fontsize=6)
    ax.set_xlabel("Task diversity $M$")
    ax.set_ylabel("Margin / oracle margin\n(Laplace $-$ Gaussian)", linespacing=2.0)
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("(b) Which reference is closer", fontsize=7)
    for side in ("top", "right"):
        for a in axes:
            a.spines[side].set_visible(False)
            a.grid(alpha=.14, lw=.5)

    out = destination / "prior_shape_discrimination.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Creator": "prior_shape_discrimination.py"})
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-diversities", type=int, nargs="+", default=[32, 128, 256])
    parser.add_argument("--run-template", default="lr_m{m}_none_n500",
                        help="run directory under outputs/pmc_diagnostics; {m} is the task diversity")
    parser.add_argument("--length", type=int, default=500, help="rollout length used for w_hat")
    parser.add_argument("--n-blocks", type=int, default=8,
                        help="disjoint blocks used for the standard error")
    parser.add_argument("--max-ed-samples", type=int, default=4096,
                        help="cap on the energy-distance sample size; it costs O(n^2)")
    parser.add_argument("--match-variance", action="store_true",
                        help="rescale the PMC samples to unit variance first, so the "
                             "comparison isolates shape rather than scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/prior_shape")
    args = parser.parse_args()

    rows: list[dict] = []
    marginals: dict[int, np.ndarray] = {}
    inputs: list[dict] = []
    dim = 0

    for m in args.task_diversities:
        run = ROOT / args.run_template.format(m=m)
        path = run / "prior_trajectories.npz"
        if not path.exists():
            print(f"skipping M={m}: {path} not found")
            continue
        samples = pmc_prior_samples(run, args.length)
        raw_variance = float(samples.var())
        if args.match_variance:
            # Both references are unit-variance by construction, so a PMC
            # sample whose variance differs is penalised for scale as well as
            # shape. Rescaling globally preserves shape and removes that.
            samples = samples / np.sqrt(raw_variance)
        marginals[m] = samples
        dim = samples.shape[1]
        inputs.append({"path": str(path.relative_to(REPO)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

        stats = evaluate(samples, args.seed + m, args.n_blocks, args.max_ed_samples)
        # Oracle: a fresh Gaussian sample is the best case at this budget.
        oracle_rng = np.random.default_rng(args.seed + 10_000 + m)
        oracle = evaluate(oracle_rng.standard_normal(samples.shape), args.seed + m,
                          args.n_blocks, args.max_ed_samples)

        row = {"task_diversity": m, "n_samples": int(samples.shape[0]), "length": args.length,
               "sample_variance": raw_variance, "variance_matched": bool(args.match_variance)}
        row.update(stats)
        for metric in ("ed", "sw"):
            row[f"{metric}_margin_oracle"] = oracle[f"{metric}_margin"]
            row[f"{metric}_margin_oracle_se"] = oracle[f"{metric}_margin_se"]
            # How many standard errors the margin sits above zero.
            se = row[f"{metric}_margin_se"]
            row[f"{metric}_margin_z"] = row[f"{metric}_margin"] / se if se > 0 else float("nan")
            oracle_se = row[f"{metric}_margin_oracle_se"]
            row[f"{metric}_margin_oracle_z"] = (
                row[f"{metric}_margin_oracle"] / oracle_se if oracle_se > 0 else float("nan"))
        rows.append(row)

    if not rows:
        raise SystemExit("no prior rollouts found; run the LR diagnostics first")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with (args.output_dir / "prior_shape_discrimination.csv").open("w") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    control = prediction_space_gap(np.random.default_rng(args.seed), rows[0]["n_samples"], dim)
    figure_path = figure(rows, marginals, args.output_dir, dim)
    (args.output_dir / "prior_shape_discrimination.json").write_text(json.dumps({
        "generator": "scripts/prior_shape_discrimination.py",
        "laplace_scale": LAPLACE_SCALE,
        "prediction_space_control": control,
        "rows": rows,
        "inputs": inputs,
    }, indent=2) + "\n")

    print(f"{'M':>6} {'ED margin':>24} {'SW margin':>24}   (margin > 0 favours Gaussian)")
    for row in rows:
        ed = f"{row['ed_margin']:+.4g} ± {row['ed_margin_se']:.2g} ({row['ed_margin_z']:+.1f}σ)"
        sw = f"{row['sw_margin']:+.4g} ± {row['sw_margin_se']:.2g} ({row['sw_margin_z']:+.1f}σ)"
        print(f"{row['task_diversity']:>6} {ed:>24} {sw:>24}")
    print()
    print("oracle (a true Gaussian sample of the same size; the best case here):")
    for row in rows:
        print(f"{row['task_diversity']:>6}"
              f" ED {row['ed_margin_oracle']:+.4g} ± {row['ed_margin_oracle_se']:.2g}"
              f" ({row['ed_margin_oracle_z']:+.1f}σ)"
              f"   SW {row['sw_margin_oracle']:+.4g} ± {row['sw_margin_oracle_se']:.2g}"
              f" ({row['sw_margin_oracle_z']:+.1f}σ)")
    print()
    print(f"prediction space: population gap {control['population_gap']:.0f}, "
          f"sampled RMS gap {control['sampled_rms_gap']:.4g} against RMS prediction "
          f"{control['sampled_rms_prediction']:.4g}")
    print(f"\nwrote {figure_path} and its CSV/JSON in {args.output_dir}")


if __name__ == "__main__":
    main()
