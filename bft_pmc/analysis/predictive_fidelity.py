"""Next-token fidelity of trained BFTs against the exact Bayesian predictive.

This asks a different question from PMC:
not what the model's implicit prior is, but whether the model's next-token
distribution *is* the Bayes-optimal one, position by position, measured in
prediction space. It needs no rollouts -- one forward pass per sequence gives
every position at once -- so it is several orders of magnitude cheaper than
the PMC diagnostics.

Two data laws per family, each compared against its own exact predictive:

  in_distribution  sequences from a task in the training pool, against the
                   memorizing predictive (the Bayes optimum for the training
                   law: a uniform mixture over the pool, reweighted by the
                   observed prefix)
  population       sequences from a fresh task drawn from the population
                   prior, against the generalizing predictive (the exact
                   Dirichlet-multinomial)

Linear regression is not supported: its model emits a categorical over
discretised response buckets and the codebase has no exact predictive in that
space, only point predictors. Adding it means deriving the bucketed Gaussian
posterior predictive, which is not done here.

Alignment between the model's outputs and the exact predictive is the one
place a silent error can hide, so ``--self-test`` checks it with no
checkpoint: with a single-task pool the Bayes-optimal predictive is known in
closed form, an ideal model returning it must score KL = 0, and any
off-by-one in the indexing shows up immediately.

    uv run eval.py predictive-fidelity --self-test
    uv run eval.py predictive-fidelity --family bau --checkpoint-paths ... --seq-len 500
    uv run eval.py predictive-fidelity --family markov --checkpoint-paths ... \
        --config-path outputs/markov/training/<run>/resolved_config.yaml

The paper's numbers (next-token fidelity, in the appendix on sufficient
conditions for PMC) come from ``slurm/predictive_fidelity.slurm``: 256
sequences of 500 tokens per family at M = 32. Each run writes
``fidelity_<family>.csv`` (one row per checkpoint), a per-position CSV and the
settings it ran with. This replaces the balls-and-urns-only script of the
rebuttal period, which wrote ``paper_data/rebuttal_diagnostics/fidelity/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bft_pmc.analysis.checkpoints import load_run_info
from bft_pmc.balls_and_urns.baselines import (
    generalising_predictive,
    memorising_predictive,
)
from bft_pmc.markov.analysis_common import load_trained_markov_artifacts
from bft_pmc.markov.baselines import bi_inf_predictive, bi_ret_predictive
from bft_pmc.metrics import symmetrised_kl
from bft_pmc.paths import OUTPUTS
from pfn_transformerlens.model.PFN import UnsupervisedPFN


def bau_exact(
    tokens: torch.Tensor, law: str, pool: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Exact balls-and-urns next-token predictive after each token.

    ``in_distribution`` uses the memorizing predictive over the urn ``pool``;
    ``population`` the Dirichlet(``alpha``)-multinomial predictive.
    """
    return (
        memorising_predictive(tokens, pool)
        if law == "in_distribution"
        else generalising_predictive(tokens, alpha)
    )


def markov_exact(
    tokens: torch.Tensor, law: str, pool: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Exact 1-Markov next-state predictive after each state.

    ``in_distribution`` uses the memorizing predictive over the transition
    matrices in ``pool``; ``population`` the row-wise Dirichlet(``alpha``)
    predictive. Entry t predicts state t + 1.
    """
    return (
        bi_ret_predictive(tokens, pool)
        if law == "in_distribution"
        else bi_inf_predictive(tokens, alpha)
    )


def scored_positions(exact: torch.Tensor, model_probs: torch.Tensor) -> int:
    """The number of positions both predictives define, after alignment."""
    return min(exact.shape[1], model_probs.shape[1])


def fidelity(
    exact: torch.Tensor, model_probs: torch.Tensor
) -> tuple[float, np.ndarray]:
    """Symmetrised KL, overall and per position, on the common positions."""
    n = scored_positions(exact, model_probs)
    exact, model_probs = exact[:, :n], model_probs[:, :n]
    per_position = np.array(
        [
            symmetrised_kl(exact[:, t : t + 1], model_probs[:, t : t + 1])
            for t in range(n)
        ]
    )
    return float(symmetrised_kl(exact, model_probs)), per_position


def self_test() -> None:
    """With a one-task pool the memorizing predictive is that task itself.

    An ideal model returns it exactly, so the fidelity must be zero. This
    pins the alignment: shifting either series by one position makes the KL
    jump to a clearly non-zero value, which the test asserts.
    """
    torch.manual_seed(0)
    k, batch, length = 5, 8, 40

    # --- balls and urns. The pool needs several urns: with one, the
    # memorizing predictive is that urn at every position, so a shifted
    # comparison would also score zero and the test would pass vacuously.
    pool = torch.distributions.Dirichlet(torch.ones(k)).sample((4,))
    tokens = torch.multinomial(
        pool[torch.randint(4, (batch,))], length, replacement=True
    )
    exact = bau_exact(tokens, "in_distribution", pool, torch.ones(k))
    value, _ = fidelity(exact, exact.clone())
    assert value < 1e-6, f"BAU alignment: expected 0, got {value}"
    shifted, _ = fidelity(exact[:, 1:], exact[:, :-1].clone())
    print(f"  BAU     aligned KL = {value:.3e}   shifted-by-one KL = {shifted:.3e}")
    assert shifted > 1e-3, "BAU test is not position-sensitive"

    # --- 1-Markov: a single transition matrix; the optimum is its own row
    matrix = torch.distributions.Dirichlet(torch.ones(k)).sample((k,)).unsqueeze(0)
    chain = torch.zeros(batch, length, dtype=torch.long)
    for t in range(1, length):
        chain[:, t] = torch.multinomial(matrix[0][chain[:, t - 1]], 1).squeeze(-1)
    exact = markov_exact(chain, "in_distribution", matrix, torch.ones(k, k))
    # bi_*_predictive indexes by the *current* state: entry t predicts t+1.
    ideal = matrix[0][chain[:, :-1]]
    value, _ = fidelity(exact, ideal)
    assert value < 1e-6, f"Markov alignment: expected 0, got {value}"
    shifted, _ = fidelity(exact[:, 1:], ideal[:, :-1])
    print(f"  Markov  aligned KL = {value:.3e}   shifted-by-one KL = {shifted:.3e}")
    assert shifted > 1e-3, "shifting should break the match; the test is not sensitive"
    print("self-test passed: indexing is aligned for both families")


def run_bau(path: str, args: argparse.Namespace) -> dict[str, tuple[float, np.ndarray]]:
    """Mean and per-position symmetrised KL of one balls-and-urns checkpoint."""
    info = load_run_info(
        Path(path), device=args.device, n_ctx_override=args.n_ctx_override
    )
    model = info["model"]
    assert isinstance(model, UnsupervisedPFN)
    device = next(model.parameters()).device
    pool = torch.as_tensor(info["tasks"], dtype=torch.float32)
    vocab = int(pool.shape[1])
    alpha = torch.ones(vocab) * args.alpha_value

    laws = {
        "in_distribution": pool[torch.randint(pool.shape[0], (args.n_sequences,))],
        "population": torch.distributions.Dirichlet(alpha).sample((args.n_sequences,)),
    }
    out = {}
    for law, urns in laws.items():
        tokens = torch.multinomial(urns, args.seq_len, replacement=True)
        totals, per_position, batches = 0.0, None, 0
        with torch.no_grad():
            for start in range(0, args.n_sequences, args.batch_size):
                chunk = tokens[start : start + args.batch_size]
                bos = torch.full((chunk.shape[0], 1), vocab, dtype=chunk.dtype)
                prediction = model.predict_on_prompt(
                    torch.cat([bos, chunk], dim=1).to(
                        device=device, dtype=torch.float32
                    )
                )
                if isinstance(prediction, tuple):
                    prediction = prediction[0]
                # Output i is the prediction after reading input i; dropping the
                # last one leaves one prediction per data token.
                value, positions = fidelity(
                    bau_exact(chunk, law, pool, alpha), prediction.probs.cpu()[:, :-1]
                )
                totals += value
                per_position = (
                    positions if per_position is None else per_position + positions
                )
                batches += 1
        out[law] = (totals / batches, per_position / batches)
    return out


def run_markov(
    path: str, args: argparse.Namespace
) -> dict[str, tuple[float, np.ndarray]]:
    """Mean and per-position symmetrised KL of one 1-Markov checkpoint."""
    artifacts = load_trained_markov_artifacts(
        args.config_path,
        Path(path),
        seq_len_override=args.seq_len_override,
        rope_scaling=args.rope_scaling,
    )
    model, dataset = artifacts.model, artifacts.dataset
    device = artifacts.device
    pool = artifacts.transition_matrices.detach().cpu().float()
    k = int(dataset.k)
    alpha = torch.ones(k, k) * args.alpha_value

    fresh = torch.distributions.Dirichlet(torch.ones(k)).sample((args.n_sequences, k))
    laws = {
        "in_distribution": pool[torch.randint(pool.shape[0], (args.n_sequences,))],
        "population": fresh,
    }
    out = {}
    for law, matrices in laws.items():
        stationary = dataset._compute_stationary_batch(matrices.to(device))
        chains = dataset._generate_chains(
            matrices.to(device), stationary, args.seq_len
        ).cpu()
        reference = pool if law == "in_distribution" else alpha
        totals, per_position, batches = 0.0, None, 0
        with torch.no_grad():
            for start in range(0, args.n_sequences, args.batch_size):
                chunk = chains[start : start + args.batch_size]
                logits = model(dataset.prepend_bos(chunk.to(device)))
                # Input is [BOS, y_1..y_S]; output at index t+1 is the
                # prediction after reading y_{t+1}, which is what
                # bi_*_predictive indexes at t.
                probs = torch.softmax(logits[:, 1:, :k], dim=-1).cpu()
                value, positions = fidelity(
                    markov_exact(chunk, law, reference, alpha), probs
                )
                totals += value
                per_position = (
                    positions if per_position is None else per_position + positions
                )
                batches += 1
        out[law] = (totals / batches, per_position / batches)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="check the indexing with no checkpoint, then exit",
    )
    parser.add_argument(
        "--family",
        choices=("bau", "markov"),
        help="task family to score: bau or markov",
    )
    parser.add_argument(
        "--checkpoint-paths", nargs="+", default=[], help="checkpoints to score"
    )
    parser.add_argument(
        "--config-path", default=None, help="markov: the run's resolved_config.yaml"
    )
    parser.add_argument("--seq-len", type=int, default=500, help="data tokens scored")
    parser.add_argument(
        "--n-sequences", type=int, default=256, help="sequences per data law"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="sequences per forward batch"
    )
    parser.add_argument(
        "--alpha-value",
        type=float,
        default=1.0,
        help="concentration of the symmetric Dirichlet population prior",
    )
    parser.add_argument(
        "--n-ctx-override",
        type=int,
        default=None,
        help="bau: run the model past its trained context",
    )
    parser.add_argument(
        "--seq-len-override",
        type=int,
        default=None,
        help="markov: run the model past its trained context",
    )
    parser.add_argument(
        "--rope-scaling",
        default="interpolate",
        help="markov: how rotary positions are extended (interpolate or extrapolate)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for the tasks and sequences"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS / "predictive_fidelity",
        help="directory for the CSVs and the settings file",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return
    if not args.family or not args.checkpoint_paths:
        raise SystemExit("--family and --checkpoint-paths are required")

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = run_bau if args.family == "bau" else run_markov

    rows, columns = [], {}
    for path in args.checkpoint_paths:
        results = runner(path, args)
        label = (
            Path(path).parent.name
            if Path(path).stem.startswith("checkpoint_step")
            else Path(path).stem
        )
        row = {
            "checkpoint": label,
            "path": path,
            "family": args.family,
            "seq_len": args.seq_len,
            "n_sequences": args.n_sequences,
        }
        for law, (mean_kl, per_position) in results.items():
            row[f"sym_kl_{law}"] = mean_kl
            columns[f"{label}_{law}"] = per_position
            print(f"{label} [{law}]: symmetrised KL = {mean_kl:.6f}")
        rows.append(row)

    keys = list(rows[0])
    with (args.output_dir / f"fidelity_{args.family}.csv").open("w") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    length = min(len(v) for v in columns.values())
    with (args.output_dir / f"fidelity_{args.family}_per_position.csv").open(
        "w"
    ) as handle:
        handle.write("position," + ",".join(columns) + "\n")
        for t in range(length):
            handle.write(
                f"{t}," + ",".join(f"{columns[c][t]:.8g}" for c in columns) + "\n"
            )
    (args.output_dir / f"fidelity_{args.family}_settings.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    print(f"written to {args.output_dir}")


if __name__ == "__main__":
    main()
