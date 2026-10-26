"""Command registry behind the repository's ``train.py`` and ``eval.py``.

Every experiment, diagnostic and figure generator in :mod:`bft_pmc` is a
module with its own command-line interface. The two root scripts dispatch to
them by name, so a run never depends on where a module sits in the package::

    uv run train.py lr --num-tasks 32 ...        # -> bft_pmc.linear_regression.train
    uv run eval.py plot-lr-sweep --help          # -> bft_pmc.linear_regression.plot_sweep_combined

Arguments after the command name are passed to the module unchanged; put
``--help`` after a command to see its own options.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """One named entry point: the module it runs and a one-line summary."""

    name: str
    module: str
    summary: str


@dataclass(frozen=True)
class Section:
    """A titled group of commands in the help listing."""

    title: str
    commands: tuple[Command, ...]


TRAIN_SECTIONS: tuple[Section, ...] = (
    Section(
        "Task families",
        (
            Command(
                "lr",
                "bft_pmc.linear_regression.train",
                "Linear regression on a finite Gaussian task pool.",
            ),
            Command(
                "bau",
                "bft_pmc.balls_and_urns.train",
                "Balls and urns on a finite Dirichlet task pool.",
            ),
            Command(
                "markov",
                "bft_pmc.markov.train",
                "1-Markov chains on a finite pool of transition matrices.",
            ),
            Command(
                "beta-bernoulli",
                "bft_pmc.balls_and_urns.beta_bernoulli",
                "Continuous-prior Beta-Bernoulli model, then PMC (Figure 1).",
            ),
        ),
    ),
    Section(
        "Sweeps",
        (
            Command(
                "sweep",
                "bft_pmc.sweep_runner",
                "Run a YAML-defined training sweep across local GPUs.",
            ),
        ),
    ),
)

EVAL_SECTIONS: tuple[Section, ...] = (
    Section(
        "Evaluate checkpoints (metrics and PMC sample bundles)",
        (
            Command(
                "lr-sweep",
                "bft_pmc.linear_regression.sweep_analysis",
                "Final linear-regression checkpoints across task diversity.",
            ),
            Command(
                "bau-sweep",
                "bft_pmc.balls_and_urns.sweep_analysis",
                "Final balls-and-urns checkpoints across task diversity.",
            ),
            Command(
                "markov-sweep",
                "bft_pmc.markov.sweep_analysis",
                "Markov checkpoints across task diversity and training time.",
            ),
            Command(
                "lr-dynamics",
                "bft_pmc.linear_regression.distribution_dynamics",
                "Every saved checkpoint of one linear-regression run.",
            ),
            Command(
                "bau-dynamics",
                "bft_pmc.balls_and_urns.distribution_dynamics",
                "Every saved checkpoint of one balls-and-urns run.",
            ),
            Command(
                "markov-pmc",
                "bft_pmc.markov.run_pmc",
                "Standalone PMC artifacts for one Markov checkpoint.",
            ),
            Command(
                "markov-threshold",
                "bft_pmc.markov.task_diversity_threshold",
                "Markov task-diversity threshold experiment.",
            ),
        ),
    ),
    Section(
        "Prior-only metrics and evaluation data",
        (
            Command(
                "lr-prior-delta-mse",
                "bft_pmc.linear_regression.prior_delta_mse",
                "Zero-context Delta-MSE sidecar for the LR prior panels.",
            ),
            Command(
                "bau-prior-kl",
                "bft_pmc.balls_and_urns.prior_predictive_kl",
                "Prior predictive KL sidecar for the balls-and-urns prior panels.",
            ),
            Command(
                "markov-prior-kl",
                "bft_pmc.markov.prior_predictive_kl",
                "First-step predictive KL sidecar for the Markov prior panels.",
            ),
            Command(
                "bau-eval-dataset",
                "bft_pmc.balls_and_urns.generate_eval_dataset",
                "Fixed balls-and-urns evaluation prompts for a checkpoint sweep.",
            ),
        ),
    ),
    Section(
        "PMC reliability diagnostics (Appendix D)",
        (
            Command(
                "lr-rollout-diagnostics",
                "bft_pmc.linear_regression.rollout_diagnostics",
                "Recovery and drift vs rollout length and count, one LR checkpoint.",
            ),
            Command(
                "bau-rollout-diagnostics",
                "bft_pmc.balls_and_urns.rollout_diagnostics",
                "The same for one balls-and-urns checkpoint.",
            ),
            Command(
                "markov-rollout-diagnostics",
                "bft_pmc.markov.rollout_diagnostics",
                "The same for one Markov checkpoint.",
            ),
            Command(
                "path-stability",
                "bft_pmc.analysis.path_stability",
                "Scaled-L1 path stability of saved rollouts (jellyfish plots).",
            ),
            Command(
                "recovery-vs-dimension",
                "bft_pmc.analysis.dimension_sweep",
                "Aggregate diagnostics runs by latent dimension or task diversity.",
            ),
            Command(
                "replot-pmc-diagnostics",
                "bft_pmc.analysis.replot_pmc_diagnostics",
                "Redraw diagnostics figures from the data their runs saved.",
            ),
            Command(
                "predictive-fidelity",
                "bft_pmc.analysis.predictive_fidelity",
                "Next-token fidelity against the exact predictive (BAU, Markov).",
            ),
            Command(
                "prior-shape",
                "bft_pmc.linear_regression.prior_shape_discrimination",
                "Gaussian vs variance-matched Laplace prior from saved PMC rollouts.",
            ),
            Command(
                "bau-samples-count-panel",
                "bft_pmc.balls_and_urns.samples_count_panel",
                "Recovery vs rollout count from a saved balls-and-urns sample bundle.",
            ),
            Command(
                "compare-pos-encodings",
                "bft_pmc.analysis.pos_encoding_comparison",
                "Compare the learned, rotary and no-encoding sweeps.",
            ),
        ),
    ),
    Section(
        "Paper figures (from saved metrics or sample bundles)",
        (
            Command(
                "plot-lr-sweep",
                "bft_pmc.linear_regression.plot_sweep_combined",
                "LR task-diversity sweep.",
            ),
            Command(
                "plot-lr-dynamics",
                "bft_pmc.linear_regression.plot_dynamics_combined",
                "LR training dynamics.",
            ),
            Command(
                "plot-lr-prior",
                "bft_pmc.linear_regression.plot_sweep_prior",
                "LR prior-only sweep (Delta-MSE, ED, SW).",
            ),
            Command(
                "plot-lr-sweep-and-dynamics",
                "bft_pmc.linear_regression.plot_stitched_sweep_dynamics",
                "LR sweep and dynamics side by side.",
            ),
            Command(
                "plot-lr-marginals",
                "bft_pmc.linear_regression.plot_marginals",
                "LR marginal grids, one per sample bundle.",
            ),
            Command(
                "plot-lr-stitched-marginals",
                "bft_pmc.linear_regression.plot_stitched_marginals",
                "LR marginals across every task diversity.",
            ),
            Command(
                "plot-lr-single-marginal",
                "bft_pmc.linear_regression.plot_single_marginal",
                "One publication-sized LR marginal panel.",
            ),
            Command(
                "plot-identifiability",
                "bft_pmc.linear_regression.plot_intro_identifiability",
                "Gaussian vs spike-and-slab identifiability example (Figure 2).",
            ),
            Command(
                "plot-bau-sweep",
                "bft_pmc.balls_and_urns.plot_sweep_combined",
                "Balls-and-urns task-diversity sweep.",
            ),
            Command(
                "plot-bau-dynamics",
                "bft_pmc.balls_and_urns.plot_dynamics_combined",
                "Balls-and-urns training dynamics.",
            ),
            Command(
                "plot-bau-prior",
                "bft_pmc.balls_and_urns.plot_sweep_prior",
                "Balls-and-urns prior-only sweep (KL, ED, SW).",
            ),
            Command(
                "plot-bau-marginals",
                "bft_pmc.balls_and_urns.plot_marginals",
                "Balls-and-urns marginal grids, one per sample bundle.",
            ),
            Command(
                "plot-bau-stitched-marginals",
                "bft_pmc.balls_and_urns.plot_stitched_marginals",
                "Balls-and-urns marginals across every task diversity.",
            ),
            Command(
                "plot-markov-sweep",
                "bft_pmc.markov.plot_sweep_combined",
                "Markov task-diversity sweep.",
            ),
            Command(
                "plot-markov-dynamics",
                "bft_pmc.markov.plot_dynamics_combined",
                "Markov training dynamics.",
            ),
            Command(
                "plot-markov-prior",
                "bft_pmc.markov.plot_sweep_prior",
                "Markov prior-only sweep (KL, ED, SW).",
            ),
            Command(
                "plot-markov-marginals",
                "bft_pmc.markov.plot_matrix_marginals",
                "Transition-matrix marginal grids.",
            ),
            Command(
                "plot-reliability",
                "bft_pmc.analysis.plot_reliability_appendix",
                "Every reliability figure of Appendix D and the supplement.",
            ),
        ),
    ),
)

TRAIN_EXAMPLES = """\
  uv run train.py lr --num-tasks 32 --num-steps 10 --batch-size 8 --seq-len 16 --n-ctx 32 --save-every 5 --no-use-wandb
  uv run train.py bau --num-tasks 32 --num-steps 10 --batch-size 8 --seq-len 16 --save-every 5 --no-use-wandb
  uv run train.py markov --config-path configs/markov/train.yaml --n-chains 32 --max-steps 10 --no-use-wandb
  uv run train.py beta-bernoulli --replot-from paper_data/beta_bernoulli/pmc_samples.npz
  uv run train.py sweep configs/sweeps/lr_task_diversity_sweep.yaml --dry-run"""

EVAL_EXAMPLES = """\
  uv run eval.py plot-lr-sweep --metrics-csv paper_data/lr/sweep/metrics.csv --prompt-length 8
  uv run eval.py lr-sweep --checkpoint-root checkpoints/lr/task_diversity_none
  uv run eval.py bau-rollout-diagnostics --checkpoint-path checkpoints/bau/task_diversity_none/M32/checkpoint_step_100000.pt
  uv run eval.py path-stability --inputs lr=outputs/pmc_diagnostics/lr_m32_none --realisations 20
  uv run eval.py replot-pmc-diagnostics --roots outputs/pmc_diagnostics"""


def all_commands(sections: Sequence[Section]) -> dict[str, Command]:
    """Commands of every section, keyed by name."""
    return {
        command.name: command for section in sections for command in section.commands
    }


def format_help(
    program: str, kind: str, sections: Sequence[Section], examples: str
) -> str:
    """The usage text printed by ``<program> --help``."""
    width = max(len(name) for name in all_commands(sections)) + 2
    lines = [f"Usage: uv run {program} <{kind}> [{kind} options...]", ""]
    for section in sections:
        lines.append(f"{section.title}:")
        lines.extend(f"  {c.name:<{width}}{c.summary}" for c in section.commands)
        lines.append("")
    lines += [
        "Examples:",
        examples,
        "",
        f"Run `uv run {program} <{kind}> --help` for its options.",
    ]
    return "\n".join(lines)


def dispatch(
    argv: Sequence[str] | None,
    *,
    program: str,
    kind: str,
    sections: Sequence[Section],
    examples: str,
) -> None:
    """Run the module named by ``argv[0]`` as ``__main__`` with the remaining args."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(format_help(program, kind, sections, examples))
        return
    commands = all_commands(sections)
    name = args.pop(0)
    command = commands.get(name)
    if command is None:
        raise SystemExit(
            f"Unknown {kind} '{name}'. Expected one of: {', '.join(sorted(commands))}"
        )
    # alter_sys mirrors running the module directly: sys.argv[0] becomes the
    # module's file and sys.modules["__main__"] is the module, so pickling and
    # spawned DataLoader workers behave as they would under `python -m`.
    sys.argv = [command.module, *args]
    runpy.run_module(command.module, run_name="__main__", alter_sys=True)


def train_main(argv: Sequence[str] | None = None) -> None:
    """Entry point of ``train.py``."""
    dispatch(
        argv,
        program="train.py",
        kind="setting",
        sections=TRAIN_SECTIONS,
        examples=TRAIN_EXAMPLES,
    )


def eval_main(argv: Sequence[str] | None = None) -> None:
    """Entry point of ``eval.py``."""
    dispatch(
        argv,
        program="eval.py",
        kind="command",
        sections=EVAL_SECTIONS,
        examples=EVAL_EXAMPLES,
    )
