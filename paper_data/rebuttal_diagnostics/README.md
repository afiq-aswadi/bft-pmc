# Rebuttal-period diagnostics (historical)

These are the PMC rollout diagnostics reported in the author response, produced
on the `pmc-diagnostics` branch before the camera-ready retraining. They are
kept as the record of those numbers. **They are not camera-ready results**: the
camera-ready diagnostics in Appendix D were rerun on the retrained models (see
[`slurm/PMC_DIAGNOSTICS.md`](../../slurm/PMC_DIAGNOSTICS.md)), and replace these
runs rather than repeat them.

| Directory | Model | Prompt length | Rollout length N |
| --- | --- | --- | --- |
| `markov_m32/` | the pre-retrain 1-Markov M=32 model, 1024-state context | 8 | 500 |
| `bau_nope_vocab{6,12,24,48}/` | balls-and-urns M=32 models without positional encoding, vocabulary V, context extended to 1024 at rollout time | 16 | 500 |
| `bau_paper_m32/` | the submission's balls-and-urns M=32 checkpoint | 16 | 239 |
| `fidelity/` | next-token fidelity of the four `bau_nope_vocab*` models (256 sequences of 255 tokens) | – | – |

Each diagnostics directory holds, for `prior` and `posterior`:
`*_recovery_vs_length.csv` (energy distance and sliced Wasserstein to both
references at each N), `*_recovery_vs_count.csv` (mean and standard deviation
over subsets of R rollouts), `*_drift_vs_length.csv` (drift quantiles), and the
figure drawn from each.

## Reading them

- **The balls-and-urns rebuttal table comes from `bau_nope_vocab12`**, not from
  `bau_paper_m32`, which stops at N = 239.
- **Balls-and-urns drift is on the wrong scale.** These CSVs divide drift by V;
  they predate commit `cf9564c`, after which drift is divided by the free
  parameter count V − 1. Multiply by V/(V − 1) for the published scale: the
  V = 12 posterior endpoint 0.058449354 becomes 0.0637629, the rebuttal's
  0.0638.
- **No rollouts were saved**, so these runs have no path-stability (jellyfish)
  plot and their figures cannot be redrawn from per-rollout data. That is why
  the figures are committed here, unlike everywhere else in `paper_data/`.
- The checkpoint paths inside `fidelity/fidelity.csv` point at a temporary
  directory on the machine that ran it (the username in them reads `<user>`);
  the checkpoints are not distributed.

## Provenance

The SLURM jobs that produced the diagnostics directories are kept, unchanged, in
[`slurm/rebuttal/`](../../slurm/rebuttal/). The fidelity numbers were written by
the balls-and-urns-only `balls_and_urns/predictive_fidelity.py`, which the
camera-ready cleanup replaced with the two-family
[`bft_pmc/analysis/predictive_fidelity.py`](../../bft_pmc/analysis/predictive_fidelity.py)
(`uv run eval.py predictive-fidelity`). The original script is in the git history
at commit `11ce805`.
