# `bft_pmc.linear_regression`

In-context linear regression, y = wᵀx + noise with x ∈ ℝ⁸. The model is
trained on tasks w drawn uniformly from a finite pool of M draws from N(0, I)
(the task diversity M), and PMC recovers its implicit distribution over w.
The two references are the exact posterior over the finite pool
(**memorizing**, the dMMSE estimator) and Bayesian linear regression under the
population prior N(0, I) (**generalizing**, the ridge estimator).

## Modules

**Task, references and estimator**

| Module | Contents |
| --- | --- |
| `priors.py` | `DiscretePrior`: uniform over a fixed pool of Gaussian tasks |
| `likelihoods.py` | the linear likelihood used by the data generator |
| `baselines.py` | memorizing and generalizing predictors and posteriors |
| `evals.py` | prediction-space evaluation (`ICLEvaluator`, `mse`) |
| `predictive_monte_carlo.py` | PMC: generate (x, y) continuations and fit least squares on each rollout |
| `analysis/` | the sweep-evaluation pipeline: `config.py` (its options), `data.py` (checkpoints, prompts, evaluation datasets), `metrics.py` (prediction- and latent-space metrics, reference posterior samples), `plotting.py` (per-run figures), `runner.py` (orchestration) |

**Commands**

| Module | Command | What it does |
| --- | --- | --- |
| `train.py` | `train.py lr` | train one model |
| `sweep_analysis.py` | `eval.py lr-sweep` | evaluate the final checkpoint of every run in a sweep |
| `distribution_dynamics.py` | `eval.py lr-dynamics` | evaluate every checkpoint of one run |
| `rollout_diagnostics.py` | `eval.py lr-rollout-diagnostics` | recovery error against rollout length and count, drift (Appendix D) |
| `prior_delta_mse.py` | `eval.py lr-prior-delta-mse` | zero-context Delta MSE, for the prior figure |
| `prior_shape_discrimination.py` | `eval.py prior-shape` | Gaussian against variance-matched Laplace prior, from saved prior rollouts |
| `plot_sweep_combined.py` | `eval.py plot-lr-sweep` | task-diversity sweep figure |
| `plot_dynamics_combined.py` | `eval.py plot-lr-dynamics` | training-dynamics figure |
| `plot_sweep_prior.py` | `eval.py plot-lr-prior` | prior-only sweep: Delta MSE, energy distance, sliced Wasserstein |
| `plot_stitched_sweep_dynamics.py` | `eval.py plot-lr-sweep-and-dynamics` | sweep and dynamics side by side |
| `plot_marginals.py` | `eval.py plot-lr-marginals` | one marginal grid per sample bundle |
| `plot_stitched_marginals.py` | `eval.py plot-lr-stitched-marginals` | marginals across every task diversity |
| `plot_single_marginal.py` | `eval.py plot-lr-single-marginal` | one publication-sized marginal panel |
| `plot_intro_identifiability.py` | `eval.py plot-identifiability` | Gaussian against spike-and-slab identifiability example (Figure 2) |

The marginal plotters read the `samples/*.npz` bundles written by
`eval.py lr-sweep`; the frozen metric tables cannot reconstruct a
distribution. [`docs/pmc_outputs.md`](../../docs/pmc_outputs.md) describes the
bundles and how to produce them. All marginal plotters draw through
`bft_pmc.plotting.marginal_cell`, so a marginal has the same bins, limits and
line styles in single, grid and stitched layouts.
