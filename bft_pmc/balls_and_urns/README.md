# `bft_pmc.balls_and_urns`

Balls and urns: each sequence draws an urn θ on the (V − 1)-simplex and then
i.i.d. balls of V colours from it. The model is trained on urns drawn
uniformly from a finite pool of M urns from a symmetric Dirichlet, and PMC
recovers its implicit distribution over θ from the colour frequencies of its
rollouts. The two references are the posterior over the finite pool
(**memorizing**) and the Dirichlet-multinomial posterior under the population
prior (**generalizing**). The package also holds the continuous-prior
**Beta-Bernoulli** experiment of Figure 1.

## Modules

**Task, references and estimator**

| Module | Contents |
| --- | --- |
| `data.py` | data generators: Dirichlet prior, finite urn pool, BOS prefixing |
| `baselines.py` | memorizing and generalizing predictives, posteriors and posterior samplers |
| `evals.py` | prediction-space evaluation during training (`BAUEvaluator`) |
| `dataset.py` | saving and loading the fixed evaluation prompts and predictive samples |
| `analysis.py` | prediction- and latent-space metrics shared by the sweep and dynamics evaluations |
| `predictive_monte_carlo.py` | PMC: generate continuations and read off empirical frequencies |

**Commands**

| Module | Command | What it does |
| --- | --- | --- |
| `train.py` | `train.py bau` | train one model |
| `beta_bernoulli.py` | `train.py beta-bernoulli` | train the Beta-Bernoulli model, run PMC and draw Figure 1 |
| `generate_eval_dataset.py` | `eval.py bau-eval-dataset` | fixed evaluation prompts for a checkpoint sweep |
| `sweep_analysis.py` | `eval.py bau-sweep` | evaluate the final checkpoint of every run in a sweep |
| `distribution_dynamics.py` | `eval.py bau-dynamics` | evaluate every checkpoint of one run |
| `rollout_diagnostics.py` | `eval.py bau-rollout-diagnostics` | recovery error against rollout length and count, drift (Appendix D) |
| `samples_count_panel.py` | `eval.py bau-samples-count-panel` | recovery error against rollout count from a saved sample bundle |
| `prior_predictive_kl.py` | `eval.py bau-prior-kl` | first-token predictive KL, for the prior figure |
| `plot_sweep_combined.py` | `eval.py plot-bau-sweep` | task-diversity sweep figures |
| `plot_dynamics_combined.py` | `eval.py plot-bau-dynamics` | training-dynamics figures |
| `plot_sweep_prior.py` | `eval.py plot-bau-prior` | prior-only sweep: KL, energy distance, sliced Wasserstein |
| `plot_marginals.py` | `eval.py plot-bau-marginals` | one marginal grid per sample bundle |
| `plot_stitched_marginals.py` | `eval.py plot-bau-stitched-marginals` | marginals across every task diversity |

The memorizing evaluation prompts depend on each run's own urn pool, so a new
sweep needs `eval.py bau-eval-dataset` before `eval.py bau-sweep`; the original
paper checkpoints use `paper_data/bau/eval_datasets/`. The marginal plotters
read the `samples/*.npz` bundles the sweep writes; see
[`docs/pmc_outputs.md`](../../docs/pmc_outputs.md).
