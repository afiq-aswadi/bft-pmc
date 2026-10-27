# `bft_pmc.markov`

Finite-state 1-Markov chains: each sequence draws a k × k transition matrix T
(k = 10 in the paper) and then a chain from it. The model, a small decoder-only
transformer with rotary position embeddings, is trained on matrices drawn
uniformly from a finite pool of M, and PMC recovers its implicit distribution
over T from Laplace-smoothed transition counts of its rollouts. The two
references are the posterior over the finite pool (**memorizing**, `bi_ret`)
and the row-wise Dirichlet posterior (**generalizing**, `bi_inf`).

## Modules

**Task, model, references and estimator**

| Module | Contents |
| --- | --- |
| `config.py` | `MarkovConfig` and its YAML loading, overrides and dumping |
| `data.py` | `MarkovChainDataset`: the transition-matrix pool, stationary distributions and chain sampling |
| `model.py` | `MarkovTransformer`, with optional rotary position interpolation for long rollouts |
| `baselines.py` | memorizing and generalizing predictives and posterior samplers, plus the order-0 (unigram) predictives used as misspecified references during training |
| `evals.py` | the in- and out-of-distribution KL and baseline deltas logged during training |
| `predictive_monte_carlo.py` | PMC: generate continuations and estimate a transition matrix from each |
| `samples_saving.py` | generating and saving prior and posterior PMC sample bundles |
| `analysis_common.py` | loading a trained run: model, config and chain pool |
| `kl_history.py` | KL trajectories for the figures, from W&B exports or training logs |
| `run_names.py` | parsing sweep run names |
| `plotting.py` | shared Markov figure helpers |

**Commands**

| Module | Command | What it does |
| --- | --- | --- |
| `train.py` | `train.py markov` | train one model, recording its dynamics and final PMC samples |
| `sweep_analysis.py` | `eval.py markov-sweep` | evaluate the checkpoints of every run in a sweep |
| `run_pmc.py` | `eval.py markov-pmc` | PMC samples for one checkpoint |
| `task_diversity_threshold.py` | `eval.py markov-threshold` | train across task diversities and estimate the threshold |
| `rollout_diagnostics.py` | `eval.py markov-rollout-diagnostics` | recovery error against rollout length and count, drift (Appendix D) |
| `prior_predictive_kl.py` | `eval.py markov-prior-kl` | first-step predictive KL, for the prior figure |
| `plot_sweep_combined.py` | `eval.py plot-markov-sweep` | task-diversity sweep figures (KL, energy distance, sliced Wasserstein) |
| `plot_dynamics_combined.py` | `eval.py plot-markov-dynamics` | training-dynamics figures |
| `plot_sweep_prior.py` | `eval.py plot-markov-prior` | prior-only sweep figure |
| `plot_matrix_marginals.py` | `eval.py plot-markov-marginals` | transition-matrix marginal grids |

Training reads [`configs/markov/train.yaml`](../../configs/markov/train.yaml)
and writes each run's resolved config and chain pool to
`outputs/markov/training/<run>/`; every later command needs them as well as the
checkpoint. Long rollouts past the trained 256 positions use
`--seq-len-override` with `--rope-scaling interpolate` (rescale every position
into the trained range) or `extrapolate` (keep positions as trained). The
matrix-marginal plotter reads the sample bundles of `eval.py markov-sweep`;
see [`docs/pmc_outputs.md`](../../docs/pmc_outputs.md).
