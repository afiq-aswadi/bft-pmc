# Balls-and-urns experiments

This package contains the Dirichlet-multinomial task-diversity experiments.

## Entry points

- `train.py`: train one balls-and-urns PFN;
- `sweep_analysis.py`: evaluate final checkpoints across task diversity;
- `distribution_dynamics.py`: evaluate one run throughout training;
- `plot_sweep_combined.py`: posterior task-diversity figure;
- `plot_dynamics_combined.py`: posterior training-dynamics figure;
- `plot_sweep_prior.py`: prior-only KL and distribution metrics;
- `plot_marginals.py`: one marginal grid per saved sample bundle;
- `plot_stitched_marginals.py`: multi-M marginal grids.

`baselines.py`, `evals.py`, `dataset.py`, and `predictive_monte_carlo.py` contain
the analytic references and shared evaluation logic. Marginal rendering is
shared with the other task families through `plotting/marginal_cell.py`.

The marginal plotters require `samples/*.npz` bundles emitted by the
balls-and-urns sweep analysis; the precomputed bundles are not committed, and
the frozen metric CSVs are not enough to reconstruct a distribution. See the
[generation, layout, and plotting commands](../paper_data/README.md#generate-sample-bundles-for-marginal-plots).
