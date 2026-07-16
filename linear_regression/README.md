# Linear-regression experiments

This package contains the LR task-diversity experiments and their figure code.

## Entry points

- `train.py`: train one PFN on a finite Gaussian task pool;
- `sweep_analysis.py`: evaluate final checkpoints across task diversity;
- `distribution_dynamics.py`: evaluate one run throughout training;
- `plot_sweep_combined.py`: posterior task-diversity figure;
- `plot_dynamics_combined.py`: posterior training-dynamics figure;
- `plot_sweep_prior.py`: prior-only prediction and distribution metrics;
- `plot_stitched_sweep_dynamics.py`: combined sweep-and-dynamics paper layout;
- `plot_marginals.py`: one full-dimensional marginal grid per run;
- `plot_stitched_marginals.py`: multi-M marginal grids;
- `plot_single_marginal.py`: one publication-sized marginal panel.

`analysis/` contains the reusable sweep-analysis components. All marginal
plotters use `plotting/marginal_cell.py` for the actual density and CDF
rendering, so the same marginal has the same bins, limits, and line semantics in
single, grid, and stitched layouts.

These marginal plotters require `samples/*.npz` bundles emitted by the LR sweep
analysis; the precomputed bundles are not committed, and the frozen metric CSVs
are not enough to reconstruct a distribution. See the
[generation, layout, and plotting commands](../paper_data/README.md#generate-sample-bundles-for-marginal-plots).

The two reference predictors are the exact posterior over the finite task pool
(memorising) and Bayesian linear regression under the population Gaussian
prior (generalising).
