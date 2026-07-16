# Markov-chain experiments

This package contains the finite-state Markov task-diversity experiments.

## Entry points

- `train.py`: train one transient Markov PFN;
- `sweep_analysis.py`: evaluate checkpoint sweeps with ED and SW metrics;
- `task_diversity_threshold.py`: run the threshold experiment;
- `run_pmc.py`: generate posterior Monte Carlo artifacts;
- `plot_sweep_combined.py`: posterior KL, ED, and SW task-diversity figure;
- `plot_dynamics_combined.py`: posterior KL, ED, and SW training dynamics;
- `plot_sweep_prior.py`: prior-only KL, ED, and SW figure;
- `plot_matrix_marginals.py`: transition-matrix marginal grids.

Regenerate the bundled aggregate figures with shipped data:

```bash
uv run eval.py plot-markov-sweep \
  --runs-dir paper_data/markov/sweep/runs \
  --metrics-csv paper_data/markov/sweep/metrics.csv \
  --out-path outputs/paper_reproduction/markov/sweep/sweep_combined.png
```

The transition-matrix marginal plotter requires the larger PMC sample bundles
emitted by Markov sweep analysis; the precomputed bundles are not committed,
and the frozen metric CSVs are not enough to reconstruct a distribution. See
the
[generation, layout, and plotting commands](../paper_data/README.md#generate-sample-bundles-for-marginal-plots).
