# `bft_pmc.analysis`

Tools shared by the three task families: checkpoint and rollout I/O, and the
PMC reliability diagnostics of Appendix D, which read what the per-family
diagnostics runs save rather than a model.

## Modules

| Module | Command | Contents |
| --- | --- | --- |
| `checkpoints.py` | – | find the checkpoints of a run and rebuild the model from one (`load_run_info`) |
| `rollouts.py` | – | save and load PMC rollout bundles beside their sample bundles |
| `pmc_diagnostics.py` | – | shared machinery of `eval.py *-rollout-diagnostics`: recovery, drift and count curves, their figures, run records and the `figures.json` index |
| `path_stability.py` | `eval.py path-stability` | scaled-L1 path stability of saved rollouts (the jellyfish plots) |
| `dimension_sweep.py` | `eval.py recovery-vs-dimension` | aggregate diagnostics runs by latent dimension or task diversity |
| `replot_pmc_diagnostics.py` | `eval.py replot-pmc-diagnostics` | redraw diagnostics figures from their saved data |
| `predictive_fidelity.py` | `eval.py predictive-fidelity` | next-token fidelity against the exact predictive (balls and urns, 1-Markov) |
| `pos_encoding_comparison.py` | `eval.py compare-pos-encodings` | compare the learned, rotary and no-encoding sweeps |
| `plot_reliability_appendix.py` | `eval.py plot-reliability` | every reliability figure of Appendix D and the supplement, with a provenance manifest |

The family-specific halves of the diagnostics (sampling rollouts and the
estimator on each prefix) live in each family's `rollout_diagnostics.py`. What
the diagnostics write is described in
[`docs/pmc_outputs.md`](../../docs/pmc_outputs.md#reliability-diagnostics),
and the settings of every camera-ready run in
[`slurm/PMC_DIAGNOSTICS.md`](../../slurm/PMC_DIAGNOSTICS.md).
