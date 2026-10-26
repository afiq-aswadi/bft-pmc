# Camera-ready figure inputs

Byte-for-byte copies of the inputs behind the camera-ready figures that were
redrawn from the retrained runs, so that those figures can be regenerated from
the repository alone. `SHA256SUMS` lists every file with its checksum and the
`outputs/` path it was copied from.

| Directory | Contents | Figures |
| --- | --- | --- |
| `lr/sweep/metrics.csv` | the no-positional-encoding linear-regression sweep (`outputs/lr/sweep_analysis_none/sweep_20260821_221637/`) | `lr_sweep_combined_2x2` |
| `bau/sweep/metrics.csv` | the no-positional-encoding balls-and-urns sweep (`outputs/bau/sweep_analysis_none/20260819_194316/`) | `bau_sweep_posterior{,_2x2}` |
| `pmc_diagnostics/<run>/` | the recovery-against-length and -count CSVs, dimension and task-diversity aggregates, path-stability table, and the run record of every diagnostics run the reliability figures read (37 runs) | all 41 `reliability_*` figures of Appendix D and the supplement |

Redraw them with:

```bash
uv run eval.py plot-lr-sweep --metrics-csv paper_data/camera_ready/lr/sweep/metrics.csv \
    --prompt-length 32 --output-dir outputs/camera_ready/lr/sweep
uv run eval.py plot-bau-sweep --metrics-csv paper_data/camera_ready/bau/sweep/metrics.csv \
    --output-dir outputs/camera_ready/bau/sweep
uv run eval.py plot-reliability --diagnostics-root paper_data/camera_ready/pmc_diagnostics \
    --paper-dir outputs/camera_ready/reliability
```

`scripts/reproduce_paper.sh` runs the same commands.

## Verification

On 2026-10-26 these copies were checked against the paper (`icl2latex`):

- All 41 reliability figures redrawn from them render pixel-identical to the
  PDFs in the paper. The figure manifest in the paper repository is older than
  the 2026-10-22 rerun of Appendix D.3: for the four figures
  `reliability_{length,count}_{prior,posterior}` it still lists the runs
  without `_pools`, which the figures no longer use. The other 37 entries
  match these files' checksums. Redrawing with
  `--paper-dir ../icl2latex` rewrites the manifest consistently.
- The three sweep figures are pixel-identical to the paper's when drawn with
  the layout each was made with. Since then the row labels moved left in both
  `plot_sweep_combined.py` (`xy=(-0.55, 0.5)` to `(-0.72, 0.5)`), and the
  paper's full balls-and-urns figure also predates the legend's move up
  (`anchor_y` 1.06 to 1.12; its 2x2 version already has it). These change the
  canvas, not the plotted values.

## Not included

The prior-panel sidecar CSVs of these sweeps (`prior_delta_mse.csv`,
`prior_predictive_kl.csv`), the sweeps' PMC sample bundles, and the
diagnostics runs' trajectories and rollouts stay on the cluster;
[`docs/figure_provenance.md`](../../docs/figure_provenance.md) lists what each
remaining figure needs.
