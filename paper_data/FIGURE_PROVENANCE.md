# Figure provenance manifest

Every figure referenced by the paper (arXiv branch of icl2latex), its
generator, its inputs, and where those inputs live. "M3" means the inputs
exist only in the consolidated store on `/scratch2/qh36/` (single copy until
the RDS Vault backup lands). Update this table whenever a figure is added,
removed, or regenerated.

| Figures (count) | Generator | Inputs | Input location |
| --- | --- | --- | --- |
| `lr_sweep_combined_2x2`, `bau_sweep_posterior{,_2x2}`, `bau_sweep_prior_kl_ed_sw`, `markov_sweep_combined{,_2x2}`, `markov_sweep_prior_kl_ed_sw`, `lr_sweep_prior_delta_ed_sw` (8) | `linear_regression/plot_sweep_combined.py`, `balls_and_urns/plot_sweep_combined.py`, `markov/plot_sweep_combined.py`, `{linear_regression,balls_and_urns,markov}/plot_sweep_prior.py` | `paper_data/{lr,bau,markov}/sweep/` | repo |
| `lr_dynamics_combined_logx_2x2`, `bau_dynamics_combined_logx{,_2x2}`, `markov_dynamics_M8{,_2x2}` (5) | `linear_regression/plot_dynamics_combined.py`, `balls_and_urns/plot_dynamics_combined.py`, `markov/plot_dynamics_combined.py` | `paper_data/{lr,bau}/dynamics/`, `paper_data/markov/dynamics_m8/` | repo |
| `lr_sweep_ed_id_1x1`, `lr_dynamics_ed_id_1x1` (2, intro preview) | the two LR plotters above (`*_ed_id_1x1` emission) | `paper_data/lr/{sweep,dynamics}/` | repo |
| `lr_grid_marginal_4dim_T8_*` (3), `lr_stitched_*` (6), `lr_1x1_M8_dim0.pdf` (1) | `linear_regression/plot_marginals.py`, `linear_regression/plot_stitched_marginals.py`, `linear_regression/plot_single_marginal.py` | LR sweep `samples_enriched/*.npz` (~200 MB) | M3: `linear-regression/outputs/sweep_analysis/sweep_20260323_082051/` |
| `bau_grid_marginal_4dim_tqoxn029__*` (3), `stitched_bau_*` (6) | `balls_and_urns/plot_marginals.py --num-dims 4`, `balls_and_urns/plot_stitched_marginals.py` | BAU `samples/*.npz` (run `tqoxn029`) | M3: `balls-and-urns/outputs/bau_experiments_210426/.../20260421_104643/` |
| `markov_K4_M4_*` (3), `markov_full_M*` (60) | `markov/plot_matrix_marginals.py` (K4 = `--max-classes 4 --n-chains 4 --print-frac 0.45`) | Markov `pmc_samples.npz` per run | M3: `markov/outputs/markov_distribution_distance_{full,prior_n1024}/samples/` |
| `beta_bernoulli_main_figure`, `grid_beta_bernoulli` (2) | `balls_and_urns/beta_bernoulli.py --replot-from paper_data/beta_bernoulli/pmc_samples.npz` (writes `beta_bernoulli_pmc_grid` and `beta_bernoulli_main_figure`; the latter is grid panel `--main-figure-index` extracted on its own axes) | committed 100k-step run bundle | `paper_data/beta_bernoulli/` (`pmc_samples.npz`, `model.pt`) |
| `intro_identifiability_spike_slab` (1) | `scripts/plot_intro_identifiability.py` (ported from `notes/laplace_gaussian_identifiability.ipynb`, 2026-07-17) | analytic (no data) | repo |
| `ar-pfn-mask.pgf` (1) | **unknown** — no generator in this repo | none (schematic) | needs a committed script |

## Figure typography (2026-07-17)

All plotters now route font sizes through `plotting/paper_style.py`
(`apply_paper_style(fig_width_in, print_frac)`), which scales text so it
prints at 7.5pt labels / 6.5pt ticks / 7pt legends regardless of canvas
size. On 2026-07-17 ALL paper figures were regenerated with the shared style
(sample bundles pulled from M3 to a local temp dir): the six sweep/dynamics
combined figures (+2x3 twins), the three prior-sweep figures, the two
Beta-Bernoulli figures (via the new `--replot-from` mode), the LR 4-dim
grids, `lr_1x1_M8_dim0.pdf`, the BAU 4-dim grids, both stitched sets, and
the Markov K4 + full matrix marginals (priors from the n1024 bundles merged
over the full-run posteriors). Posterior panels now label the green/blue
references $\Pi^{mem}(\cdot|c)$ / $\Pi^{gen}(\cdot|c)$ instead of the
prior symbols (`plotting.marginal_cell.legend_handles(posterior=...)` and the
Markov matrix legend).

## Known gaps (as of 2026-07-17)

1. The one remaining **unknown** row above (`ar-pfn-mask.pgf`) is the last provenance hole.
2. All marginal-grid figures (73 of 97) depend on M3-only sample bundles;
   they are documented in README "External data" but have a single copy
   until the Vault backup completes.
3. `paper_data/markov/dynamics_m8/` duplicates the M=8 `wandb_kl_history.csv`
   from `paper_data/markov/sweep/runs/` (same run, same file): the sweep uses
   its last row, the dynamics plot uses the full trajectory. Deliberate.
