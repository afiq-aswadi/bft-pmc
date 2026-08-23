# Figure provenance manifest

Every figure referenced by the paper (arXiv branch of icl2latex), its
generator, its inputs, and where those inputs live. "M3" means the inputs
exist only on the cluster: as of 2026-08-23 that is
`/projects/qh36/bft-pmc/outputs/` (the older `/scratch2/qh36/` store holds the
pre-retrain runs). Update this table whenever a figure is added, removed, or
regenerated.

## Read this before regenerating anything (2026-08-23)

Four traps, each of which cost real time to rediscover:

1. **Run the family scripts with `python -m`, never by path.** Invoking
   `markov/plot_*.py` or `balls_and_urns/plot_*.py` directly puts that package
   first on `sys.path`, where `markov/plotting.py` shadows the top-level
   `plotting/` package and `balls_and_urns/analysis.py` shadows `analysis/`.
   Both fail with import errors that look unrelated to the cause.
2. **LR's sweep plotter defaults to `--prompt-length 0`**, which takes the
   prior branch and emits only `sweep_combined.png`. The paper's
   `lr_sweep_combined_2x2` needs `--prompt-length 8`.
3. **LR and BAU dynamics plotters both write `dynamics_combined*.png`.** Give
   them separate `--output-dir`s or BAU silently overwrites LR.
4. **Markov matrix marginals take `--prompt-length`.** The 2026-08 sweeps use
   32; runs before that used 8, which is still the default.

`scripts/rebuild_marginals.sh` performs every marginal-grid regeneration in
one command, with the correct invocations, for all three positional-encoding
variants. Prefer it to reconstructing the calls by hand.

| Figures (count) | Generator | Inputs | Input location |
| --- | --- | --- | --- |
| `lr_sweep_combined_2x2`, `bau_sweep_posterior{,_2x2}`, `bau_sweep_prior_kl_ed_sw`, `markov_sweep_combined{,_2x2}`, `markov_sweep_prior_kl_ed_sw`, `lr_sweep_prior_delta_ed_sw` (8) | `linear_regression/plot_sweep_combined.py`, `balls_and_urns/plot_sweep_combined.py`, `markov/plot_sweep_combined.py`, `{linear_regression,balls_and_urns,markov}/plot_sweep_prior.py` | `paper_data/{lr,bau,markov}/sweep/` | repo |
| `lr_dynamics_combined_logx_2x2`, `bau_dynamics_combined_logx{,_2x2}`, `markov_dynamics_M8{,_2x2}` (5) | `linear_regression/plot_dynamics_combined.py`, `balls_and_urns/plot_dynamics_combined.py`, `markov/plot_dynamics_combined.py` | `paper_data/{lr,bau}/dynamics/`, `paper_data/markov/dynamics_m8/` | repo |
| `lr_grid_marginal_4dim_T8_*` (3), `lr_stitched_*` (6), `lr_1x1_M8_dim0.pdf` (1) | `python -m linear_regression.plot_marginals --num-dims 4`, `.plot_stitched_marginals`, `.plot_single_marginal` | LR sweep `samples/*.npz` | M3: `outputs/lr/sweep_analysis_{learned,rope,none}/sweep_*/` |
| `bau_grid_marginal_4dim_*` (3), `stitched_bau_*` (6) | `python -m balls_and_urns.plot_marginals --num-dims 4`, `.plot_stitched_marginals` | BAU `samples/*.npz` | M3: `outputs/bau/sweep_analysis_{learned,rope,none}/*/` |
| `markov_K4_M4_*` (3), `markov_full_M*` (60) | `python -m markov.plot_matrix_marginals --prompt-length 32` (K4 = `--max-classes 4 --n-chains 4 --print-frac 0.45`) | Markov `samples/M{n}_{prior,in_distribution_L32,out_of_distribution_L32}.npz` | M3: `outputs/markov/sweep_analysis/samples/` |
| `path_stability{,_by_family}`, `path_stability_{learned,rope,none,markov}` (6) | `eval.py path-stability` (`--realisations 1` reproduces the TabMGP Fig. 3 estimand; `0` averages over all prompts) | `*_rollouts.npz` beside every sample bundle | M3: `outputs/path_stability/` |
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

## Regeneration of 2026-08-23

All marginal grids were rebuilt from the post-retrain sweeps (256-token
contexts, prompt length 32, rollout length = context length) via
`scripts/rebuild_marginals.sh`, 23 steps, no failures. Counts produced, which
are a *superset* of the figures the paper selects — the generators emit one
per sample bundle, the paper picks specific `T8` / `M4` cases:

| Family | per encoding | total |
| --- | --- | --- |
| LR | 68 grid + 8 stitched + 3 single | 237 |
| BAU | 78 grid + 12 stitched | 270 |
| Markov (no encoding variants) | — | 120 full + 12 K4 |

`lr_1x1_M8_dim0` was recorded without its prompt source, so all three
(`discrete`, `gaussian`, `random`) are emitted with the source in the
filename; match against the paper before committing one.

The 16 `paper_data`-backed figures were regenerated locally into
`outputs/paper_figures/{lr,bau,markov,misc}/`.

**Comparability caveat.** The LR eval data differs between `rope` and the
other two encodings: `data_generalising/baseline_generalising_mse` — a fixed
estimator, so model-independent — reads 0.1196 for `learned` and `none` and
0.1268 for `rope`, with memorising baselines differing by 16-18%. `learned`
vs `none` is a clean comparison; anything involving `rope` is confounded and
should use the `delta_vs_baseline_*` columns, which subtract the difficulty.
BAU is unaffected (identical seed 42 across all three).

## Known gaps (as of 2026-08-23)

1. The one remaining **unknown** row above (`ar-pfn-mask.pgf`) is the last provenance hole.
2. All marginal-grid figures depend on M3-only sample bundles; they are
   documented in README "External data" but have a single copy until the
   Vault backup completes.
3. Trainer logs (`num_tasks_*.json`) were not persisted by the retrain, so the
   positional-encoding comparison has no training-loss curves. Only wall-clock
   is available: RoPE cost ~17% more time (16:13 vs 13:53).
4. LR and BAU checkpoints were saved every 500 steps rather than the
   "200 log-spaced ∪ every 500" the appendix specifies, so nothing exists
   before step 500 and the transient-generalisation figures start after the
   transient they illustrate.
3. `paper_data/markov/dynamics_m8/` duplicates the M=8 `wandb_kl_history.csv`
   from `paper_data/markov/sweep/runs/` (same run, same file): the sweep uses
   its last row, the dynamics plot uses the full trajectory. Deliberate.
