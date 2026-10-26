# Figure provenance

Every figure in the paper and supplement, the command that draws it, its
inputs, and where those inputs live. Update this file whenever a figure is
added, removed or regenerated.

**Where inputs live.** *repo* means committed under `paper_data/`. *outputs*
means the untracked `outputs/` tree of the authors' checkouts, pulled from the
cluster with [`scripts/m3/`](../scripts/README.md#cluster-sync-scripts); the
code that writes it is here, the files are not. *cluster* means the inputs
(PMC sample bundles, rollouts) exist only on the authors' cluster storage,
`/projects/qh36/bft-pmc/outputs/` on Monash M3 (the older `/scratch2/qh36/`
store holds the pre-retraining runs).

## Figures and their inputs

| Figures (count) | Command | Inputs | Location |
| --- | --- | --- | --- |
| `beta_bernoulli_main_figure`, `grid_beta_bernoulli` (2) | `train.py beta-bernoulli --replot-from paper_data/beta_bernoulli/pmc_samples.npz` (writes `beta_bernoulli_pmc_grid` and `beta_bernoulli_main_figure`; the latter is grid panel `--main-figure-index` on its own axes) | the 100k-step run's PMC samples and model | repo: `paper_data/beta_bernoulli/` |
| `intro_identifiability_spike_slab` (1) | `eval.py plot-identifiability` | analytic, no data | — |
| `lr_sweep_combined_2x2`, `lr_sweep_prior_delta_ed_sw`, `bau_sweep_posterior{,_2x2}`, `bau_sweep_prior_kl_ed_sw` (5) | `eval.py plot-lr-sweep --prompt-length 32`, `plot-lr-prior`, `plot-bau-sweep`, `plot-bau-prior` | `lr/sweep_analysis_none/sweep_20260821_221637/metrics.csv`, `bau/sweep_analysis_none/20260819_194316/metrics.csv`, plus their prior sidecar CSVs | outputs |
| `markov_sweep_combined{,_2x2}`, `markov_sweep_prior_kl_ed_sw` (3) | `eval.py plot-markov-sweep`, `plot-markov-prior` | `paper_data/markov/sweep/` (unverified; see the correction below) | repo |
| `lr_dynamics_combined_logx_2x2`, `bau_dynamics_combined_logx{,_2x2}`, `markov_dynamics_M8{,_2x2}` (5) | `eval.py plot-lr-dynamics`, `plot-bau-dynamics`, `plot-markov-dynamics` | `paper_data/{lr,bau}/dynamics/`, `paper_data/markov/dynamics_m8/` (unverified; see below) | repo |
| `lr_grid_marginal_4dim_T8_*` (3), `lr_stitched_*` (6), `lr_1x1_M8_dim0.pdf` (1) | `eval.py plot-lr-marginals --num-dims 4`, `plot-lr-stitched-marginals`, `plot-lr-single-marginal` | LR sweep `samples/*.npz` | cluster: `outputs/lr/sweep_analysis_{learned,rope,none}/sweep_*/` |
| `bau_grid_marginal_4dim_*` (3), `stitched_bau_*` (6) | `eval.py plot-bau-marginals --num-dims 4`, `plot-bau-stitched-marginals` | balls-and-urns sweep `samples/*.npz` | cluster: `outputs/bau/sweep_analysis_{learned,rope,none}/*/` |
| `markov_K4_M4_*` (3), `markov_full_M*` (60) | `eval.py plot-markov-marginals --prompt-length 32` (K4: `--max-classes 4 --n-chains 4 --print-frac 0.45`) | Markov `samples/M{n}_{prior,in_distribution_L32,out_of_distribution_L32}.npz` | cluster: `outputs/markov/sweep_analysis/samples/` |
| `reliability_*` (41: Appendix D and the supplement) | `eval.py plot-reliability --paper-dir ../icl2latex` | the per-run CSVs and run records of `outputs/pmc_diagnostics/`; the written `reliability_figure_manifest.json` records every input's SHA-256 and every plotted value | outputs |
| `ar-pfn-mask.pgf` (1) | **unknown**: no generator in this repository | none (schematic) | — |

`scripts/rebuild_marginals.sh` regenerates every marginal grid, with the right
options, for all three positional encodings; `scripts/rebuild_sweep_figures.sh`
does the same for the sweep figures. Prefer them to reconstructing the calls by
hand. Numbers quoted in the text rather than plotted come from
`eval.py prior-shape` (Gaussian against Laplace prior, `slurm/prior_shape.slurm`)
and `eval.py predictive-fidelity` (`slurm/predictive_fidelity.slurm`).

## Traps

Three, each of which cost real time to rediscover:

1. **The LR sweep plotter defaults to `--prompt-length 0`**, which takes the
   prior branch and writes only `sweep_combined.png`. The posterior panels
   need the prompt length of the sweep: 32 for the camera-ready runs, 8 for
   `paper_data/lr/sweep/`.
2. **The LR and balls-and-urns dynamics plotters both write
   `dynamics_combined*.png`.** Give them separate `--output-dir`s, or the
   second silently overwrites the first.
3. **The Markov matrix marginals take `--prompt-length`.** The 2026-08 sweeps
   use 32; earlier runs used 8, which is still the default.

(A fourth trap, running family scripts by path and having `markov/plotting.py`
shadow the top-level `plotting/` package, disappeared when the code moved into
the single `bft_pmc` package. Run everything through `eval.py` and `train.py`.)

## History

### Correction, 2026-09-20

The sweep row previously recorded `paper_data/{lr,bau,markov}/sweep/` as the
input of all eight sweep figures. That is wrong for linear regression and
balls-and-urns: those figures were swapped to the no-positional-encoding runs,
and `paper_data/{lr,bau}/sweep/metrics.csv` is superseded. The two disagree
materially: the stale balls-and-urns file puts the posterior switch at
$M=2^8$–$2^9$ and shows no prediction-space switch at all, whereas the NoPE run
gives $2^7$–$2^8$ and $2^9$–$2^{10}$, which is what the paper states and what
was verified against these CSVs. The Markov sweep row and the dynamics row
carry the same risk and have **not** been re-checked.

### Regeneration of 2026-08-23

All marginal grids were rebuilt from the post-retraining sweeps (256-token
contexts, prompt length 32, rollout length = context length) with
`scripts/rebuild_marginals.sh`: 23 steps, no failures. The generators emit one
figure per sample bundle, a superset of the figures the paper selects (it picks
specific `T8` / `M4` cases):

| Family | Per encoding | Total |
| --- | --- | --- |
| LR | 68 grid + 8 stitched + 3 single | 237 |
| Balls and urns | 78 grid + 12 stitched | 270 |
| Markov (no encoding variants) | — | 120 full + 12 K4 |

`lr_1x1_M8_dim0` was recorded without its prompt source, so all three
(`discrete`, `gaussian`, `random`) are emitted with the source in the
filename; match against the paper before committing one.

**Comparability caveat.** The LR evaluation data differs between `rope` and
the other two encodings: `data_generalising/baseline_generalising_mse`, a fixed
estimator and so model-independent, reads 0.1196 for `learned` and `none` and
0.1268 for `rope`, with memorizing baselines differing by 16–18%. `learned`
against `none` is a clean comparison; anything involving `rope` is confounded
and should use the `delta_vs_baseline_*` columns, which subtract the
difficulty. Balls-and-urns is unaffected (identical seed 42 across all three).

### Figure typography, 2026-07-17

All plotters route font sizes through `bft_pmc/plotting/paper_style.py`
(`apply_paper_style(fig_width_in, print_frac)`), which scales text so that it
prints at 7.5 pt labels, 6.5 pt ticks and 7 pt legends whatever the canvas
size. On 2026-07-17 every paper figure was regenerated with the shared style.
Posterior panels label the green and blue references $\Pi^{mem}(\cdot|c)$ and
$\Pi^{gen}(\cdot|c)$ rather than the prior symbols
(`bft_pmc.plotting.marginal_cell.legend_handles(posterior=...)` and the Markov
matrix legend).

## Known gaps

1. `ar-pfn-mask.pgf` has no committed generator: the last provenance hole.
2. The marginal figures depend on cluster-only sample bundles.
3. The retraining did not persist the trainer logs (`num_tasks_*.json`), so
   the positional-encoding comparison has no training-loss curves; only
   wall-clock time is known (RoPE cost about 17% more: 16:13 against 13:53).
4. The task-diversity sweeps saved LR and balls-and-urns checkpoints every 500
   steps, not the "200 log-spaced and every 500" schedule the appendix
   describes, so nothing exists before step 500 there.
   `slurm/retrain_transient.slurm` retrains the M = 32 runs with the
   documented schedule for the transient-generalization figures.
5. `paper_data/markov/dynamics_m8/` duplicates the M = 8 `wandb_kl_history.csv`
   of `paper_data/markov/sweep/runs/` (same run, same file): the sweep reads its
   last row, the dynamics plot the whole trajectory. Deliberate.
