# PMC rollout diagnostics: settings record

The rollout diagnostics behind the rebuttal tables, re-run for the camera-ready on M3:
**recovery error vs rollout length**, **recovery error vs rollout count** and **drift vs
length**, for linear regression (LR), balls and urns (BAU) and 1-Markov, the latent-dimension
sweeps built from them, the same diagnostics repeated at **task diversities either side of each
family's threshold**, and the **jellyfish plot** of Ng et al. (TabMGP, Figures 3 and 8) from the
same models. Every run draws 1024 rollouts. This file records every setting, what each run
saves, and how to reproduce or redraw it.

The BAU and Markov diagnostics are computed exactly as on the `pmc-diagnostics` branch: the
camera-ready copies of `analysis/pmc_diagnostics.py` and the two family scripts only add
saving and a context check. Run on the same tiny models with the same arguments, the
`pmc-diagnostics` code and this code wrote identical trajectories, recovery curves, drift
curves, per-path drift and count curves (30 of 30 CSV/NPZ outputs). The branch had no LR
version; `linear_regression/rollout_diagnostics.py` follows the other two and takes its
sampler, prompts, baselines and estimator from the camera-ready LR sweep.

Rebuttal items covered (see `camera-ready-todo.md`): B2 (jellyfish plots), B3
(rollout-length, rollout-count and dimension figures), C1 (the LR rollout diagnostics), C2 (the
reliability analyses at other task diversities), D1 (1-Markov dimension sweep).

## Models

Every model has the camera-ready configuration of `slurm/train.slurm`, trained by the
camera-ready code:

| Family | Models |
| --- | --- |
| LR | the camera-ready `checkpoints/lr/task_diversity_{learned,rope,none}/M32` checkpoints |
| BAU | the camera-ready `checkpoints/bau/task_diversity_{learned,rope,none}/M32` checkpoints (vocabulary 12). For the dimension sweep, vocabularies 6, 24 and 48 are trained with the `task_diversity_none/M32` command, changing only `--vocab-size`; vocabulary 12 is that camera-ready model itself |
| Markov | the camera-ready `checkpoints/markov/task_diversity/markov_M32` checkpoint (k = 10, seq_len 256). For the dimension sweep, k = 3, 4, 6 and 8 are trained with the `markov_M32` command, changing only `--k` |

The task-diversity runs use other checkpoints of the same camera-ready sweeps, at the task
diversities of [Task diversities](#task-diversities): `checkpoints/lr/task_diversity_{pos}/M{M}`,
`checkpoints/bau/task_diversity_{pos}/M{M}` and `checkpoints/markov/task_diversity/markov_M{M}`.
Nothing is trained for them.

## Results not rerun

The rebuttal runs on `pmc-diagnostics` are not repeated here. Their results stay in
`diagnostics_results/`, byte-identical to that branch:

| Directory | Model | Prompt | N |
| --- | --- | --- | --- |
| `markov_m32` | the pre-retrain Markov M=32 model, seq_len 1024 (`/scratch2/qh36/<user>/markov_m32/`) | 8 | 500 |
| `bau_nope_vocab{6,12,24,48}` | NoPE BAU models trained on that branch, context extended to 1024 at rollout time | 16 | 500 |
| `bau_paper_m32` | the submission's BAU M=32 checkpoint, from before the camera-ready retraining | 16 | 239 |

These models are outside the camera-ready configuration, so the results are the rebuttal's
record, not camera-ready results:

- **Markov:** the model has a 1024-state context, where the camera-ready Markov models have 256.
- **NoPE BAU:** the models were trained from another copy of the code, with a logarithmic
  checkpoint schedule instead of the camera-ready linear one.

The camera-ready runs below replace these runs rather than repeat them.

Two things to know when reading them:

- **BAU drift scale:** the committed BAU drift CSVs divide by V; they predate commit `cf9564c`.
  Multiply them by V/(V − 1) for the published scale.
- **No rollouts:** none of these runs saved rollouts, so they have no jellyfish plot or
  per-rollout redraw.

## Rollout length

The camera-ready models were trained with a 256-token context. Two lengths are run:

- **N = 223 after a 32-token prompt, inside the context, for every family.** It fills the
  context (BOS + 32 + 223 = 256 positions for BAU, 32 + 223 = 255 of the 256 (x, y) pairs a
  512-token LR model holds, 255 of 256 states for Markov) and is the setting behind every PMC
  result in the camera-ready paper (`slurm/analyse.slurm`), so it shares the paper's estimates
  and the 0–223 axis of the camera-ready jellyfish.
- **N = 500 after the same prompt, past the context.** For LR and BAU: all three positional
  encodings at M = 32 and at the other task diversities, and the BAU vocabulary sweep. 500 is
  the forward-sampling length of Ng et al. Markov runs it at M = 32 only, and by position
  interpolation rather than a longer rotary table (see below), which makes it a different model
  setting from the N = 223 Markov run rather than a longer rollout of the same one.

### Rolling out past the training context

`--n-ctx-override` builds the model with a longer context
(`pfn_transformerlens.checkpointing.load_checkpoint`). Attention masks and rotary tables are
deterministic in the context length and are rebuilt. The positional encodings then behave as
follows past the trained length:

| Encoding | Past the trained context |
| --- | --- |
| `none` | no positional signal, so the forward pass is defined at any length |
| `rope` | rotary angles computed for the new positions, which the model never trained on |
| `learned` | the embedding table has no rows there; it is extended by repeating the last trained position of each token slot: the final row for BAU, and for LR's interleaved x, y tokens the final x-slot row for x tokens and the final y-slot row for y tokens |

Every trained row is kept, so on sequences within the trained context the extended model's
outputs are identical to the original's (tested for all three encodings, sequence and pair
models). The context used: 1024 positions for BAU (BOS + 32 + 500 = 533), 1064 tokens for LR
(2 × (32 + 500)). What `rope` and `learned` models do at positions they never saw is part of
what these runs measure; the in-context N = 223 runs of the same models are the reference.

**Markov** has its own transformer (`markov/model.py`), so it takes `--seq-len-override`
instead. Its rotary tables are non-persistent buffers, absent from the checkpoint and rebuilt
at construction, so a longer context changes no saved weight. `--rope-scaling` then picks what
happens past the trained context:

| Mode | Past the trained context | Cost |
| --- | --- | --- |
| `interpolate` (default, and what `markov_m32_n500` uses) | every position is rescaled by 256/1024, so no rotation angle leaves the trained range | the trained positions move too, so N ≤ 223 does **not** reproduce `markov_m32_ctx256` |
| `extrapolate` | positions stay as trained; the table is simply longer | the first 256 positions are bit-identical, but longer rollouts use angles the model never saw |

Interpolation is the default because this model is the one family that reads an explicit
position at every step: unlike BAU's `none` encoding, a longer table alone would extrapolate
RoPE, and a degrading curve could not be told apart from the estimator degrading. The
interpolated run is therefore read against itself across N, not point-by-point against the
N = 223 run.

## Task diversities

The rebuttal promised to repeat the analysis "at a few critical task diversities M, in
particular near the task diversity threshold". The task-diversity groups run it at the last task
diversity on the memorizing side and the first on the generalizing side of each family's
threshold, with every argument of the M = 32 run at the same length and only the checkpoint
changed. M = 32 itself is the m32 groups' run, which the task-diversity aggregates read.

| Family | Task diversities | Encodings | Lengths |
| --- | --- | --- | --- |
| LR | 128, 256 | learned, rope, none | N = 223 and N = 500 |
| BAU | 256, 512 | learned, rope, none | N = 223 and N = 500 |
| Markov | 16, 64, 128 | — | N = 223 |

The thresholds come from the camera-ready task-diversity sweeps (`slurm/analyse.slurm`: 128
in-distribution 32-token prompts, 100 rollouts each, N = 223), read from
`outputs/pos_encoding_comparison/sweep_metrics.csv` for LR and BAU. The distance of the PMC
posterior to each baseline, the closer one in bold:

| Family | Encoding | M | ED mem | ED gen | SW mem | SW gen |
| --- | --- | --- | --- | --- | --- | --- |
| LR | learned | 128 | **0.118** | 0.194 | **0.0419** | 0.0817 |
| LR | learned | 256 | 0.309 | **0.0474** | 0.0921 | **0.0423** |
| LR | rope | 128 | **0.142** | 0.171 | **0.0481** | 0.0768 |
| LR | rope | 256 | 0.325 | **0.049** | 0.0963 | **0.043** |
| LR | none | 128 | **0.151** | 0.182 | **0.0498** | 0.0798 |
| LR | none | 256 | 0.404 | **0.0422** | 0.114 | **0.0417** |
| BAU | learned | 256 | **0.0506** | 0.0547 | **0.0192** | 0.0245 |
| BAU | learned | 512 | 0.0623 | **0.0321** | 0.0228 | **0.0189** |
| BAU | rope | 256 | 0.0559 | **0.0505** | **0.0202** | 0.0234 |
| BAU | rope | 512 | 0.0586 | **0.0324** | 0.0226 | **0.0191** |
| BAU | none | 256 | 0.0556 | **0.0511** | **0.0204** | 0.0237 |
| BAU | none | 512 | 0.0628 | **0.0314** | 0.0229 | **0.019** |

Every LR encoding is closer to the memorizing baseline at M = 128 and to the generalizing one at
M = 256, by both metrics. At BAU M = 256 the two metrics already disagree for `rope` and `none`,
and every encoding is on the generalizing side at M = 512. The Markov sweep has no CSV here, so
its threshold is read off `outputs/markov/figures_sweep/markov_sweep_combined.png`
(in-distribution prompts). Energy distance ranks the generalizing baseline closer at every M,
which is the N = 223 failure described under [Caveats](#caveats), so it does not place a
threshold. Sliced Wasserstein crosses between M = 16 and M = 32, and the prediction-space
symmetrised KL between M = 64 and M = 128. M = 16 and 64 bracket the latent crossover around the
existing M = 32 run, and M = 64 and 128 bracket the prediction-space one.

To change the grid, set `LR_TASK_DIVERSITIES`, `BAU_TASK_DIVERSITIES` or
`MARKOV_TASK_DIVERSITIES`. Every value must be one `slurm/train.slurm` trains (LR 1–65536, BAU
1–4096, Markov 4–2048, powers of two), and M = 32 is refused.

### Checkpoints on scratch

Training kept every checkpoint (13 GB per LR run), so checkpoints can sit in M3 scratch space
rather than in the repository. The runs need only the final one of each model
(44 MB for LR, about 1 MB for BAU and Markov), plus, for Markov, the run's `resolved_config.yaml`
and chain pool. `CHECKPOINT_SOURCE` names a directory laid out like the repository root:
the preflight looks for each input in the repository first and then at the same relative path
under `CHECKPOINT_SOURCE`, and copies what it finds there into the repository before submitting
anything. The runs then read, hash and keep the repository copies, as the M = 32 runs do, and a
scratch purge cannot take an input away. A file already in the repository is never overwritten.
The copies are listed in the submission's `staged_inputs.txt`.

On M3 nothing needed copying: the dry run of 16 Sep 2026 found every task-diversity input,
Markov configs and chain pools included, at its repository path under `/projects/qh36/bft-pmc`.
The check follows symlinks, so a repository path that links into scratch counts as present and
is not copied.

## Launching

```bash
bash scripts/sync_to_m3.sh                          # on the Mac: send the code to /projects/qh36/bft-pmc
```

`sync_to_m3.sh` sends only files uncommitted in the clone. `analysis/checkpoints.py` is
committed on `camera-ready` but older on M3, so send it once as well:

```bash
rsync -avR analysis/checkpoints.py haotongm@m3.massive.org.au:/projects/qh36/bft-pmc/
```

On an M3 login node, from `/projects/qh36/bft-pmc`:

```bash
bash slurm/submit_pmc_diagnostics.sh --dry-run      # print the 88 sbatch commands, submit nothing
bash slurm/submit_pmc_diagnostics.sh                # submit all eight groups
bash slurm/submit_pmc_diagnostics.sh --only lr-m32,bau-m32
```

The task-diversity runs alone (51 jobs). Their inputs are all at their repository paths on M3;
add `CHECKPOINT_SOURCE=<scratch copy of the repository root>` in front only if the dry run reports
one missing:

```bash
bash slurm/submit_pmc_diagnostics.sh --dry-run --only lr-taskdiv,bau-taskdiv,markov-taskdiv
bash slurm/submit_pmc_diagnostics.sh --only lr-taskdiv,bau-taskdiv,markov-taskdiv
```

- **Groups:** `lr-m32`, `bau-m32`, `markov-m32`, `bau-dim`, `markov-dim`, `lr-taskdiv`,
  `bau-taskdiv`, `markov-taskdiv`. `slurm/submit_markov_dim_sweep.sh` is `--only markov-dim`.
- **Jobs:** 7 training, 50 diagnostics (23 at M = 32 and in the dimension sweeps, 27 at other
  task diversities), 16 aggregates (3 against dimension, 13 against task diversity) and 15
  jellyfish (4 and 11).
- **Jellyfish jobs:** submitted after the groups, each waiting on the runs it draws.
- **bau-dim needs bau-m32:** its vocabulary-12 point is bau-m32's `none` model. Submit both
  together, or bau-m32 first; otherwise the preflight check stops the submission.
- **Task-diversity groups need their m32 group:** each aggregate reads M = 32 from the m32
  group's run at the same encoding and length. Those runs exist from the 16 Sep 2026 submission;
  submitted together, the aggregates wait on them.
- **Preflight:** checkpoints the script does not train must exist, in the repository or under
  `CHECKPOINT_SOURCE`, or nothing is submitted.
- **Training skip:** a model whose final checkpoint already exists is not retrained.
- **Submission record:** see [Reproducing the experiments](#reproducing-the-experiments).
- **Environment overrides:** `OUTPUT_ROOT` (default `outputs/pmc_diagnostics`),
  `CHECKPOINT_SOURCE`, and the three task-diversity grids.

## Runs

N counts generated steps after the prompt (tokens for BAU, (x, y) pairs for LR, states for
Markov); the length grid is where the recovery and drift curves are read. Output directories
are under `outputs/pmc_diagnostics/`.

| Output directory | Group | Checkpoint | Context | Prompt | N | Length grid |
| --- | --- | --- | --- | --- | --- | --- |
| `lr_m32_{learned,rope,none}` | lr-m32 | `checkpoints/lr/task_diversity_{pos}/M32/checkpoint_step_150000.pt` | 512 tokens | 32 | 223 | 8 16 32 64 128 223 |
| `lr_m32_{learned,rope,none}_n500` | lr-m32 | same | 512 extended to 1064 | 32 | 500 | 8 16 32 64 128 223 256 384 500 |
| `bau_m32_{learned,rope,none}` | bau-m32 | `checkpoints/bau/task_diversity_{pos}/M32/checkpoint_step_100000.pt` | 256 | 32 | 223 | 8 16 32 64 128 223 |
| `bau_m32_{learned,rope,none}_n500` | bau-m32 | same | 256 extended to 1024 | 32 | 500 | 8 16 32 64 128 223 256 384 500 |
| `markov_m32_ctx256` | markov-m32 | `checkpoints/markov/task_diversity/markov_M32/checkpoint_step_100000.pt`, config `outputs/markov/training/markov_M32/resolved_config.yaml` | 256 | 32 | 223 | 8 16 32 64 128 223 |
| `bau_vocab{6,24,48}_none` | bau-dim | `checkpoints/bau/vocab_sweep_none/vocab{V}/checkpoint_step_100000.pt` (trained here) | 256 | 32 | 223 | 8 16 32 64 128 223 |
| `bau_vocab{6,24,48}_none_n500` | bau-dim | same | 256 extended to 1024 | 32 | 500 | 8 16 32 64 128 223 256 384 500 |
| `markov_k{3,4,6,8}` | markov-dim | `checkpoints/markov_dim_sweep/k{K}/checkpoint_step_100000.pt` (trained here), config `outputs/markov_dim_sweep/k{K}/resolved_config.yaml` | 256 | 32 | 223 | 8 16 32 64 128 223 |
| `lr_m{128,256}_{learned,rope,none}` | lr-taskdiv | `checkpoints/lr/task_diversity_{pos}/M{M}/checkpoint_step_150000.pt` | 512 tokens | 32 | 223 | 8 16 32 64 128 223 |
| `lr_m{128,256}_{learned,rope,none}_n500` | lr-taskdiv | same | 512 extended to 1064 | 32 | 500 | 8 16 32 64 128 223 256 384 500 |
| `bau_m{256,512}_{learned,rope,none}` | bau-taskdiv | `checkpoints/bau/task_diversity_{pos}/M{M}/checkpoint_step_100000.pt` | 256 | 32 | 223 | 8 16 32 64 128 223 |
| `bau_m{256,512}_{learned,rope,none}_n500` | bau-taskdiv | same | 256 extended to 1024 | 32 | 500 | 8 16 32 64 128 223 256 384 500 |
| `markov_m{16,64,128}_ctx256` | markov-taskdiv | `checkpoints/markov/task_diversity/markov_M{M}/checkpoint_step_100000.pt`, config `outputs/markov/training/markov_M{M}/resolved_config.yaml` | 256 | 32 | 223 | 8 16 32 64 128 223 |

Each script checks that its rollouts fit before generating anything: prompt + N ≤ n_ctx for
BAU, prompt + N ≤ n_ctx / 2 pairs for LR, prompt + N ≤ seq_len for Markov.

Dimension aggregates (`eval.py recovery-vs-dimension`), read at the largest rollout length
their inputs share:

| Output directory | Inputs (latent dimension = run) | Read at |
| --- | --- | --- |
| `bau_dimension_sweep` | 5 = `bau_vocab6_none`, 11 = `bau_m32_none`, 23 = `bau_vocab24_none`, 47 = `bau_vocab48_none` | N = 223 |
| `bau_dimension_sweep_n500` | the same four models' N = 500 runs | N = 500 |
| `markov_dimension_sweep` | 6 = `markov_k3`, 12 = `markov_k4`, 30 = `markov_k6`, 56 = `markov_k8` | N = 223 |

Latent dimension is the number of free parameters: vocabulary − 1 for BAU, k(k − 1) for Markov.

Task-diversity aggregates (`eval.py recovery-vs-dimension --x-axis task-diversity`), one per
family, encoding and length, each reading its M = 32 point from the m32 group's run at the same
encoding and length:

| Output directory | Inputs (task diversity = run) | Read at |
| --- | --- | --- |
| `lr_task_diversity_{pos}` | 32 = `lr_m32_{pos}`, 128 = `lr_m128_{pos}`, 256 = `lr_m256_{pos}` | N = 223 |
| `lr_task_diversity_{pos}_n500` | the same models' N = 500 runs | N = 500 |
| `bau_task_diversity_{pos}` | 32 = `bau_m32_{pos}`, 256 = `bau_m256_{pos}`, 512 = `bau_m512_{pos}` | N = 223 |
| `bau_task_diversity_{pos}_n500` | the same models' N = 500 runs | N = 500 |
| `markov_task_diversity` | 16, 32, 64, 128 = `markov_m{M}_ctx256` | N = 223 |

## Settings shared by every run

| Setting | LR | BAU | Markov |
| --- | --- | --- | --- |
| Rollouts R | 1024 | 1024 | 1024 |
| Prompt sources | `prior` (no prompt) and `posterior` (one in-distribution prompt) | same | same |
| Posterior prompt | x ~ N(0, I); y from one pool task, noise sd 0.5 | tokens from one urn of the task pool | states from one training chain |
| Rollout sampler | `predictive_monte_carlo_beta_chunked`: x ~ N(0, 1) per coordinate, y from the model | model tokens | model states |
| Estimator θ̂ | least squares on prompt + rollout | token frequencies | transition counts, Laplace smoothing 1.0 |
| Generalizing baseline | N(0, I) prior, or the Gaussian ridge posterior | Dirichlet(1) prior or posterior | Dirichlet(1) rows, prior or posterior |
| Memorizing baseline | uniform over the pool, or its posterior | uniform over the pool, or its posterior | uniform over the pool, or its posterior |
| Prior drift anchor | each rollout's own estimate at N = 16 | same | same |
| Posterior drift anchor | prompt-only estimate | same | same |
| Drift scale d | 8 (the weight dimension) | V − 1 | k(k − 1) |
| Rollout-count grid | 16 32 64 128 256 512 1024, 50 resamples (1 at R = 1024) | same | same |
| Sliced-Wasserstein projections | 100 (fixed projection seed 42) | same | same |
| Jellyfish realisations | 20 prompts, 128 rollouts each | same | same |
| Temperature · seed | 1.0 · 42 | 1.0 · 42 | 1.0 · 42 |
| Rollout chunk size | 128; 64 at N = 500 | 256 | 128 |
| Hardware | GPU, 4 CPUs, 32 GB, 4 h; 12 h at N = 500 | GPU, 4 CPUs, 24 GB, 4 h | CPU, 8 CPUs, 24 GB, 12 h |

## What each diagnostic computes

- **Recovery error vs rollout length.** For each grid N, the estimate from the prompt plus the
  first N generated steps of each rollout forms an R-sample set. Energy distance and sliced
  Wasserstein-1 compare it with R draws from the generalizing and memorizing baselines.
- **Drift vs length** (`plot_drift`, the `pmc-diagnostics` convention). For each rollout,
  (1/d)·‖θ̂_N − anchor‖₁. The figure shows up to 200 per-rollout traces, their mean, and the
  10–90% band, on a log₂ axis.
- **Recovery error vs rollout count.** At the final N, 50 random subsets of R rollouts per R
  give a mean and standard deviation of each metric. The Monte Carlo floor applies the same
  metrics to a fresh reference draw of size R, so it is what a perfect model would score.
- **Jellyfish** (`analysis/path_stability.py`, Ng et al., TabMGP, Figures 3 and 8). The expected
  scaled L1 norm E over F_N of (1/p)·‖θ(F_n) − θ(F_N)‖₁ against N − n, where z_{1:n} is one
  realisation of the observed data, θ(F_n) the estimator on that prompt, θ(F_N) the same
  estimator on the prompt plus the first N − n generated steps, and the expectation is the mean
  over that realisation's 128 rollouts. Each run draws **20 realisations**, each its own prompt
  from its own task, and the figures draw one curve per realisation, as in Figure 8. Curves are
  evaluated at every step from 0 (where they are exactly 0) to the run's N, with p = dim θ: 8 for
  LR, V for BAU, k² for Markov (the flattened transition matrix, Laplace-smoothed as in the
  Markov PMC estimator).

## Jellyfish jobs

`slurm/pmc_jellyfish.slurm` runs `eval.py path-stability --stride 1 --realisations 20` on the
`<run>_rollouts.npz` files the diagnostics write, so each run contributes 20 curves, one per
realisation of the observed data. Labels name the family and positional encoding.

| Output directory | Waits on | Inputs |
| --- | --- | --- |
| `jellyfish_m32` | the N = 223 runs of lr-m32 and bau-m32, `markov_m32_ctx256` | `lr-{pos}`, `bau-{pos}`, `markov` |
| `jellyfish_m32_n500` | the N = 500 runs of lr-m32 and bau-m32 | `lr-{pos}`, `bau-{pos}` |
| `jellyfish_dimension` | the N = 223 runs of bau-dim (with `bau_m32_none`) and markov-dim | `bau-none` ×4, `markov` ×4 |
| `jellyfish_dimension_n500` | the N = 500 runs of bau-dim (with `bau_m32_none_n500`) | `bau-none` ×4 |
| `jellyfish_lr_m{128,256}` | the N = 223 runs of lr-taskdiv at that M | `lr-{pos}` |
| `jellyfish_lr_m{128,256}_n500` | the N = 500 runs of lr-taskdiv at that M | `lr-{pos}` |
| `jellyfish_bau_m{256,512}` | the N = 223 runs of bau-taskdiv at that M | `bau-{pos}` |
| `jellyfish_bau_m{256,512}_n500` | the N = 500 runs of bau-taskdiv at that M | `bau-{pos}` |
| `jellyfish_markov_m{16,64,128}` | the markov-taskdiv run at that M | `markov` |

The task-diversity jellyfish are drawn per family, since each family has its own task
diversities; the M = 32 panels are in `jellyfish_m32` and `jellyfish_m32_n500`.

Each writes `path_stability.csv` (the curves), `path_stability_per_prompt.csv`,
`theta_reference.npz` (θ(F_n) and θ(F_N) for every run), `config.json`, and the figures: one per
positional encoding with one panel per family (`path_stability_learned.png` has the `lr` and
`bau` panels of the camera-ready figure), `path_stability_by_family.png` coloured by encoding,
and the single-panel `path_stability.png`, each also as PDF.

## Training

The script trains only the models the dimension sweeps add. Each command is the camera-ready
one from `slurm/train.slurm`, with only the swept quantity changed. Both run on a GPU with 8
CPUs, 64 GB and 12 h, as in `train.slurm`.

**BAU vocabulary sweep** (`slurm/train_bau_vocab.slurm`), the `task_diversity_none/M32`
command with `--vocab-size {6,24,48}`:

```bash
python train.py bau --num-tasks 32 --pos-encoding none --vocab-size {6,24,48} --alpha-value 1.0 \
  --d-model 64 --d-mlp 256 --n-layers 2 --n-heads 1 --d-head 64 --n-ctx 256 --act-fn gelu \
  --batch-size 64 --seq-len 256 --num-steps 100000 --learning-rate 5e-4 --warmup-steps 0 \
  --save-checkpoint --checkpoint-schedule linear --save-every 500 \
  --checkpoint-root checkpoints/bau/vocab_sweep_none/vocab{V} \
  --eval-every 1000 --eval-batches 10 --seed 42 --no-use-wandb \
  --log-file checkpoints/bau/vocab_sweep_none/logs/vocab_{V}.json
```

**1-Markov sweep** (`slurm/train_markov_dim.slurm`), the `markov_M32` command with `--k {3,4,6,8}`:

```bash
python train.py markov --config-path markov/train.yaml --k {3,4,6,8} --n-chains 32 \
  --run-name k{K} --checkpoint-root checkpoints/markov_dim_sweep --output-root outputs/markov_dim_sweep \
  --pmc-num-samples 100 --pmc-prompt-len 32 --pmc-generation-length 223 --no-use-wandb
```

Everything else comes from `markov/train.yaml`, as for the camera-ready Markov models:
seq_len 256, batch size 128, learning rate 6e-4, 100000 steps, and the default seed 42.

## What each run saves

Each diagnostics directory holds, for each prompt source (`prior`, `posterior`):

| File | Contents |
| --- | --- |
| `{source}_trajectories.npz` | every rollout including the prompt (BAU/Markov int16 R × (prompt + N); LR float32 x R × (prompt + N) × 8 and y), the prompt, the urn or task index (−1 for the prior), checkpoint path |
| `{source}_baselines.npz` | the R reference samples of both baselines, the task pool, α or the noise sd, the prompt; for LR also the ridge posterior mean and covariance and the memorizing posterior weights |
| `{source}_recovery_vs_length.csv` | ED and SW against both baselines at each grid N |
| `{source}_drift_vs_length.csv` | drift mean and 10/50/90% quantiles at each grid N |
| `{source}_drift_paths.npz` | per-rollout drift (R × grid), the running estimates at every grid N, the anchor |
| `{source}_recovery_vs_count.csv` | mean and sd per R, for the model and the Monte Carlo floor |
| `{source}_recovery_vs_count_resamples.csv` | every draw behind those means: R, resample, model or floor, metric, value |
| `{source}_*.png`, `.pdf` | the three figures |

and once per run:

- `<run>_rollouts.npz`: the jellyfish rollouts in the format `analysis/path_stability.py`
  reads — 20 realisations × 128 rollouts × (prompt + N) — with the prompt length, the task-pool
  size, the checkpoint step and the urn or task behind each realisation. These are drawn after
  every other diagnostic, so they leave the rollouts above untouched.
- `run_record.json`: every CLI setting, the command, host, SLURM job and node, the submission
  record it belongs to, Python version, every installed package and its version, the platform,
  torch, CUDA and cuDNN versions and GPU names, git commit and uncommitted files, the SHA-256 of
  every file the run read, and what the run found at load time (family, device, dimension, drift
  scale, task-pool size, model context, positional encoding, the checkpoint's training metadata).
  LR and BAU runs read only the checkpoint, which holds the task pool. Markov runs also read the
  `resolved_config.yaml` and the chain pool (`transition_matrices.npy` and
  `stationary_distributions.npy`, or `pmc_samples.npz`) that the loader finds, and all of these
  are hashed.
- `figures.json`: each figure's kind, title and the files it is drawn from.

Arrays are saved as compressed NPZ files. Before compression, a BAU or Markov run holds at
most about 20 MB of arrays and an LR run about 55 MB (110 MB at N = 500), the jellyfish
realisations included. The 23 M = 32 and dimension-sweep runs together come to well under 1 GB,
and the 27 task-diversity runs to about as much again. The aggregates save
`recovery_vs_dimension.csv` or `recovery_vs_task_diversity.csv`, its figure, `run_record.json`
(including the input directories and the checksums of the CSVs read) and `figures.json`.

## Reproducing the experiments

Every submission saves, under `outputs/pmc_diagnostics/submissions/<timestamp>/`:

| File | Contents |
| --- | --- |
| `code.tar.gz`, `source_files.txt` | every source file as it was on disk at submission, tracked or not |
| `git_commit.txt`, `git_status.txt`, `git_diff.patch` | the commit and the uncommitted changes on top of it |
| `environment.txt` | Python, platform, torch, CUDA and cuDNN versions, and every installed package |
| `inputs_sha256.txt` | checksums of the checkpoints and configs the runs read |
| `staged_inputs.txt` | each input copied from `CHECKPOINT_SOURCE`, as `<source> -> <repository path>` (only when something was copied) |
| `submission.txt` | time, host, user, directory, groups, environment overrides and the task-diversity grids |
| `sbatch_commands.txt` | every sbatch command with its job id |
| `logs/` | every job's stdout and stderr |

With the run records, the training commands above (seeds included) and the checkpoints, this
is everything needed to rerun any experiment from scratch: unpack `code.tar.gz`, install the
packages in `environment.txt` (or `uv sync` with the `uv.lock` inside the tarball), and run the
recorded commands. Every input is inside the repository, so nothing depends on another user's
scratch space, or on scratch at all: the task-diversity checkpoints are copied in from scratch
before submission (see [Checkpoints on scratch](#checkpoints-on-scratch)).

The trajectories and baselines are enough to recompute every curve without a checkpoint, and
the CSV and NPZ files are enough to redraw every figure. Rollouts are drawn on the GPU, so a
rerun on different hardware reproduces the distributions but not necessarily the same draws.

## Pulling and redrawing

```bash
bash scripts/pull_pmc_diagnostics_from_m3.sh                      # on the Mac: outputs and submission records
bash scripts/pull_pmc_diagnostics_from_m3.sh --with-checkpoints   # also every input file the run records name
uv run eval.py replot-pmc-diagnostics --roots outputs/pmc_diagnostics
```

`replot-pmc-diagnostics` reads every `figures.json` and draws each figure again through the
same plotting code; on a local test run, all 31 redrawn figures were pixel-identical to the
originals. The jellyfish figures are recomputed from the pulled rollouts in seconds with the
same `eval.py path-stability` command the jellyfish jobs run.

## Caveats

- **Positions never trained on.** The N = 500 `rope` and `learned` runs use rotary angles, or
  repeated positional embeddings, beyond anything the models saw in training; `none` models
  have no positional signal. Captions should say which encoding a curve comes from, and the
  N = 223 run of the same model is the in-context reference.
- **Markov ED at k = 10 within 256 tokens.** This is exactly the `markov_m32_ctx256` setting.
  Rollouts from an exact Bayesian memorizer, with a 32-state prompt and N = 223, score 0.445
  against the memorizing reference and 0.359 against the generalizing one, so energy distance
  picks the wrong reference; sliced Wasserstein ranks them correctly. A Markov model with a
  256-state context cannot roll out further, so read sliced Wasserstein for this run.
- **Markov dimension-sweep cap.** k stops at 8 because the ranking check passed for k ≤ 9 at 256
  states (then an 8-state prompt and N = 248). The camera-ready setting reads the final estimate
  from 255 states; the check was not re-run for it.
- **Two scalings of the same distance.** Drift follows `pmc-diagnostics` (d = V − 1 or
  k(k − 1)); the jellyfish follows `path_stability.py` (p = V or k²). For LR both are 8.
- **Markov energy distance at every task diversity.** In the camera-ready Markov sweep, energy
  distance puts the generalizing baseline closer at every M, down to M = 4, where the
  prediction-space KL and sliced Wasserstein both put the model on the memorizing side. Read the
  Markov task-diversity runs by sliced Wasserstein too.
- **One prompt near the threshold.** The thresholds were placed with 128 prompts and 100
  rollouts each; every diagnostics run conditions its posterior on one in-distribution prompt.
  Close to a threshold (BAU M = 256, where the sweep's two metrics already disagree) that one
  prompt can land on either side, so a run's ranking at the final length is not by itself a
  statement about the threshold. What these runs measure is how the recovery error, its
  spread over rollout counts and the drift settle at each M.
