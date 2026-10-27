# Running the experiments on SLURM

The job scripts behind every run in the paper. Submit everything from the
**repository root**; each job calls `python train.py` or `python eval.py` from
the activated `.venv`. Steps 0 to 4 below are the task-diversity pipeline,
step 5 the reliability diagnostics of Appendix D, and step 6 the remaining
appendix experiments.

| Group | Files | What they do |
| --- | --- | --- |
| Task-diversity pipeline | `train.slurm`, `analyse.slurm`, `path_stability.slurm`, `submit_all.sh` | train every family as a job array (1 GPU per task), evaluate the checkpoints (1 GPU per task), draw the path-stability plot (CPU) |
| Reliability diagnostics | `submit_pmc_diagnostics.sh` and the jobs it submits: `{lr,bau,markov}_rollout_diagnostics.slurm`, `lr_rollout_diagnostics_cpu.slurm`, `pmc_jellyfish.slurm`, `aggregate_dimension_sweep.slurm`, `train_bau_vocab.slurm`, `train_markov_dim.slurm`; follow-ups `submit_markov_dim_sweep.sh`, `submit_markov_n500.sh`, `submit_count_r8192.sh` | recovery error against rollout length and count, drift, jellyfish plots and dimension sweeps; [`PMC_DIAGNOSTICS.md`](PMC_DIAGNOSTICS.md) records every run |
| Other appendix experiments | `retrain_transient.slurm`, `transient_eval.slurm`, `submit_prior_shape.sh`, `prior_shape.slurm`, `predictive_fidelity.slurm` | dense-early retraining for the training-dynamics figures, the Gaussian-against-Laplace prior test, next-token fidelity |
| Historical | `rebuttal/` | the rebuttal-period jobs, kept unchanged as a record; see its README |

Every script explains itself in its header comment: why it exists, its
settings, and what it costs.

## 0. One-time setup (login node)

```bash
cd /path/to/bft-pmc
uv sync                 # creates .venv; the job scripts activate it directly
mkdir -p logs           # SLURM will NOT create the log directory for you
uv run pytest -q        # optional sanity check
```

The scripts are configured for Monash M3: GPU jobs run on `--partition=gpu`
(A100/A40/T4, no QoS required) and the CPU-only diagnostic on `--partition=comp`.
M3 uses partition + QoS rather than `--account`, so no account flag is set.
Check `mon_qos` if you have access to a restricted partition (e.g. FIT's
`--partition=fit --qos=fitq`) and want to use it instead. No `module load cuda`
is needed: the PyTorch wheel ships its own CUDA runtime.

LR is the one job that cares which GPU it lands on. `--partition=gpu` mixes
A100s, A40s and T4s; a T4 will be several times slower and may not fit the
24-hour walltime. Pin it at submit time, which overrides the script directive:

```bash
sbatch --job-name=bft-lr --array=0-50%8 --time=24:00:00 \
  --gres=gpu:A100:1 --export=ALL,FAMILY=lr slurm/train.slurm
```

The scripts call `python` from the activated venv rather than `uv run`, so 50
concurrent array tasks never contend on the uv lock.

## 1. Calibrate the walltime

The `--time` values in the submit commands below are placeholders. Measure one
task first:

```bash
srun --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:30:00 --pty bash
source .venv/bin/activate
python train.py lr --num-tasks 32 --num-steps 200 --n-ctx 512 --seq-len 256 \
  --batch-size 256 --no-use-wandb --checkpoint-root /tmp/cal
```

Read the it/s from the tqdm bar and scale: LR is 150 000 steps, BAU 100 000,
Markov 100 000. Add ~25% headroom. **There is no resume support in the LR/BAU
CLIs** — a job that hits the walltime restarts from step 0, so overestimate.

If LR runs out of GPU memory, halve `--batch-size` to 128 (this departs from the
paper configuration; note it if you do).

## 2. Train

```bash
sbatch --job-name=bft-lr     --array=0-50%8 --time=24:00:00 --export=ALL,FAMILY=lr     slurm/train.slurm
sbatch --job-name=bft-bau    --array=0-38%8 --time=12:00:00 --export=ALL,FAMILY=bau    slurm/train.slurm
sbatch --job-name=bft-markov --array=0-9%5  --time=12:00:00 --export=ALL,FAMILY=markov slurm/train.slurm
```

`%8` caps concurrent tasks; lower it if your QoS limits running jobs. The array
index maps to (task count × positional encoding) — index 0-16 is `learned`,
17-33 `rope`, 34-50 `none` for LR. Each run gets its own checkpoint directory,
`checkpoints/<family>/task_diversity_<pos>/M<count>/`.

## 3. Evaluate

Note each training job's ID from `sbatch`, then chain:

```bash
sbatch --job-name=bft-lr-eval  --array=0-2 --time=12:00:00 --dependency=afterok:<LR_JOBID>     --export=ALL,FAMILY=lr     slurm/analyse.slurm
sbatch --job-name=bft-bau-eval --array=0-2 --time=08:00:00 --dependency=afterok:<BAU_JOBID>    --export=ALL,FAMILY=bau    slurm/analyse.slurm
sbatch --job-name=bft-mk-eval  --array=0-0 --time=24:00:00 --dependency=afterok:<MARKOV_JOBID> --export=ALL,FAMILY=markov slurm/analyse.slurm
```

`afterok:<jobid>` on an array waits for **every** task to succeed; one failed
task blocks the dependent job (it stays `DependencyNeverSatisfied` — cancel it,
fix the run, resubmit).

## 4. Jellyfish plot

```bash
sbatch --dependency=afterok:<LR_EVAL>:<BAU_EVAL>:<MK_EVAL> slurm/path_stability.slurm
```

Writes `outputs/path_stability/`: `path_stability.csv`,
`path_stability_per_prompt.csv`, `theta_reference.npz`, and the figure.

## 5. Rollout diagnostics

Recovery error vs rollout length and count, drift vs length and the jellyfish plot for
linear regression, balls and urns and Markov, plus latent-dimension sweeps for balls and urns
and Markov, all with 1024 rollouts and every model in the configuration of step 2. Linear
regression and balls and urns also roll out past their training context, to N=500. Independent
of steps 3–4; needs the M=32 checkpoints from step 2:

```bash
bash slurm/submit_pmc_diagnostics.sh --dry-run
bash slurm/submit_pmc_diagnostics.sh
```

The same diagnostics at task diversities either side of each family's threshold need other
checkpoints from step 2. When those sit in scratch space, point `CHECKPOINT_SOURCE` at the
scratch directory laid out like the repository root; the final checkpoints are copied into the
repository before anything is submitted:

```bash
CHECKPOINT_SOURCE=/path/to/scratch/bft-pmc \
  bash slurm/submit_pmc_diagnostics.sh --only lr-taskdiv,bau-taskdiv,markov-taskdiv
```

[`PMC_DIAGNOSTICS.md`](PMC_DIAGNOSTICS.md) records every run's settings, what it and each
submission save, and how to reproduce the runs or redraw their figures.

Three follow-up submitters extend the batch to the settings the appendix finally plots, each
explained in its header: `submit_markov_n500.sh` (the Markov dimension and task-diversity runs
at N=500, by rotary position interpolation), `submit_count_r8192.sh` (the three M=32 count
curves with an 8192-rollout pool) and `submit_markov_dim_sweep.sh` (the Markov dimension group
alone).

## 6. Other appendix experiments

```bash
# Training dynamics with checkpoints dense early in training (the sweep's checkpoints start at
# step 500, after the balls-and-urns transient): retrain the M=32 runs, then evaluate every
# checkpoint. Array indices 0-2 are LR x {learned, rope, none}, 3-5 balls-and-urns.
sbatch --array=0-5 slurm/retrain_transient.slurm
sbatch --array=0-5 --dependency=afterok:<RETRAIN_JOBID> slurm/transient_eval.slurm

# Gaussian against variance-matched Laplace prior: R=8192 prior rollouts per task diversity,
# then the analysis of the saved rollouts (CPU, no checkpoint).
bash slurm/submit_prior_shape.sh
sbatch --dependency=afterok:<PRIOR_JOBIDS> slurm/prior_shape.slurm

# Next-token fidelity of the balls-and-urns and Markov M=32 models, to position 500.
sbatch slurm/predictive_fidelity.slurm
```

## Disk

**Checkpoints dominate.** LR and BAU save one checkpoint per 500 steps
(`--checkpoint-schedule linear --save-every 500`); Markov saves one per
evaluation, i.e. every 200 steps. Each checkpoint stores Adam's two moment
buffers alongside the weights, so it is ~3x the model size:

| Family | Params | Per checkpoint | Per run | Runs | Total |
| --- | --- | --- | --- | --- | --- |
| LR | 3.69 M | 44 MB | 300 ckpts = 13.3 GB | 51 | **678 GB** |
| BAU | 0.12 M | 1.4 MB | 200 ckpts = 0.3 GB | 39 | 11 GB |
| Markov | 0.10 M | 1.2 MB | 502 ckpts = 0.6 GB | 10 | 6 GB |

Raise `--save-every` to thin LR further: 2000 gives 75 checkpoints per run and
170 GB total, 150000 keeps only the final checkpoint (2 GB total, enough for the
task-diversity sweep and the jellyfish plot but not for training dynamics).

Rollout bundles are the next largest output, at the configured budgets
(128 prompts × 100 rollouts, 1024 prior rollouts, N = 255):

| Family | Per bundle | Total |
| --- | --- | --- |
| LR (float32 x and y) | ~117 MB posterior | ~6 GB per encoding, **~18 GB** for all three |
| BAU (int16 tokens) | ~6.5 MB | < 1 GB for all three |
| Markov (int16 states, **one bundle per checkpoint**) | ~6.5 MB | **~25 GB compressed** |

Markov dominates because `markov-sweep` evaluates every checkpoint (one per 200
steps). Check your quota before submitting. To cut it, pass `--no-save-rollouts`
to the Markov analysis — but then the diagnostic has no Markov curves.

## Monitoring

```bash
squeue -u $USER                        # queue state
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode   # after the fact
tail -f logs/bft-lr_<jobid>_0.out      # live log for array task 0
scancel <jobid>                        # whole array
scancel <jobid>_5                      # one array task
```

Common failures:

- *Job never starts, `DependencyNeverSatisfied`* — an upstream array task failed.
- *`No such file or directory: logs/...`* — you skipped `mkdir -p logs`.
- *`No checkpoints found in ...`* — the analysis ran against the per-run
  directory instead of the sweep root. The analysis root is
  `task_diversity_<pos>`, not `task_diversity_<pos>/M32`.
- *CUDA OOM in LR* — lower `--batch-size`.
