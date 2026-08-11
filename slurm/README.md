# Running the sweeps on SLURM

Three job files, submitted in order. Everything runs from the **repository root**.

| File | What it does | Resources |
| --- | --- | --- |
| `train.slurm` | Trains one task family as a job array | 1 GPU per task |
| `analyse.slurm` | Evaluates checkpoints, writes metrics + PMC sample/rollout bundles | 1 GPU per task |
| `path_stability.slurm` | Path-stability (jellyfish) diagnostic from saved rollouts | CPU only |

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
