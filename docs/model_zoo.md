# Model zoo: checkpoint layout

Model checkpoints are not distributed with this repository, apart from the
small Beta-Bernoulli model in `paper_data/beta_bernoulli/model.pt`: the full
sweeps run to hundreds of gigabytes (see [`slurm/README.md`](../slurm/README.md#disk)).
This page records where the training commands write checkpoints, so that a
rerun, or checkpoints restored from elsewhere, line up with the evaluation
commands. Paths are relative to the repository root.

## Layout

| Experiment | Written by | Checkpoints | Read by |
| --- | --- | --- | --- |
| Linear-regression task-diversity sweep | `slurm/train.slurm` (`FAMILY=lr`) | `checkpoints/lr/task_diversity_<pos>/M<M>/checkpoint_step_<step>.pt` | `eval.py lr-sweep`, `lr-dynamics`, `lr-rollout-diagnostics` |
| Balls-and-urns task-diversity sweep | `slurm/train.slurm` (`FAMILY=bau`) | `checkpoints/bau/task_diversity_<pos>/M<M>/checkpoint_step_<step>.pt` | `eval.py bau-sweep`, `bau-dynamics`, `bau-rollout-diagnostics` |
| 1-Markov task-diversity sweep | `slurm/train.slurm` (`FAMILY=markov`) | `checkpoints/markov/task_diversity/markov_M<M>/checkpoint_step_<step>.pt` | `eval.py markov-sweep`, `markov-rollout-diagnostics` |
| Transient-generalization retrain (dense early checkpoints, M = 32) | `slurm/retrain_transient.slurm` | `checkpoints/<lr,bau>/transient_td32_<pos>/` | `slurm/transient_eval.slurm` |
| Balls-and-urns vocabulary sweep (V = 6, 24, 48) | `slurm/train_bau_vocab.slurm` | `checkpoints/bau/vocab_sweep_none/vocab<V>/` | `eval.py bau-rollout-diagnostics` |
| 1-Markov state-space sweep (k = 3, 4, 6, 8) | `slurm/train_markov_dim.slurm` | `checkpoints/markov_dim_sweep/k<k>/` | `eval.py markov-rollout-diagnostics` |
| Markov task-diversity threshold | `eval.py markov-threshold` | `checkpoints/markov/task_diversity_threshold/<run>/` | the same command |
| Beta-Bernoulli | `train.py beta-bernoulli` | `outputs/bau/beta_bernoulli/checkpoints/` | the same command, with `--checkpoint-path` |

`<pos>` is the positional encoding (`learned`, `rope` or `none`). The variant is
also stored in the model config inside every linear-regression and
balls-and-urns checkpoint, and `bft_pmc.pos_encoding.positional_encoding_of`
reads it back. The final checkpoint of the task-diversity sweeps is step
150000 for linear regression and 100000 for balls-and-urns and Markov.

## What a checkpoint holds

- **Linear regression and balls-and-urns** (the vendored `pfn_transformerlens`
  trainer): model, optimizer and scheduler state, the model and training
  configs, the training task pool (`task_distribution`) and run metadata.
  `bft_pmc.analysis.checkpoints.load_run_info` rebuilds the model and returns
  the pool. Its `n_ctx_override` argument (`--n-ctx-override` in the
  diagnostics commands) builds the model with a longer context; a learned
  positional table is extended by repeating the last trained position of each
  token slot.
- **1-Markov** (`train.py markov`): model and optimizer state, the step and the
  config. The training chain pool and resolved config are written separately,
  to `outputs/markov/training/<run>/` (`transition_matrices.npy`,
  `stationary_distributions.npy`, `resolved_config.yaml`). Evaluation needs
  them as well as the checkpoint, so keep the two trees together.

## What can be done without checkpoints

Everything that reads saved metrics, sample bundles or rollouts: the aggregate
sweep and dynamics figures from [`paper_data/`](../paper_data/README.md), the
marginal figures from sample bundles, and every reliability figure and replot
from the diagnostics outputs. [`figure_provenance.md`](figure_provenance.md)
lists the inputs of each paper figure, and [`pmc_outputs.md`](pmc_outputs.md)
the files each command writes.
