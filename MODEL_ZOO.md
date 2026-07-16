# Model Zoo

This document records the checkpoint layout used by the experiment and
evaluation code, together with the frozen paper artifacts included under
`paper_data/`.

## Included artifacts

| Artifact | Location | Use case |
| --- | --- | --- |
| Linear-regression sweep metrics | `paper_data/lr/sweep/metrics.csv` | Replot the main task-diversity sweep |
| Linear-regression dynamics metrics | `paper_data/lr/dynamics/metrics.csv` | Replot the TD=32 training-dynamics figure |
| Balls-and-urns sweep metrics | `paper_data/bau/sweep/metrics.csv` | Replot the main BAU task-diversity sweep |
| Balls-and-urns dynamics metrics | `paper_data/bau/dynamics/metrics.csv` | Replot the BAU TD=32 dynamics figure |
| Markov sweep distance metrics | `paper_data/markov/sweep/metrics.csv` | Replot the main Markov sweep (ED/SW vs M, plus posterior rows) |
| Markov sweep KL histories | `paper_data/markov/sweep/runs/*/wandb_kl_history.csv` | Replot the full Markov sweep with KL + ED + SW |
| Markov sweep prior-only metrics | `paper_data/markov/sweep/metrics_prior_n1024.csv` | Replot the higher-sample prior-only Markov sweep slice |
| Markov sweep prior predictive KL | `paper_data/markov/sweep/prior_predictive_kl.csv` | Replot the prior-predictive KL curve used with the prior-only sweep |
| Markov M=32 dynamics metrics | `paper_data/markov/dynamics_m32/runs/*/metrics.csv` | Replot the M=32 Markov training-time dynamics figure |
| Markov M=32 dynamics KL history | `paper_data/markov/dynamics_m32/runs/*/wandb_kl_history.csv` | Replot the KL panels for the bundled M=32 Markov dynamics run |
| Markov M=8 dynamics metrics | `paper_data/markov/dynamics_m8/runs/*/metrics.csv` | Replot the M=8 Markov training-time dynamics figure |
| Markov M=8 dynamics KL history | `paper_data/markov/dynamics_m8/runs/*/wandb_kl_history.csv` | Replot the KL panels for the bundled M=8 run |
| Evaluation datasets | `paper_data/*/eval_datasets/` | Reuse the frozen evaluation prompts/tokens |

## Expected checkpoint layout

The experiment code expects the following layout:

| Setting | Checkpoint root | Primary consumer |
| --- | --- | --- |
| Linear regression | `checkpoints/lr/task_diversity/<run_id>/checkpoint_step_150000.pt` | `uv run eval.py lr-sweep ...` |
| Balls and urns | `checkpoints/bau/task_diversity/<run_id>/checkpoint_step_100000.pt` | `uv run eval.py bau-sweep ...` |
| Markov task diversity | `checkpoints/markov/task_diversity/<run_name>/checkpoint_step_*.pt` | `uv run eval.py markov-sweep ...` |
| Markov threshold | `checkpoints/markov/task_diversity_threshold/<run_name>/checkpoint_step_*.pt` | `uv run eval.py markov-threshold ...` |

## Notes

- `paper_data/` is sufficient to regenerate the aggregate sweep and dynamics
  figures produced by `scripts/reproduce_paper.sh` from the frozen metrics
  CSVs.
- Distribution-marginal figures additionally require PMC sample bundles. The
  repository includes the generation code and documents the commands in
  [`paper_data/README.md`](paper_data/README.md#generate-sample-bundles-for-marginal-plots).
- Checkpoint-based evaluation scripts assume the directory structure listed
  above.
- Checkpoints are not distributed in this repository; the table documents the
  layout produced by the training and sweep commands.
