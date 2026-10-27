# Configuration files

| File | Read by | Contents |
| --- | --- | --- |
| `markov/train.yaml` | `train.py markov` (`--config-path`) | the paper's Markov settings: k = 10, 256 positions, 100k steps, evaluation every 200 steps. Command-line options override single values |
| `markov/task_diversity_threshold.yaml` | `eval.py markov-threshold` | the task-diversity grid and the gap tolerance of the threshold experiment |
| `sweeps/lr_task_diversity_sweep.yaml` | `train.py sweep` | the linear-regression sweep: M = 2⁰…2¹⁶ under each positional encoding |
| `sweeps/bau_task_diversity_sweep.yaml` | `train.py sweep` | the balls-and-urns sweep: M = 2⁰…2¹² under each positional encoding |
| `sweeps/markov_task_diversity_sweep.yaml` | `train.py sweep` | the Markov sweep: M = 2²…2¹¹ |

A sweep file holds a command template with `~NAME~` placeholders, the values
to substitute (the Cartesian product of each parameter group), the GPUs to
use and how many jobs share each. `uv run train.py sweep <file> --dry-run`
prints every command it would launch. The same runs on SLURM are
[`slurm/train.slurm`](../slurm/train.slurm).
