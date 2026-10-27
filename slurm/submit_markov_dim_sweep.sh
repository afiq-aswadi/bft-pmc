#!/bin/bash
# Submit the 1-Markov latent-dimension sweep: four training runs, four
# diagnostics jobs each waiting on its own training run, and a final
# aggregation that waits on all four diagnostics.
#
#   bash slurm/submit_markov_dim_sweep.sh --dry-run   # print, submit nothing
#   bash slurm/submit_markov_dim_sweep.sh             # submit
#
# Run from the repository root, on a login node. This is the markov-dim group of
# slurm/submit_pmc_diagnostics.sh, which holds the settings of every
# diagnostics run; slurm/PMC_DIAGNOSTICS.md records them.
#
# The sweep varies the state space k at fixed task diversity M=32 and at the
# camera-ready context of seq_len 256. The latent task is the transition matrix,
# so the dimension grows quadratically:
#
#   k=3 ->  6 free parameters
#   k=4 -> 12
#   k=6 -> 30
#   k=8 -> 56
#
# which spans the balls-and-urns sweep's 5, 11, 23 and 47.
#
# The 256-state context caps the rollout, now N=223 after a 32-state prompt
# (the camera-ready PMC setting), and that caps k. A rollout visits each of the
# k^2 matrix entries about N/k^2 times, so for large k the Laplace-smoothed
# estimate is mostly its prior. Rollouts drawn from the exact memorizing and
# generalizing posteriors, scored exactly as the diagnostics score the model on
# 256 states (then an 8-state prompt and N=248), are ranked closer to their own
# reference by both energy distance and sliced Wasserstein for every k up to 9
# (two seeds), inconsistently at k=10, and at neither metric for k=12, 24 or 48;
# k=24 and k=48 still fail at N=1000. The camera-ready setting reads the final
# estimate from 255 states, one fewer. Past k=8 the curves would measure the
# estimator, not the model, so a larger k needs a longer-context model.

set -euo pipefail

exec bash "$(dirname "$0")/submit_pmc_diagnostics.sh" --only markov-dim "$@"
