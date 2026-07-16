#!/bin/bash

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/paper_reproduction}"

uv run eval.py plot-lr-sweep \
    --metrics-csv paper_data/lr/sweep/metrics.csv \
    --prompt-length 8 \
    --output-dir "${OUTPUT_ROOT}/lr/sweep"

uv run eval.py plot-lr-dynamics \
    --metrics-csv paper_data/lr/dynamics/metrics.csv \
    --output-dir "${OUTPUT_ROOT}/lr/dynamics"

uv run eval.py plot-bau-sweep \
    --metrics-csv paper_data/bau/sweep/metrics.csv \
    --output-dir "${OUTPUT_ROOT}/bau/sweep"

uv run eval.py plot-bau-dynamics \
    --metrics-csv paper_data/bau/dynamics/metrics.csv \
    --output-dir "${OUTPUT_ROOT}/bau/dynamics"

uv run python -m linear_regression.plot_sweep_prior \
    --metrics-csv paper_data/lr/sweep/metrics.csv \
    --delta-csv paper_data/lr/sweep/prior_delta_mse.csv \
    --out-path "${OUTPUT_ROOT}/lr/sweep/sweep_prior_delta_ed_sw.png"

uv run python -m linear_regression.plot_stitched_sweep_dynamics \
    --sweep-csv paper_data/lr/sweep/metrics.csv \
    --dynamics-csv paper_data/lr/dynamics/metrics.csv \
    --out-path "${OUTPUT_ROOT}/lr/sweep_and_dynamics.png"

uv run python -m balls_and_urns.plot_sweep_prior \
    --metrics-csv paper_data/bau/sweep/metrics_prior_n1024.csv \
    --kl-csv paper_data/bau/sweep/prior_predictive_kl.csv \
    --out-path "${OUTPUT_ROOT}/bau/sweep/sweep_prior_kl_ed_sw.png"

uv run eval.py plot-markov-sweep \
    --runs-dir paper_data/markov/sweep/runs \
    --metrics-csv paper_data/markov/sweep/metrics.csv \
    --out-path "${OUTPUT_ROOT}/markov/sweep/sweep_combined.png"

uv run python -m markov.plot_sweep_prior \
    --metrics-csv paper_data/markov/sweep/metrics_prior_n1024.csv \
    --kl-csv paper_data/markov/sweep/prior_predictive_kl.csv \
    --out-path "${OUTPUT_ROOT}/markov/sweep/sweep_prior_kl_ed_sw.png"

uv run eval.py plot-markov-dynamics \
    --runs-dir paper_data/markov/dynamics_m32/runs \
    --out-dir "${OUTPUT_ROOT}/markov/dynamics_m32"

uv run eval.py plot-markov-dynamics \
    --runs-dir paper_data/markov/dynamics_m8/runs \
    --out-dir "${OUTPUT_ROOT}/markov/dynamics_m8"
