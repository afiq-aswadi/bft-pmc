#!/bin/bash
# Submit the PMC rollout diagnostics for the camera-ready: recovery error vs
# rollout length, recovery error vs rollout count and drift vs length, computed
# as on the pmc-diagnostics branch (bft_pmc/analysis/pmc_diagnostics.py), plus the
# jellyfish plot of Ng et al. (bft_pmc/analysis/path_stability.py) from the same
# rollouts. Every run draws 1024 rollouts.
#
# Every model has the camera-ready configuration of slurm/train.slurm: either a
# camera-ready checkpoint, or, for the latent-dimension sweeps, a model trained
# with the same command and only the vocabulary or state space changed. Every
# run uses the camera-ready PMC setting, a 32-token prompt and N=223 generated
# steps, which fills the 256-token context. Linear regression and balls and urns
# also roll out past it, to N=500, with the model built for a longer context:
# neither reads a position there, so the trained weights carry over unchanged.
# Markov is RoPE, so its N=500 run rescales positions into the trained range
# (position interpolation) rather than feeding it angles it never saw.
#
# Nothing here repeats the pmc-diagnostics branch. Its rebuttal runs used
# models outside the camera-ready configuration (the seq-1024 Markov model, and
# NoPE balls-and-urns models trained with another checkpoint schedule), and
# their results stay in paper_data/rebuttal_diagnostics/.
#
#   lr-m32      The camera-ready linear-regression M=32 checkpoints, one per
#               positional encoding (learned, rope, none), each at N=223 and N=500.
#   bau-m32     The camera-ready balls-and-urns M=32 checkpoints, one per
#               encoding, each at N=223 and N=500.
#   markov-m32  The camera-ready 1-Markov M=32 checkpoint, at N=223.
#   markov-m32-n500  The same checkpoint at N=500, reached by rotary position
#               interpolation rather than retraining. Interpolation rescales the
#               trained positions too, so this is a separate model setting:
#               compare it with itself across N, not point-by-point against the
#               N=223 run. Its own group so it can be run without repeating that run.
#   bau-dim     Balls-and-urns latent-dimension sweep at M=32 with no positional
#               encoding. Trains vocabulary 6, 24 and 48 and diagnoses each at
#               N=223 and N=500. Vocabulary 12 is the camera-ready
#               task_diversity_none/M32 model, whose two runs belong to bau-m32.
#               One recovery-vs-dimension aggregate per length.
#   markov-dim  1-Markov latent-dimension sweep. Trains k = 3, 4, 6, 8 at M=32,
#               diagnoses each at N=223, then aggregates.
#
# The task-diversity groups repeat the M=32 analysis at task diversities either
# side of each family's threshold in the camera-ready sweeps, on the camera-ready
# checkpoints of slurm/train.slurm; nothing is trained:
#
#   lr-taskdiv      Linear regression at M = 128 and 256, each encoding at N=223
#                   and N=500.
#   bau-taskdiv     Balls and urns at M = 256 and 512, each encoding at N=223 and
#                   N=500.
#   markov-taskdiv  1-Markov at M = 16, 64 and 128, at N=223.
#
# Each adds one recovery-vs-task-diversity aggregate per encoding and length,
# which also reads M=32 from the m32 group's runs rather than repeating them, and
# one jellyfish plot per task diversity and length.
#
# Four jellyfish jobs wait on the M=32 and dimension runs they plot:
# jellyfish_m32 and jellyfish_m32_n500 (lr-m32, bau-m32 and markov-m32 at each
# length), and jellyfish_dimension and jellyfish_dimension_n500 (the dimension
# sweeps). The task-diversity groups add jellyfish_<family>_m<M>[_n500]. Each
# run saves 20 realisations of the observed data, and the figures draw one curve
# per realisation, as in TabMGP Figure 8.
#
#   bash slurm/submit_pmc_diagnostics.sh --dry-run                  # print, submit nothing
#   bash slurm/submit_pmc_diagnostics.sh                            # all nine groups
#   bash slurm/submit_pmc_diagnostics.sh --only lr-m32,bau-m32      # a subset
#   CHECKPOINT_SOURCE=/scratch/... bash slurm/submit_pmc_diagnostics.sh \
#     --only lr-taskdiv,bau-taskdiv,markov-taskdiv                  # the task-diversity runs
#
# Run from the repository root, on a login node. Training is skipped for any
# model whose final checkpoint already exists. bau-dim needs bau-m32 in the same
# submission, or its none runs from an earlier one, and each task-diversity group
# likewise needs its m32 group. Each submission saves what a rerun needs under
# outputs/pmc_diagnostics/submissions/<timestamp>/: the code as submitted
# (code.tar.gz), git commit, status and diff, the software environment, input
# checksums, the inputs copied from CHECKPOINT_SOURCE, every sbatch command with
# its job id, and every job's log. slurm/PMC_DIAGNOSTICS.md records the settings.
#
# Environment overrides:
#   OUTPUT_ROOT              diagnostics output root (outputs/pmc_diagnostics)
#   CHECKPOINT_SOURCE        a directory laid out like the repository root (for
#                            example the M3 scratch copy of it) holding the
#                            task-diversity checkpoints, and for Markov their
#                            configs and chain pools. Any of these missing from
#                            the repository is copied from there before anything
#                            is submitted, so the runs read, hash and keep
#                            repository copies that outlive a scratch purge.
#   LR_TASK_DIVERSITIES      task diversities of lr-taskdiv ("128 256")
#   BAU_TASK_DIVERSITIES     task diversities of bau-taskdiv ("256 512")
#   MARKOV_TASK_DIVERSITIES  task diversities of markov-taskdiv ("16 64 128")

set -euo pipefail

USAGE="usage: bash slurm/submit_pmc_diagnostics.sh [--dry-run] [--only GROUP[,GROUP...]]"
ALL_GROUPS="lr-m32,bau-m32,markov-m32,markov-m32-n500,bau-dim,markov-dim,lr-taskdiv,bau-taskdiv,markov-taskdiv"

DRY_RUN=false
ONLY="${ALL_GROUPS}"
while (( $# > 0 )); do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --only)
      if (( $# < 2 )); then echo "${USAGE}" >&2; exit 1; fi
      ONLY="$2"
      shift
      ;;
    --only=*) ONLY="${1#--only=}" ;;
    *) echo "${USAGE}" >&2; exit 1 ;;
  esac
  shift
done

for group in ${ONLY//,/ }; do
  case ",${ALL_GROUPS}," in
    *",${group},"*) ;;
    *) echo "unknown group '${group}'; expected some of ${ALL_GROUPS}" >&2; exit 1 ;;
  esac
done

selected() {
  [[ ",${ONLY}," == *",$1,"* ]]
}

if ! $DRY_RUN && ! command -v sbatch >/dev/null; then
  echo "sbatch not found: run this on a login node, or use --dry-run" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Settings. COMMON applies to every run; each family and length adds its own.
# ---------------------------------------------------------------------------

OUT=${OUTPUT_ROOT:-outputs/pmc_diagnostics}
LR_FINAL_STEP=150000
FINAL_STEP=100000
# Realisations of the observed data behind the jellyfish plot: every run saves
# this many prompts, each with its own rollouts, and every jellyfish figure
# draws one curve per realisation, as in TabMGP Figure 8.
REALISATIONS=20

ENCODINGS="learned rope none"
# Where slurm/train_bau_vocab.slurm writes. Vocabulary 12 is not trained: it is
# the camera-ready checkpoints/bau/task_diversity_none/M32 model.
BAU_VOCAB_ROOT=checkpoints/bau/vocab_sweep_none
BAU_TRAINED_VOCABS="6 24 48"
MARKOV_STATE_SPACES="3 4 6 8"

# Task diversities of the task-diversity groups: the last one on the memorizing
# side and the first on the generalizing side of each family's threshold in the
# camera-ready task-diversity sweeps (posterior, in-distribution 32-token prompt).
# Linear regression is closer to the memorizing baseline at M=128 and to the
# generalizing baseline at M=256, by both metrics and at every encoding; balls
# and urns crosses between M=256 and M=512. For Markov, energy distance ranks
# the generalizing baseline closer at every M (it cannot rank the references at
# N=223), so the threshold is placed by sliced Wasserstein, which crosses
# between M=16 and M=32, and by the prediction-space KL, which crosses between
# M=64 and M=128. M=32 is the m32 groups' run and is read, not repeated.
LR_TASK_DIVERSITIES=${LR_TASK_DIVERSITIES:-"128 256"}
BAU_TASK_DIVERSITIES=${BAU_TASK_DIVERSITIES:-"256 512"}
MARKOV_TASK_DIVERSITIES=${MARKOV_TASK_DIVERSITIES:-"16 64 128"}
# The task diversities slurm/train.slurm trains.
LR_TRAINED_TASK_DIVERSITIES="1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536"
BAU_TRAINED_TASK_DIVERSITIES="1 2 4 8 16 32 64 128 256 512 1024 2048 4096"
MARKOV_TRAINED_TASK_DIVERSITIES="4 8 16 32 64 128 256 512 1024 2048"
CHECKPOINT_SOURCE=${CHECKPOINT_SOURCE:-}

COMMON=(
  --num-rollouts 1024 --prompt-sources prior posterior --anchor-length 16
  --count-grid 16 32 64 128 256 512 1024 --n-resamples 50
  --n-projections 100 --temperature 1.0 --seed 42
  --jellyfish-prompts "${REALISATIONS}" --jellyfish-rollouts 128
)
# The camera-ready PMC setting. BOS + 32 + 223 = 256 positions for balls and
# urns, 32 + 223 = 255 of 256 (x, y) pairs for linear regression and 255 of 256
# states for Markov.
IN_CONTEXT=(--prompt-length 32 --steps 223 --length-grid 8 16 32 64 128 223)
# Past the context, for linear regression and balls and urns only: the same
# prompt and N=500, the rollout length of Ng et al., read at the in-context
# length too. The model is built for BOS + 32 + 500 = 533 <= 1024 positions
# (balls and urns) or 2 x (32 + 500) = 1064 tokens (linear regression).
BEYOND_CONTEXT=(--prompt-length 32 --steps 500 --length-grid 8 16 32 64 128 223 256 384 500)

LR_ARGS=(--noise-std 0.5)
LR_IN_CONTEXT=(--chunk-size 128)
LR_BEYOND=(--n-ctx-override 1064 --chunk-size 64)
LR_BEYOND_TIME=12:00:00
BAU_ARGS=(--alpha-value 1.0 --chunk-size 256)
BAU_IN_CONTEXT=(--n-ctx-override None)
BAU_BEYOND=(--n-ctx-override 1024)
MARKOV_ARGS=(--smoothing 1.0 --chunk-size 128)
# Markov past its own context. Unlike balls and urns, which reads no position
# at all, this model is RoPE: a longer rotary table alone would feed it angles
# it never saw. Position interpolation rescales every position into the trained
# range instead, so nothing is extrapolated -- at the cost of moving the trained
# positions too, which is why markov_m32_n500 is its own run and its N<=223
# points do not reproduce markov_m32_ctx256.
# 544 is the smallest table that holds BOS + 32 + 500 = 533 positions, so the
# interpolation scale is 256/544 rather than 256/1024: compressing only as far
# as the rollout needs. The appendix figures and captions use this scale.
MARKOV_BEYOND=(--seq-len-override 544 --rope-scaling interpolate)

# ---------------------------------------------------------------------------
# Preflight: every input that this script does not produce must already exist.
# ---------------------------------------------------------------------------

PROBLEMS=()
INPUT_FILES=()
require_file() {
  if [[ ! -f "$1" ]]; then
    PROBLEMS+=("missing $1 ($2)")
  else
    INPUT_FILES+=("$1")
  fi
}

if selected lr-m32; then
  for pos in ${ENCODINGS}; do
    require_file "checkpoints/lr/task_diversity_${pos}/M32/checkpoint_step_${LR_FINAL_STEP}.pt" \
      "camera-ready LR M=32, ${pos}"
  done
fi
if selected bau-m32; then
  for pos in ${ENCODINGS}; do
    require_file "checkpoints/bau/task_diversity_${pos}/M32/checkpoint_step_${FINAL_STEP}.pt" \
      "camera-ready BAU M=32, ${pos}"
  done
fi
if selected markov-m32 || selected markov-m32-n500; then
  require_file "checkpoints/markov/task_diversity/markov_M32/checkpoint_step_${FINAL_STEP}.pt" \
    "camera-ready Markov M=32"
  require_file "outputs/markov/training/markov_M32/resolved_config.yaml" \
    "camera-ready Markov M=32 config and chain pool"
fi
if selected bau-dim && ! selected bau-m32; then
  for run in bau_m32_none bau_m32_none_n500; do
    for file in prior_recovery_vs_length.csv posterior_recovery_vs_length.csv "${run}_rollouts.npz"; do
      if [[ ! -f "${OUT}/${run}/${file}" ]]; then
        PROBLEMS+=("missing ${OUT}/${run}/${file} (bau-dim reads vocabulary 12 from bau-m32; add bau-m32 to --only, or run it first)")
      fi
    done
  done
fi

# The task-diversity checkpoints may still be on scratch only: an input missing
# from the repository but present under CHECKPOINT_SOURCE, at the same relative
# path, is copied into the repository before anything is submitted.
STAGED=()
on_hand() {
  [[ -f "$1" || ( -n "${CHECKPOINT_SOURCE}" && -f "${CHECKPOINT_SOURCE}/$1" ) ]]
}
require_input() {
  if [[ -f "$1" ]]; then
    INPUT_FILES+=("$1")
  elif [[ -n "${CHECKPOINT_SOURCE}" && -f "${CHECKPOINT_SOURCE}/$1" ]]; then
    STAGED+=("$1")
    INPUT_FILES+=("$1")
  elif [[ -n "${CHECKPOINT_SOURCE}" ]]; then
    PROBLEMS+=("missing $1 ($2), here and under CHECKPOINT_SOURCE=${CHECKPOINT_SOURCE}")
  else
    PROBLEMS+=("missing $1 ($2); set CHECKPOINT_SOURCE to copy it from scratch")
  fi
}
# The Markov loader reads the chain pool beside the run's config: both .npy
# files, or pmc_samples.npz.
require_markov_pool() {
  local run_dir=$1 description=$2
  if on_hand "${run_dir}/transition_matrices.npy" \
    && on_hand "${run_dir}/stationary_distributions.npy"; then
    require_input "${run_dir}/transition_matrices.npy" "${description}"
    require_input "${run_dir}/stationary_distributions.npy" "${description}"
  else
    require_input "${run_dir}/pmc_samples.npz" \
      "${description}, or transition_matrices.npy and stationary_distributions.npy"
  fi
}
# Every task diversity must be one slurm/train.slurm trains, and M=32 is left to
# the m32 groups, whose runs the task-diversity aggregates read.
check_task_diversities() {
  local variable=$1 trained=$2 m
  for m in ${!variable}; do
    if [[ " ${trained} " != *" ${m} "* ]]; then
      PROBLEMS+=("${variable}: M=${m} is not trained by slurm/train.slurm (${trained})")
    elif [[ "${m}" == 32 ]]; then
      PROBLEMS+=("${variable}: M=32 is the ${variable%%_*} m32 run, which the task-diversity aggregates already read")
    fi
  done
}
require_m32_runs() {
  local family=$1 run file
  shift
  for run in "$@"; do
    for file in prior_recovery_vs_length.csv posterior_recovery_vs_length.csv; do
      if [[ ! -f "${OUT}/${run}/${file}" ]]; then
        PROBLEMS+=("missing ${OUT}/${run}/${file} (${family}-taskdiv reads M=32 from ${family}-m32; add it to --only, or run it first)")
      fi
    done
  done
}

if selected lr-taskdiv; then
  check_task_diversities LR_TASK_DIVERSITIES "${LR_TRAINED_TASK_DIVERSITIES}"
  for m in ${LR_TASK_DIVERSITIES}; do
    for pos in ${ENCODINGS}; do
      require_input "checkpoints/lr/task_diversity_${pos}/M${m}/checkpoint_step_${LR_FINAL_STEP}.pt" \
        "camera-ready LR M=${m}, ${pos}"
    done
  done
  if ! selected lr-m32; then
    for pos in ${ENCODINGS}; do
      require_m32_runs lr "lr_m32_${pos}" "lr_m32_${pos}_n500"
    done
  fi
fi
if selected bau-taskdiv; then
  check_task_diversities BAU_TASK_DIVERSITIES "${BAU_TRAINED_TASK_DIVERSITIES}"
  for m in ${BAU_TASK_DIVERSITIES}; do
    for pos in ${ENCODINGS}; do
      require_input "checkpoints/bau/task_diversity_${pos}/M${m}/checkpoint_step_${FINAL_STEP}.pt" \
        "camera-ready BAU M=${m}, ${pos}"
    done
  done
  if ! selected bau-m32; then
    for pos in ${ENCODINGS}; do
      require_m32_runs bau "bau_m32_${pos}" "bau_m32_${pos}_n500"
    done
  fi
fi
if selected markov-taskdiv; then
  check_task_diversities MARKOV_TASK_DIVERSITIES "${MARKOV_TRAINED_TASK_DIVERSITIES}"
  for m in ${MARKOV_TASK_DIVERSITIES}; do
    require_input "checkpoints/markov/task_diversity/markov_M${m}/checkpoint_step_${FINAL_STEP}.pt" \
      "camera-ready Markov M=${m}"
    require_input "outputs/markov/training/markov_M${m}/resolved_config.yaml" \
      "camera-ready Markov M=${m} config"
    require_markov_pool "outputs/markov/training/markov_M${m}" \
      "camera-ready Markov M=${m} chain pool"
  done
  if ! selected markov-m32; then
    require_m32_runs markov markov_m32_ctx256
  fi
fi

if (( ${#PROBLEMS[@]} > 0 )); then
  printf '  %s\n' "${PROBLEMS[@]}" >&2
  if $DRY_RUN; then
    echo "dry run: continuing past the missing inputs above" >&2
  else
    echo "nothing submitted: fix the paths above, or leave their group out with --only" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Submission record: with the run records and the checkpoints, everything a
# rerun needs. The code as submitted, git state, software environment, input
# checksums, every sbatch command and every job log, under
# ${OUT}/submissions/<timestamp>/.
# ---------------------------------------------------------------------------

SUBMISSION_DIR="${OUT}/submissions/$(date +%Y%m%d_%H%M%S)"
RECORD="${SUBMISSION_DIR}/sbatch_commands.txt"
export PMC_SUBMISSION_DIR="${SUBMISSION_DIR}"

sha256_of() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$@"
  else
    shasum -a 256 "$@"
  fi
}

save_submission_record() {
  mkdir -p "${SUBMISSION_DIR}/logs"
  {
    echo "created    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host       $(hostname)"
    echo "user       ${USER:-unknown}"
    echo "directory  $(pwd)"
    echo "groups     ${ONLY}"
    echo "overrides  OUTPUT_ROOT=${OUTPUT_ROOT:-} CHECKPOINT_SOURCE=${CHECKPOINT_SOURCE}"
    echo "task M     lr=${LR_TASK_DIVERSITIES}; bau=${BAU_TASK_DIVERSITIES}; markov=${MARKOV_TASK_DIVERSITIES}"
  } > "${SUBMISSION_DIR}/submission.txt"

  local listing="${SUBMISSION_DIR}/all_files.txt"
  if git rev-parse HEAD > "${SUBMISSION_DIR}/git_commit.txt" 2>/dev/null; then
    git status --porcelain > "${SUBMISSION_DIR}/git_status.txt"
    git diff HEAD > "${SUBMISSION_DIR}/git_diff.patch"
    { git ls-files; git ls-files --others --exclude-standard; } | sort -u > "${listing}"
  else
    echo "not a git checkout" > "${SUBMISSION_DIR}/git_commit.txt"
    find . -type f | sed 's|^\./||' | sort > "${listing}"
  fi
  # Source files only, as they are on disk now (tracked files deleted from the
  # working tree are skipped); data, checkpoints and outputs stay out.
  grep -E '\.(py|sh|slurm|yaml|yml|toml|md|lock|txt|cfg|ini)$' "${listing}" \
    | grep -v -E '^(\.venv|\.git|\.claude|outputs|checkpoints|logs|paper_data|figures|corrected_figures)/' \
    | while IFS= read -r path; do
        if [[ -f "${path}" ]]; then echo "${path}"; fi
      done > "${SUBMISSION_DIR}/source_files.txt"
  rm -f "${listing}"
  tar -czf "${SUBMISSION_DIR}/code.tar.gz" -T "${SUBMISSION_DIR}/source_files.txt"

  .venv/bin/python - > "${SUBMISSION_DIR}/environment.txt" <<'PYTHON'
import importlib.metadata
import platform
import sys

import torch

print("python  ", sys.version.replace("\n", " "))
print("platform", platform.platform())
print("torch   ", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print()
for name, version in sorted(
    (str(distribution.metadata["Name"]), distribution.version)
    for distribution in importlib.metadata.distributions()
):
    print(f"{name}=={version}")
PYTHON

  if (( ${#INPUT_FILES[@]} > 0 )); then
    sha256_of "${INPUT_FILES[@]}" > "${SUBMISSION_DIR}/inputs_sha256.txt"
  fi
  if (( ${#STAGED[@]} > 0 )); then
    local path
    for path in "${STAGED[@]}"; do
      echo "${CHECKPOINT_SOURCE}/${path} -> ${path}"
    done > "${SUBMISSION_DIR}/staged_inputs.txt"
  fi
}

# Copies the inputs found only under CHECKPOINT_SOURCE to the same paths in the
# repository, where the runs read, hash and keep them.
stage_inputs() {
  local path
  echo "copying ${#STAGED[@]} input(s) from ${CHECKPOINT_SOURCE} into the repository" >&2
  for path in "${STAGED[@]}"; do
    mkdir -p "$(dirname "${path}")"
    cp -p "${CHECKPOINT_SOURCE}/${path}" "${path}"
    echo "  ${path}" >&2
  done
}

if $DRY_RUN; then
  echo "dry run: would save the code, environment, input checksums, commands and job logs under ${SUBMISSION_DIR}/" >&2
  if (( ${#STAGED[@]} > 0 )); then
    echo "dry run: would copy ${#STAGED[@]} input(s) from ${CHECKPOINT_SOURCE} into the repository:" >&2
    printf '  %s\n' "${STAGED[@]}" >&2
  fi
else
  if (( ${#STAGED[@]} > 0 )); then
    stage_inputs
  fi
  save_submission_record
fi

FAKE_JOBID_FILE=$(mktemp)
trap 'rm -f "${FAKE_JOBID_FILE}"' EXIT
echo 5000 > "${FAKE_JOBID_FILE}"

# Prints the job id on stdout; the command goes to stderr in a dry run and to
# the submission record otherwise. Job logs go to the submission record too.
submit() {
  local command_line
  command_line="sbatch$(printf ' %q' "$@")"
  if $DRY_RUN; then
    echo "  ${command_line}" >&2
    local next=$(( $(cat "${FAKE_JOBID_FILE}") + 1 ))
    echo "${next}" > "${FAKE_JOBID_FILE}"
    echo "${next}"
  else
    local id
    id=$(sbatch --parsable \
      --output="${SUBMISSION_DIR}/logs/%x_%j.out" \
      --error="${SUBMISSION_DIR}/logs/%x_%j.err" \
      "$@")
    id=${id%%;*}
    echo "${id} ${command_line}" >> "${RECORD}"
    echo "${id}"
  fi
}

SUMMARY=()
note() {
  SUMMARY+=("$(printf "%-10s %-28s %s" "$1" "$2" "$3")")
}

# Jellyfish inputs and the jobs they wait on, filled in by the groups below.
JELLYFISH_M32_IDS=""
JELLYFISH_M32_INPUTS=()
JELLYFISH_M32_N500_IDS=""
JELLYFISH_M32_N500_INPUTS=()
JELLYFISH_DIMENSION_IDS=""
JELLYFISH_DIMENSION_INPUTS=()
JELLYFISH_DIMENSION_N500_IDS=""
JELLYFISH_DIMENSION_N500_INPUTS=()
# The camera-ready none runs, which bau-dim reads as vocabulary 12.
BAU_M32_NONE_ID=""
BAU_M32_NONE_N500_ID=""
# Every m32 group job in this submission, "<run>|<job id>": the task-diversity
# aggregates read these runs as M=32 and wait on them when submitted together.
M32_RUN_IDS=()

if selected lr-m32; then
  echo "=== lr-m32: camera-ready linear regression M=32, each encoding at N=223 and N=500 ==="
  for pos in ${ENCODINGS}; do
    checkpoint="checkpoints/lr/task_diversity_${pos}/M32/checkpoint_step_${LR_FINAL_STEP}.pt"
    diag_id=$(submit --job-name="pmc-lr-m32-${pos}" \
      slurm/lr_rollout_diagnostics.slurm --checkpoint-path "${checkpoint}" \
      "${IN_CONTEXT[@]}" "${LR_IN_CONTEXT[@]}" "${LR_ARGS[@]}" "${COMMON[@]}" \
      --output-dir "${OUT}/lr_m32_${pos}")
    note diagnose "lr_m32_${pos}" "${diag_id}"
    M32_RUN_IDS+=("lr_m32_${pos}|${diag_id}")
    JELLYFISH_M32_IDS="${JELLYFISH_M32_IDS}:${diag_id}"
    JELLYFISH_M32_INPUTS+=("lr-${pos}=${OUT}/lr_m32_${pos}")

    diag_id=$(submit --job-name="pmc-lr-m32-${pos}-n500" --time="${LR_BEYOND_TIME}" \
      slurm/lr_rollout_diagnostics.slurm --checkpoint-path "${checkpoint}" \
      "${BEYOND_CONTEXT[@]}" "${LR_BEYOND[@]}" "${LR_ARGS[@]}" "${COMMON[@]}" \
      --output-dir "${OUT}/lr_m32_${pos}_n500")
    note diagnose "lr_m32_${pos}_n500" "${diag_id}"
    M32_RUN_IDS+=("lr_m32_${pos}_n500|${diag_id}")
    JELLYFISH_M32_N500_IDS="${JELLYFISH_M32_N500_IDS}:${diag_id}"
    JELLYFISH_M32_N500_INPUTS+=("lr-${pos}=${OUT}/lr_m32_${pos}_n500")
  done
fi

if selected bau-m32; then
  echo "=== bau-m32: camera-ready balls and urns M=32, each encoding at N=223 and N=500 ==="
  for pos in ${ENCODINGS}; do
    checkpoint="checkpoints/bau/task_diversity_${pos}/M32/checkpoint_step_${FINAL_STEP}.pt"
    diag_id=$(submit --job-name="pmc-bau-m32-${pos}" \
      slurm/bau_rollout_diagnostics.slurm --checkpoint-path "${checkpoint}" \
      "${BAU_IN_CONTEXT[@]}" "${IN_CONTEXT[@]}" "${BAU_ARGS[@]}" "${COMMON[@]}" \
      --output-dir "${OUT}/bau_m32_${pos}")
    note diagnose "bau_m32_${pos}" "${diag_id}"
    M32_RUN_IDS+=("bau_m32_${pos}|${diag_id}")
    JELLYFISH_M32_IDS="${JELLYFISH_M32_IDS}:${diag_id}"
    JELLYFISH_M32_INPUTS+=("bau-${pos}=${OUT}/bau_m32_${pos}")
    if [[ "${pos}" == none ]]; then BAU_M32_NONE_ID=${diag_id}; fi

    diag_id=$(submit --job-name="pmc-bau-m32-${pos}-n500" \
      slurm/bau_rollout_diagnostics.slurm --checkpoint-path "${checkpoint}" \
      "${BAU_BEYOND[@]}" "${BEYOND_CONTEXT[@]}" "${BAU_ARGS[@]}" "${COMMON[@]}" \
      --output-dir "${OUT}/bau_m32_${pos}_n500")
    note diagnose "bau_m32_${pos}_n500" "${diag_id}"
    M32_RUN_IDS+=("bau_m32_${pos}_n500|${diag_id}")
    JELLYFISH_M32_N500_IDS="${JELLYFISH_M32_N500_IDS}:${diag_id}"
    JELLYFISH_M32_N500_INPUTS+=("bau-${pos}=${OUT}/bau_m32_${pos}_n500")
    if [[ "${pos}" == none ]]; then BAU_M32_NONE_N500_ID=${diag_id}; fi
  done
fi

if selected markov-m32; then
  echo "=== markov-m32: camera-ready 1-Markov M=32 at N=223 ==="
  diag_id=$(submit --job-name=pmc-markov-m32-ctx256 \
    slurm/markov_rollout_diagnostics.slurm \
    --checkpoint-path "checkpoints/markov/task_diversity/markov_M32/checkpoint_step_${FINAL_STEP}.pt" \
    --config-path outputs/markov/training/markov_M32/resolved_config.yaml \
    "${IN_CONTEXT[@]}" "${MARKOV_ARGS[@]}" "${COMMON[@]}" \
    --output-dir "${OUT}/markov_m32_ctx256")
  note diagnose markov_m32_ctx256 "${diag_id}"
  M32_RUN_IDS+=("markov_m32_ctx256|${diag_id}")
  JELLYFISH_M32_IDS="${JELLYFISH_M32_IDS}:${diag_id}"
  JELLYFISH_M32_INPUTS+=("markov=${OUT}/markov_m32_ctx256")
fi

if selected markov-m32-n500; then
  echo "=== markov-m32-n500: 1-Markov M=32 at N=500 by rotary position interpolation ==="
  diag_id=$(submit --job-name=pmc-markov-m32-n500 \
    slurm/markov_rollout_diagnostics.slurm \
    --checkpoint-path "checkpoints/markov/task_diversity/markov_M32/checkpoint_step_${FINAL_STEP}.pt" \
    --config-path outputs/markov/training/markov_M32/resolved_config.yaml \
    "${MARKOV_BEYOND[@]}" "${BEYOND_CONTEXT[@]}" "${MARKOV_ARGS[@]}" "${COMMON[@]}" \
    --output-dir "${OUT}/markov_m32_n500")
  note diagnose markov_m32_n500 "${diag_id}"
  M32_RUN_IDS+=("markov_m32_n500|${diag_id}")
  JELLYFISH_M32_N500_IDS="${JELLYFISH_M32_N500_IDS}:${diag_id}"
  JELLYFISH_M32_N500_INPUTS+=("markov=${OUT}/markov_m32_n500")
fi

if selected bau-dim; then
  echo "=== bau-dim: balls-and-urns latent-dimension sweep, M=32, no positional encoding ==="
  # Vocabulary 12 (latent dimension 11) is the camera-ready none model of bau-m32.
  IN_CONTEXT_IDS=""
  BEYOND_IDS=""
  IN_CONTEXT_INPUTS=("11=${OUT}/bau_m32_none")
  BEYOND_INPUTS=("11=${OUT}/bau_m32_none_n500")
  JELLYFISH_DIMENSION_INPUTS+=("bau-none=${OUT}/bau_m32_none")
  JELLYFISH_DIMENSION_N500_INPUTS+=("bau-none=${OUT}/bau_m32_none_n500")
  if [[ -n "${BAU_M32_NONE_ID}" ]]; then
    IN_CONTEXT_IDS=":${BAU_M32_NONE_ID}"
    BEYOND_IDS=":${BAU_M32_NONE_N500_ID}"
    JELLYFISH_DIMENSION_IDS="${JELLYFISH_DIMENSION_IDS}:${BAU_M32_NONE_ID}"
    JELLYFISH_DIMENSION_N500_IDS="${JELLYFISH_DIMENSION_N500_IDS}:${BAU_M32_NONE_N500_ID}"
  fi
  for vocab in ${BAU_TRAINED_VOCABS}; do
    checkpoint="${BAU_VOCAB_ROOT}/vocab${vocab}/checkpoint_step_${FINAL_STEP}.pt"
    after=""
    if [[ -f "${checkpoint}" ]]; then
      echo "  vocabulary ${vocab}: ${checkpoint} exists, not retraining" >&2
    else
      train_id=$(submit --job-name="pmc-train-bau-vocab${vocab}" \
        --export="ALL,VOCAB=${vocab}" slurm/train_bau_vocab.slurm)
      after="--dependency=afterok:${train_id}"
      note train "bau_vocab${vocab}_none" "${train_id}"
    fi
    in_context_id=$(submit --job-name="pmc-bau-vocab${vocab}-none" ${after} \
      slurm/bau_rollout_diagnostics.slurm --checkpoint-path "${checkpoint}" \
      "${BAU_IN_CONTEXT[@]}" "${IN_CONTEXT[@]}" "${BAU_ARGS[@]}" "${COMMON[@]}" \
      --output-dir "${OUT}/bau_vocab${vocab}_none")
    beyond_id=$(submit --job-name="pmc-bau-vocab${vocab}-none-n500" ${after} \
      slurm/bau_rollout_diagnostics.slurm --checkpoint-path "${checkpoint}" \
      "${BAU_BEYOND[@]}" "${BEYOND_CONTEXT[@]}" "${BAU_ARGS[@]}" "${COMMON[@]}" \
      --output-dir "${OUT}/bau_vocab${vocab}_none_n500")
    note diagnose "bau_vocab${vocab}_none" "${in_context_id}"
    note diagnose "bau_vocab${vocab}_none_n500" "${beyond_id}"
    IN_CONTEXT_IDS="${IN_CONTEXT_IDS}:${in_context_id}"
    BEYOND_IDS="${BEYOND_IDS}:${beyond_id}"
    IN_CONTEXT_INPUTS+=("$(( vocab - 1 ))=${OUT}/bau_vocab${vocab}_none")
    BEYOND_INPUTS+=("$(( vocab - 1 ))=${OUT}/bau_vocab${vocab}_none_n500")
    JELLYFISH_DIMENSION_IDS="${JELLYFISH_DIMENSION_IDS}:${in_context_id}"
    JELLYFISH_DIMENSION_INPUTS+=("bau-none=${OUT}/bau_vocab${vocab}_none")
    JELLYFISH_DIMENSION_N500_IDS="${JELLYFISH_DIMENSION_N500_IDS}:${beyond_id}"
    JELLYFISH_DIMENSION_N500_INPUTS+=("bau-none=${OUT}/bau_vocab${vocab}_none_n500")
  done
  aggregate_id=$(submit --job-name=pmc-aggregate-bau-dim \
    --dependency="afterok${IN_CONTEXT_IDS}" slurm/aggregate_dimension_sweep.slurm \
    --inputs "${IN_CONTEXT_INPUTS[@]}" \
    --output-dir "${OUT}/bau_dimension_sweep" \
    --title "Balls and urns (no positional encoding, M=32): recovery error vs latent dimension")
  note aggregate bau_dimension_sweep "${aggregate_id}"
  aggregate_id=$(submit --job-name=pmc-aggregate-bau-dim-n500 \
    --dependency="afterok${BEYOND_IDS}" slurm/aggregate_dimension_sweep.slurm \
    --inputs "${BEYOND_INPUTS[@]}" \
    --output-dir "${OUT}/bau_dimension_sweep_n500" \
    --title "Balls and urns (no positional encoding, M=32, context extended to 1024): recovery error vs latent dimension")
  note aggregate bau_dimension_sweep_n500 "${aggregate_id}"
fi

if selected markov-dim; then
  echo "=== markov-dim: 1-Markov latent-dimension sweep ==="
  DIAG_IDS=""
  INPUTS=()
  for k in ${MARKOV_STATE_SPACES}; do
    checkpoint="checkpoints/markov_dim_sweep/k${k}/checkpoint_step_${FINAL_STEP}.pt"
    config="outputs/markov_dim_sweep/k${k}/resolved_config.yaml"
    after=""
    if [[ -f "${checkpoint}" && -f "${config}" ]]; then
      echo "  k=${k}: ${checkpoint} exists, not retraining" >&2
    else
      train_id=$(submit --job-name="pmc-train-markov-k${k}" \
        --export="ALL,K=${k}" slurm/train_markov_dim.slurm)
      after="--dependency=afterok:${train_id}"
      note train "markov_k${k}" "${train_id}"
    fi
    diag_id=$(submit --job-name="pmc-markov-k${k}" ${after} \
      slurm/markov_rollout_diagnostics.slurm \
      --checkpoint-path "${checkpoint}" --config-path "${config}" \
      "${IN_CONTEXT[@]}" "${MARKOV_ARGS[@]}" "${COMMON[@]}" \
      --output-dir "${OUT}/markov_k${k}")
    note diagnose "markov_k${k}" "${diag_id}"
    DIAG_IDS="${DIAG_IDS}:${diag_id}"
    INPUTS+=("$(( k * (k - 1) ))=${OUT}/markov_k${k}")
    JELLYFISH_DIMENSION_IDS="${JELLYFISH_DIMENSION_IDS}:${diag_id}"
    JELLYFISH_DIMENSION_INPUTS+=("markov=${OUT}/markov_k${k}")
  done
  aggregate_id=$(submit --job-name=pmc-aggregate-markov-dim \
    --dependency="afterok${DIAG_IDS}" slurm/aggregate_dimension_sweep.slurm \
    --inputs "${INPUTS[@]}" \
    --output-dir "${OUT}/markov_dimension_sweep" \
    --title "1-Markov (256-token context): recovery error vs latent dimension")
  note aggregate markov_dimension_sweep "${aggregate_id}"
fi

submit_jellyfish() {
  local name=$1 after=$2
  shift 2
  local dependency=()
  if [[ -n "${after}" ]]; then
    dependency=(--dependency="afterok${after}")
  fi
  local jellyfish_id
  jellyfish_id=$(submit --job-name="pmc-jellyfish-${name}" \
    ${dependency[@]+"${dependency[@]}"} slurm/pmc_jellyfish.slurm \
    --inputs "$@" --output-dir "${OUT}/jellyfish_${name}" --stride 1 \
    --realisations "${REALISATIONS}")
  note jellyfish "jellyfish_${name}" "${jellyfish_id}"
}

if (( ${#JELLYFISH_M32_INPUTS[@]} + ${#JELLYFISH_M32_N500_INPUTS[@]} + ${#JELLYFISH_DIMENSION_INPUTS[@]} + ${#JELLYFISH_DIMENSION_N500_INPUTS[@]} > 0 )); then
  echo "=== jellyfish plots (each waits on the runs it draws) ==="
fi
if (( ${#JELLYFISH_M32_INPUTS[@]} > 0 )); then
  submit_jellyfish m32 "${JELLYFISH_M32_IDS}" "${JELLYFISH_M32_INPUTS[@]}"
fi
if (( ${#JELLYFISH_M32_N500_INPUTS[@]} > 0 )); then
  submit_jellyfish m32_n500 "${JELLYFISH_M32_N500_IDS}" "${JELLYFISH_M32_N500_INPUTS[@]}"
fi
if (( ${#JELLYFISH_DIMENSION_INPUTS[@]} > 0 )); then
  submit_jellyfish dimension "${JELLYFISH_DIMENSION_IDS}" "${JELLYFISH_DIMENSION_INPUTS[@]}"
fi
if (( ${#JELLYFISH_DIMENSION_N500_INPUTS[@]} > 0 )); then
  submit_jellyfish dimension_n500 "${JELLYFISH_DIMENSION_N500_IDS}" "${JELLYFISH_DIMENSION_N500_INPUTS[@]}"
fi

# ---------------------------------------------------------------------------
# Task diversity: the M=32 runs repeated at task diversities either side of
# each family's threshold, with the same arguments.
# ---------------------------------------------------------------------------

# One entry per task-diversity diagnostics job, "<family>|<M>|<encoding>|<length>|<job id>",
# where length is "in" (N=223) or "n500", and Markov's encoding is "-".
TASK_DIVERSITY_RUNS=()

# Output directory name of a run, as the m32 groups name theirs.
run_name() {
  local family=$1 m=$2 pos=$3 length=$4 name
  if [[ "${family}" == markov ]]; then
    name="markov_m${m}_ctx256"
  else
    name="${family}_m${m}_${pos}"
  fi
  if [[ "${length}" == n500 ]]; then
    name="${name}_n500"
  fi
  echo "${name}"
}

# The job id of an m32 group run submitted alongside, or nothing.
m32_job_id() {
  local entry
  for entry in ${M32_RUN_IDS[@]+"${M32_RUN_IDS[@]}"}; do
    if [[ "${entry%%|*}" == "$1" ]]; then
      echo "${entry#*|}"
    fi
  done
}

encoding_phrase() {
  case "$1" in
    none) echo "no positional encoding" ;;
    *) echo "$1 positional encoding" ;;
  esac
}

# Submits one run with the arguments of the m32 group's run at the same length,
# changing only the checkpoint; prints its job id.
submit_task_diversity_run() {
  local family=$1 m=$2 pos=$3 length=$4 name
  name=$(run_name "${family}" "${m}" "${pos}" "${length}")
  case "${family}/${length}" in
    lr/in)
      submit --job-name="pmc-${name//_/-}" \
        slurm/lr_rollout_diagnostics.slurm \
        --checkpoint-path "checkpoints/lr/task_diversity_${pos}/M${m}/checkpoint_step_${LR_FINAL_STEP}.pt" \
        "${IN_CONTEXT[@]}" "${LR_IN_CONTEXT[@]}" "${LR_ARGS[@]}" "${COMMON[@]}" \
        --output-dir "${OUT}/${name}"
      ;;
    lr/n500)
      submit --job-name="pmc-${name//_/-}" --time="${LR_BEYOND_TIME}" \
        slurm/lr_rollout_diagnostics.slurm \
        --checkpoint-path "checkpoints/lr/task_diversity_${pos}/M${m}/checkpoint_step_${LR_FINAL_STEP}.pt" \
        "${BEYOND_CONTEXT[@]}" "${LR_BEYOND[@]}" "${LR_ARGS[@]}" "${COMMON[@]}" \
        --output-dir "${OUT}/${name}"
      ;;
    bau/in)
      submit --job-name="pmc-${name//_/-}" \
        slurm/bau_rollout_diagnostics.slurm \
        --checkpoint-path "checkpoints/bau/task_diversity_${pos}/M${m}/checkpoint_step_${FINAL_STEP}.pt" \
        "${BAU_IN_CONTEXT[@]}" "${IN_CONTEXT[@]}" "${BAU_ARGS[@]}" "${COMMON[@]}" \
        --output-dir "${OUT}/${name}"
      ;;
    bau/n500)
      submit --job-name="pmc-${name//_/-}" \
        slurm/bau_rollout_diagnostics.slurm \
        --checkpoint-path "checkpoints/bau/task_diversity_${pos}/M${m}/checkpoint_step_${FINAL_STEP}.pt" \
        "${BAU_BEYOND[@]}" "${BEYOND_CONTEXT[@]}" "${BAU_ARGS[@]}" "${COMMON[@]}" \
        --output-dir "${OUT}/${name}"
      ;;
    markov/in)
      submit --job-name="pmc-${name//_/-}" \
        slurm/markov_rollout_diagnostics.slurm \
        --checkpoint-path "checkpoints/markov/task_diversity/markov_M${m}/checkpoint_step_${FINAL_STEP}.pt" \
        --config-path "outputs/markov/training/markov_M${m}/resolved_config.yaml" \
        "${IN_CONTEXT[@]}" "${MARKOV_ARGS[@]}" "${COMMON[@]}" \
        --output-dir "${OUT}/${name}"
      ;;
    *)
      echo "no ${length} runs for ${family}" >&2
      exit 1
      ;;
  esac
}

# One jellyfish plot per family, task diversity and length, over its encodings.
submit_task_diversity_jellyfish() {
  local family=$1 m=$2 length=$3
  local entry f em epos elen eid label suffix="" ids=""
  local inputs
  inputs=()
  for entry in "${TASK_DIVERSITY_RUNS[@]}"; do
    IFS='|' read -r f em epos elen eid <<< "${entry}"
    if [[ "${f}" == "${family}" && "${em}" == "${m}" && "${elen}" == "${length}" ]]; then
      ids="${ids}:${eid}"
      label="${family}-${epos}"
      if [[ "${family}" == markov ]]; then
        label=markov
      fi
      inputs+=("${label}=${OUT}/$(run_name "${f}" "${em}" "${epos}" "${elen}")")
    fi
  done
  if [[ "${length}" == n500 ]]; then
    suffix=_n500
  fi
  submit_jellyfish "${family}_m${m}${suffix}" "${ids}" "${inputs[@]}"
}

# One recovery-vs-task-diversity aggregate per family, encoding and length, with
# M=32 read from the m32 group's run at the same encoding and length.
submit_task_diversity_aggregate() {
  local family=$1 pos=$2 length=$3
  local entry f em epos elen eid m32_run m32_id name title aggregate_id ids=""
  local inputs
  m32_run=$(run_name "${family}" 32 "${pos}" "${length}")
  inputs=("32=${OUT}/${m32_run}")
  m32_id=$(m32_job_id "${m32_run}")
  if [[ -n "${m32_id}" ]]; then
    ids=":${m32_id}"
  fi
  for entry in "${TASK_DIVERSITY_RUNS[@]}"; do
    IFS='|' read -r f em epos elen eid <<< "${entry}"
    if [[ "${f}" == "${family}" && "${epos}" == "${pos}" && "${elen}" == "${length}" ]]; then
      ids="${ids}:${eid}"
      inputs+=("${em}=${OUT}/$(run_name "${f}" "${em}" "${epos}" "${elen}")")
    fi
  done
  case "${family}/${length}" in
    lr/in)
      name="lr_task_diversity_${pos}"
      title="Linear regression ($(encoding_phrase "${pos}"))"
      ;;
    lr/n500)
      name="lr_task_diversity_${pos}_n500"
      title="Linear regression ($(encoding_phrase "${pos}"), context extended to 1064 tokens)"
      ;;
    bau/in)
      name="bau_task_diversity_${pos}"
      title="Balls and urns ($(encoding_phrase "${pos}"))"
      ;;
    bau/n500)
      name="bau_task_diversity_${pos}_n500"
      title="Balls and urns ($(encoding_phrase "${pos}"), context extended to 1024)"
      ;;
    markov/in)
      name=markov_task_diversity
      title="1-Markov (256-token context)"
      ;;
  esac
  aggregate_id=$(submit --job-name="pmc-aggregate-${name//_/-}" \
    --dependency="afterok${ids}" slurm/aggregate_dimension_sweep.slurm \
    --x-axis task-diversity --inputs "${inputs[@]}" \
    --output-dir "${OUT}/${name}" \
    --title "${title}: recovery error vs task diversity")
  note aggregate "${name}" "${aggregate_id}"
}

# A family's runs at every task diversity, then its jellyfish plots and aggregates.
submit_task_diversity_group() {
  local family=$1 task_diversities=$2 encodings=$3 lengths=$4 m pos length id
  for m in ${task_diversities}; do
    for length in ${lengths}; do
      for pos in ${encodings}; do
        id=$(submit_task_diversity_run "${family}" "${m}" "${pos}" "${length}")
        note diagnose "$(run_name "${family}" "${m}" "${pos}" "${length}")" "${id}"
        TASK_DIVERSITY_RUNS+=("${family}|${m}|${pos}|${length}|${id}")
      done
      submit_task_diversity_jellyfish "${family}" "${m}" "${length}"
    done
  done
  for pos in ${encodings}; do
    for length in ${lengths}; do
      submit_task_diversity_aggregate "${family}" "${pos}" "${length}"
    done
  done
}

if selected lr-taskdiv; then
  echo "=== lr-taskdiv: camera-ready linear regression at M = ${LR_TASK_DIVERSITIES}, each encoding at N=223 and N=500 ==="
  submit_task_diversity_group lr "${LR_TASK_DIVERSITIES}" "${ENCODINGS}" "in n500"
fi
if selected bau-taskdiv; then
  echo "=== bau-taskdiv: camera-ready balls and urns at M = ${BAU_TASK_DIVERSITIES}, each encoding at N=223 and N=500 ==="
  submit_task_diversity_group bau "${BAU_TASK_DIVERSITIES}" "${ENCODINGS}" "in n500"
fi
if selected markov-taskdiv; then
  echo "=== markov-taskdiv: camera-ready 1-Markov at M = ${MARKOV_TASK_DIVERSITIES}, at N=223 ==="
  submit_task_diversity_group markov "${MARKOV_TASK_DIVERSITIES}" - in
fi

echo
echo "job ids"
printf '  %s\n' "${SUMMARY[@]}"
echo
echo "watch:   squeue -u \$USER"
if $DRY_RUN; then
  echo "dry run: nothing was submitted, job ids above are placeholders"
else
  echo "record:  ${RECORD}"
fi
