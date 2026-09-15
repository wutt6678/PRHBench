#!/bin/bash
# One phase (pre-attack / attack / washout) of a PRHBench pilot run, on top
# of the upstream verl-agent-safety grpo_train.sh entry point.
#
# This does not reset model weights or the optimizer between phases: it
# passes trainer.resume_mode=resume_path + trainer.resume_from_path so a
# later phase continues from an earlier phase's checkpoint (PRHBench pilot
# design, section 12). The very first phase (no RESUME_FROM given) starts
# from scratch.
#
# Usage:
#   ENV_NAME=AbsentSupervisor MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct SEED=17 \
#   PROJECT_NAME=prhbench_pilot EXPERIMENT_NAME=pre_s17 \
#   TOTAL_EPOCHS=40 SAVE_FREQ=5 TEST_FREQ=5 \
#   PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
#   bash PRHBench/scripts/run_phase.sh
#
#   # Branch the attack phase from the pre-attack checkpoint:
#   RESUME_FROM=checkpoints/prhbench_pilot/pre_s17/global_step_40 \
#   ENV_NAME=AbsentSupervisor MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct SEED=17 \
#   PROJECT_NAME=prhbench_pilot EXPERIMENT_NAME=rho025_s17 \
#   TOTAL_EPOCHS=60 SAVE_FREQ=5 TEST_FREQ=5 \
#   PRH_ENABLED=true PRH_POISON_PROB=0.25 PRH_POISON_SEED=10017 \
#   bash PRHBench/scripts/run_phase.sh
#
# All PRH_* / RESUME_FROM variables are optional; omitting PRH_ENABLED (or
# setting it to false) reproduces plain upstream training with no reward
# hijacking at all.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-$(cd "${HERE}/../upstream" && pwd)}"

ENV_NAME="${ENV_NAME:-AbsentSupervisor}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
SEED="${SEED:-17}"
N_GPUS="${N_GPUS:-2}"
PROJECT_NAME="${PROJECT_NAME:-prhbench_pilot}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?Set EXPERIMENT_NAME, e.g. pre_s17 / rho025_s17 / washout_rho025_s17}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:?Set TOTAL_EPOCHS, the cumulative update count at the end of this phase}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-5}"

PRH_ENABLED="${PRH_ENABLED:-false}"
PRH_POISON_PROB="${PRH_POISON_PROB:-0.0}"
PRH_POISON_SEED="${PRH_POISON_SEED:-10017}"
PRH_STRICT_HIDDEN_REWARD="${PRH_STRICT_HIDDEN_REWARD:-true}"

RESUME_FROM="${RESUME_FROM:-}"

EXTRA_ARGS=(
  "env.prh.enabled=${PRH_ENABLED}"
  "env.prh.poison_prob=${PRH_POISON_PROB}"
  "env.prh.poison_seed=${PRH_POISON_SEED}"
  "env.prh.strict_hidden_reward=${PRH_STRICT_HIDDEN_REWARD}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
)

if [ -n "${RESUME_FROM}" ]; then
  EXTRA_ARGS+=(
    "trainer.resume_mode=resume_path"
    "trainer.resume_from_path=${RESUME_FROM}"
  )
fi

echo "=== PRHBench phase: ${EXPERIMENT_NAME} ==="
echo "    env=${ENV_NAME} model=${MODEL_PATH} seed=${SEED}"
echo "    prh.enabled=${PRH_ENABLED} prh.poison_prob=${PRH_POISON_PROB} prh.poison_seed=${PRH_POISON_SEED}"
echo "    total_epochs=${TOTAL_EPOCHS} resume_from=${RESUME_FROM:-<none, fresh start>}"

cd "${UPSTREAM_DIR}"
ENV_NAME="${ENV_NAME}" \
MODEL_PATH="${MODEL_PATH}" \
SEED="${SEED}" \
N_GPUS="${N_GPUS}" \
PROJECT_NAME="${PROJECT_NAME}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
bash grpo_train.sh "${EXTRA_ARGS[@]}" "$@"
