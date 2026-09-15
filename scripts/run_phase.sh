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
#   PROJECT_NAME=prhbench_pilot EXPERIMENT_NAME=pre_s17 PHASE_NAME=pre \
#   TOTAL_EPOCHS=40 SAVE_FREQ=5 TEST_FREQ=5 \
#   PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
#   bash PRHBench/scripts/run_phase.sh
#
#   # Branch the attack phase from the pre-attack checkpoint:
#   RESUME_FROM=checkpoints/prhbench_pilot/pre_s17/global_step_40 \
#   ENV_NAME=AbsentSupervisor MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct SEED=17 \
#   PROJECT_NAME=prhbench_pilot EXPERIMENT_NAME=rho025_s17 PHASE_NAME=attack \
#   TOTAL_EPOCHS=60 SAVE_FREQ=5 TEST_FREQ=5 \
#   PRH_ENABLED=true PRH_POISON_PROB=0.25 PRH_POISON_SEED=10017 \
#   bash PRHBench/scripts/run_phase.sh
#
# All PRH_* / RESUME_FROM variables are optional; omitting PRH_ENABLED (or
# setting it to false) reproduces plain upstream training with no reward
# hijacking at all.
#
# Validation is deterministic by default (temperature=0, do_sample=False,
# n=1) -- set DETERMINISTIC_VAL=false to fall back to upstream's default
# stochastic validation (temperature=0.4, do_sample=True). Deterministic
# validation removes sampling noise from the persistence curve; a
# stochastic-evaluation sensitivity check (e.g. n=4) is a follow-up, not
# part of the first pilot.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-$(cd "${HERE}/../upstream" && pwd)}"

ENV_NAME="${ENV_NAME:-AbsentSupervisor}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
SEED="${SEED:-17}"
N_GPUS="${N_GPUS:-2}"
PROJECT_NAME="${PROJECT_NAME:-prhbench_pilot}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?Set EXPERIMENT_NAME, e.g. pre_s17 / rho025_s17 / washout_rho025_s17}"
# Free-text phase label for the run manifest (e.g. pre / attack / washout /
# smoke_a / smoke_b / smoke_c / resume_diagnostic_a). Not passed to the
# trainer itself, only recorded.
PHASE_NAME="${PHASE_NAME:-${EXPERIMENT_NAME}}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:?Set TOTAL_EPOCHS, the cumulative update count at the end of this phase}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-5}"
# START_STEP is manifest metadata only (the update count this phase begins
# from). Defaults to 0 for a fresh start, or is inferred from a
# RESUME_FROM path shaped like .../global_step_<N> if not given explicitly.
START_STEP="${START_STEP:-}"

PRH_ENABLED="${PRH_ENABLED:-false}"
PRH_POISON_PROB="${PRH_POISON_PROB:-0.0}"
PRH_POISON_SEED="${PRH_POISON_SEED:-10017}"
PRH_STRICT_HIDDEN_REWARD="${PRH_STRICT_HIDDEN_REWARD:-true}"

RESUME_FROM="${RESUME_FROM:-}"

DETERMINISTIC_VAL="${DETERMINISTIC_VAL:-true}"

if [ -z "${START_STEP}" ]; then
  if [ -n "${RESUME_FROM}" ] && [[ "${RESUME_FROM}" =~ global_step_([0-9]+)$ ]]; then
    START_STEP="${BASH_REMATCH[1]}"
  else
    START_STEP=0
  fi
fi

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

if [ "${DETERMINISTIC_VAL}" = "true" ]; then
  # Overrides placed after grpo_train.sh's own hard-coded
  # val_kwargs.temperature=0.4 / do_sample=True: Hydra/OmegaConf CLI
  # overrides resolve last-value-wins for a repeated key, and these are
  # appended as trailing "$@" args to grpo_train.sh, so they take effect.
  EXTRA_ARGS+=(
    "actor_rollout_ref.rollout.val_kwargs.temperature=0"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=False"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
  )
fi

echo "=== PRHBench phase: ${EXPERIMENT_NAME} (phase=${PHASE_NAME}) ==="
echo "    env=${ENV_NAME} model=${MODEL_PATH} seed=${SEED}"
echo "    prh.enabled=${PRH_ENABLED} prh.poison_prob=${PRH_POISON_PROB} prh.poison_seed=${PRH_POISON_SEED}"
echo "    start_step=${START_STEP} total_epochs=${TOTAL_EPOCHS} resume_from=${RESUME_FROM:-<none, fresh start>}"
echo "    deterministic_val=${DETERMINISTIC_VAL}"

# --- Run manifest -----------------------------------------------------
# Written before training starts (so it exists even if the run is later
# killed) into the same directory grpo_train.sh will write checkpoints to.
CKPTS_DIR="${UPSTREAM_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${CKPTS_DIR}"

PRHBENCH_COMMIT="$(git -C "${HERE}/.." rev-parse HEAD 2>/dev/null || echo unknown)"
UPSTREAM_COMMIT="$(git -C "${UPSTREAM_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
RUN_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat > "${CKPTS_DIR}/prhbench_manifest.json" <<EOF
{
  "prhbench_commit": "${PRHBENCH_COMMIT}",
  "upstream_commit": "${UPSTREAM_COMMIT}",
  "model": "${MODEL_PATH}",
  "environment": "${ENV_NAME}",
  "phase": "${PHASE_NAME}",
  "experiment_name": "${EXPERIMENT_NAME}",
  "project_name": "${PROJECT_NAME}",
  "parent_checkpoint": $( [ -n "${RESUME_FROM}" ] && printf '"%s"' "${RESUME_FROM}" || echo null ),
  "start_step": ${START_STEP},
  "end_step": ${TOTAL_EPOCHS},
  "rho_nominal": ${PRH_POISON_PROB},
  "prh_enabled": ${PRH_ENABLED},
  "poison_seed": ${PRH_POISON_SEED},
  "env_seed": ${SEED},
  "deterministic_val": ${DETERMINISTIC_VAL},
  "n_gpus": ${N_GPUS},
  "created_at": "${RUN_TIMESTAMP}"
}
EOF
echo "    manifest: ${CKPTS_DIR}/prhbench_manifest.json"

cd "${UPSTREAM_DIR}"
ENV_NAME="${ENV_NAME}" \
MODEL_PATH="${MODEL_PATH}" \
SEED="${SEED}" \
N_GPUS="${N_GPUS}" \
PROJECT_NAME="${PROJECT_NAME}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
bash grpo_train.sh "${EXTRA_ARGS[@]}" "$@"
