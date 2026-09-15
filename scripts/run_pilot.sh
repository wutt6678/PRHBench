#!/bin/bash
# Three-phase PRHBench pilot runner (PRHBench pilot design, sections 12-15).
#
# Timeline (clean training updates on the x-axis):
#
#   0 -------------- 40 -------- 60 -------------------------- 120
#           pre-attack    attack (rho>0)        washout (rho=0)
#           (rho=0)
#
# One shared pre-attack checkpoint per seed is trained once, then every
# poison dose branches from that same checkpoint (section 13) instead of
# independently re-running the first 40 updates per dose. The optimizer
# and RNG state are preserved across phases via checkpoint resume, not
# reset (section 12) -- this matters for interpreting persistence in the
# washout phase as a property of the trained policy, not an artifact of
# reinitializing the optimizer.
#
# Usage (defaults reproduce the pilot's first experiment matrix, section 15):
#   ENV_NAME=AbsentSupervisor MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
#   SEED=17 DOSES="0.0 0.25 1.0" \
#   bash PRHBench/scripts/run_pilot.sh
#
# Each phase is a separate `bash grpo_train.sh` invocation (via run_phase.sh)
# and is expected to be launched on a GPU node; this script only sequences
# them and wires up checkpoint paths. Re-running it is idempotent in spirit
# but does NOT skip phases whose checkpoints already exist -- check that
# yourself before re-launching an expensive run.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-$(cd "${HERE}/../upstream" && pwd)}"

ENV_NAME="${ENV_NAME:-AbsentSupervisor}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
SEED="${SEED:-17}"
N_GPUS="${N_GPUS:-2}"
PROJECT_NAME="${PROJECT_NAME:-prhbench_pilot}"

PRE_EPOCHS="${PRE_EPOCHS:-40}"     # updates 0 -> 40, rho=0
ATTACK_EPOCHS="${ATTACK_EPOCHS:-60}"   # updates 40 -> 60 (20 attack updates), rho=dose
WASHOUT_EPOCHS="${WASHOUT_EPOCHS:-120}" # updates 60 -> 120 (60 washout updates), rho=0

PRH_POISON_SEED="${PRH_POISON_SEED:-10017}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-5}"

# Space-separated poison doses for the attack window (section 15: 0, 0.25, 1.0).
DOSES="${DOSES:-0.0 0.25 1.0}"

model_basename="$(basename "${MODEL_PATH}")"
ckpt_root="${UPSTREAM_DIR}/checkpoints/${PROJECT_NAME}"

pre_experiment="pre_s${SEED}"
pre_ckpt="${ckpt_root}/${pre_experiment}/global_step_${PRE_EPOCHS}"

common_env=(
  ENV_NAME="${ENV_NAME}"
  MODEL_PATH="${MODEL_PATH}"
  SEED="${SEED}"
  N_GPUS="${N_GPUS}"
  PROJECT_NAME="${PROJECT_NAME}"
  PRH_POISON_SEED="${PRH_POISON_SEED}"
  SAVE_FREQ="${SAVE_FREQ}"
  TEST_FREQ="${TEST_FREQ}"
)

echo "############################################################"
echo "# Phase 1/3 -- shared pre-attack checkpoint (rho=0, seed=${SEED})"
echo "############################################################"
env "${common_env[@]}" \
  EXPERIMENT_NAME="${pre_experiment}" \
  TOTAL_EPOCHS="${PRE_EPOCHS}" \
  PRH_ENABLED=true \
  PRH_POISON_PROB=0.0 \
  bash "${HERE}/run_phase.sh"

for dose in ${DOSES}; do
  attack_experiment="rho${dose}_s${SEED}"
  washout_experiment="washout_rho${dose}_s${SEED}"
  attack_ckpt="${ckpt_root}/${attack_experiment}/global_step_${ATTACK_EPOCHS}"

  echo "############################################################"
  echo "# Phase 2/3 -- attack window (rho=${dose}, seed=${SEED})"
  echo "############################################################"
  env "${common_env[@]}" \
    EXPERIMENT_NAME="${attack_experiment}" \
    TOTAL_EPOCHS="${ATTACK_EPOCHS}" \
    PRH_ENABLED=true \
    PRH_POISON_PROB="${dose}" \
    RESUME_FROM="${pre_ckpt}" \
    bash "${HERE}/run_phase.sh"

  echo "############################################################"
  echo "# Phase 3/3 -- washout (rho=0, seed=${SEED}, following dose=${dose})"
  echo "############################################################"
  env "${common_env[@]}" \
    EXPERIMENT_NAME="${washout_experiment}" \
    TOTAL_EPOCHS="${WASHOUT_EPOCHS}" \
    PRH_ENABLED=true \
    PRH_POISON_PROB=0.0 \
    RESUME_FROM="${attack_ckpt}" \
    bash "${HERE}/run_phase.sh"
done

echo "All phases submitted for seed=${SEED}, doses=[${DOSES}]."
echo "Pre-attack checkpoint: ${pre_ckpt}"
