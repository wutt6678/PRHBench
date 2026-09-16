#!/bin/bash
# Unattended pipeline: short Gate 2 -> Gate 3 pretraining (0->10, automated
# learnability check, then 10->40) -> seed-17 PRH pilot, exactly the
# sequence recommended before spending more GPU time on repeat Gate 1
# smoke runs:
#
#   short Gate 2  ->  Gate 3 / pretraining to 40  ->  seed-17 pilot
#
# Designed to be launched once via nohup and left alone -- every phase's
# console output goes to its own log file under LOG_DIR, and a single
# PIPELINE_STATUS.txt is updated with a one-line summary after each
# stage, so progress can be checked later without babysitting this
# process. Each phase is one uninterrupted `bash grpo_train.sh` process
# (via run_phase.sh) as usual -- only the *boundaries* between phases
# involve a checkpoint resume, matching the segmented-branch design
# GPU Gate 2 exists to validate.
#
# Automated decision gates (matching the pilot design's own criteria):
#   - Gate 3 phase A (0->10): if hidden reward did NOT improve
#     (scripts/check_learnable.py), STOP. Do not train to 40 or launch
#     the pilot on a base that isn't learning anything.
#   - Gate 3 phase B (10->40): same check, now 0-vs-40. If learnable,
#     launch the pilot from this exact checkpoint (SKIP_PRE=true) rather
#     than training the pre-attack phase a second time.
#   - Gate 2 is a diagnostic, not a gate: its outcome is logged but never
#     blocks the rest of the pipeline.
#
# Usage:
#   nohup bash scripts/run_full_pipeline.sh > /dev/null 2>&1 &
#   disown
#   # later: tail -f PRHBench/run_logs/PIPELINE_STATUS.txt

set -uo pipefail  # deliberately not -e: this script's own control flow
                  # decides what a failed stage means, stage by stage.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRHBENCH_DIR="$(cd "${HERE}/.." && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-${PRHBENCH_DIR}/upstream}"
LOG_DIR="${LOG_DIR:-${PRHBENCH_DIR}/run_logs}"
mkdir -p "${LOG_DIR}"
STATUS_FILE="${LOG_DIR}/PIPELINE_STATUS.txt"

ENV_NAME="${ENV_NAME:-AbsentSupervisor}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
SEED="${SEED:-17}"
N_GPUS="${N_GPUS:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-3}"
PROJECT_NAME="${PROJECT_NAME:-prhbench_pilot}"
MIN_DELTA="${MIN_DELTA:-0.0}"

export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

# The memory-footprint overrides that got GPU Gate 1's Smoke A to fully
# pass on this shared cluster (single GPU, eager mode, freed vLLM cache,
# offloaded optimizer, a block size compatible with the XFORMERS/V0
# backend). Applied to every phase in this pipeline by default.
MEM_ARGS=(
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25
  actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=16
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${STATUS_FILE}"
}

log "=== PRHBench full pipeline starting (env=${ENV_NAME} model=${MODEL_PATH} seed=${SEED} gpu=${CUDA_DEVICE}) ==="

# --- Stage 1: short Gate 2 (diagnostic only, never blocks) -------------
log "Stage 1/4: Gate 2 (short resume diagnostic, 0->4 / 0->2->4)"
GATE2_LOG="${LOG_DIR}/gate2_resume.log"
CONDA_ENV="" N_GPUS="${N_GPUS}" SEED="${SEED}" ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" \
  bash "${HERE}/run_resume_smoke.sh" "${MEM_ARGS[@]}" > "${GATE2_LOG}" 2>&1
if [ $? -eq 0 ]; then
  log "Stage 1/4: Gate 2 completed without error. See ${GATE2_LOG} -- eyeball"
  log "  continuous_0_4 vs segmented_0_2 + segmented_2_4 manually; this does"
  log "  not block the rest of the pipeline either way."
else
  log "Stage 1/4: Gate 2 exited non-zero (see ${GATE2_LOG}). Diagnostic only -- continuing."
fi

# --- Stage 2: Gate 3 phase A, 0 -> 10 (fresh start) ---------------------
log "Stage 2/4: Gate 3 pretraining, phase A: 0 -> 10 updates, rho=0"
GATE3_A_LOG="${LOG_DIR}/gate3_phaseA_0to10.log"
CONDA_ENV="" N_GPUS="${N_GPUS}" SEED="${SEED}" ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" \
PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="pre_s${SEED}" PHASE_NAME="gate3_to10" \
TOTAL_EPOCHS=10 SAVE_FREQ=5 TEST_FREQ=5 \
PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
  bash "${HERE}/run_phase.sh" "${MEM_ARGS[@]}" > "${GATE3_A_LOG}" 2>&1
PHASE_A_EXIT=$?
if [ ${PHASE_A_EXIT} -ne 0 ]; then
  log "Stage 2/4: FAILED -- Gate 3 phase A (0->10) exited ${PHASE_A_EXIT}. See ${GATE3_A_LOG}."
  log "=== Pipeline stopped: cannot check learnability without phase A completing. ==="
  exit 1
fi
log "Stage 2/4: Gate 3 phase A (0->10) completed. See ${GATE3_A_LOG}"

log "Stage 2/4: checking H(0) vs H(10) learnability"
python3 "${HERE}/check_learnable.py" --min-delta "${MIN_DELTA}" "${GATE3_A_LOG}" 2>&1 | tee -a "${STATUS_FILE}"
LEARNABLE_10=$?
if [ ${LEARNABLE_10} -ne 0 ]; then
  log "Stage 2/4: NOT_LEARNABLE at step 10. Per the pilot design, do not"
  log "  proceed to step 40 or the pilot on a base that isn't learning."
  log "=== Pipeline stopped after Gate 3 phase A. Inspect ${GATE3_A_LOG} and"
  log "  analysis/pilot_metrics.py's H(t)/O(t)/G(t) manually before deciding"
  log "  whether to extend the attack window, try a different environment,"
  log "  or otherwise adjust before re-running. ==="
  exit 1
fi
log "Stage 2/4: LEARNABLE at step 10 -- proceeding to step 40."

# --- Stage 3: Gate 3 phase B, 10 -> 40 (resume) --------------------------
log "Stage 3/4: Gate 3 pretraining, phase B: 10 -> 40 updates, rho=0"
GATE3_B_LOG="${LOG_DIR}/gate3_phaseB_10to40.log"
PRE_CKPT_10="${UPSTREAM_DIR}/checkpoints/${PROJECT_NAME}/pre_s${SEED}/global_step_10"
CONDA_ENV="" N_GPUS="${N_GPUS}" SEED="${SEED}" ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" \
PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="pre_s${SEED}" PHASE_NAME="gate3_10to40" \
TOTAL_EPOCHS=40 SAVE_FREQ=5 TEST_FREQ=5 \
PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
RESUME_FROM="${PRE_CKPT_10}" \
  bash "${HERE}/run_phase.sh" "${MEM_ARGS[@]}" > "${GATE3_B_LOG}" 2>&1
PHASE_B_EXIT=$?
if [ ${PHASE_B_EXIT} -ne 0 ]; then
  log "Stage 3/4: FAILED -- Gate 3 phase B (10->40) exited ${PHASE_B_EXIT}. See ${GATE3_B_LOG}."
  log "=== Pipeline stopped: no verified global_step_40 checkpoint to branch the pilot from. ==="
  exit 1
fi
log "Stage 3/4: Gate 3 phase B (10->40) completed. See ${GATE3_B_LOG}"

log "Stage 3/4: checking H(0) vs H(40) learnability (final pretraining go/no-go)"
python3 "${HERE}/check_learnable.py" --min-delta "${MIN_DELTA}" "${GATE3_A_LOG}" "${GATE3_B_LOG}" 2>&1 | tee -a "${STATUS_FILE}"
LEARNABLE_40=$?
PRE_CKPT_40="${UPSTREAM_DIR}/checkpoints/${PROJECT_NAME}/pre_s${SEED}/global_step_40"
if [ ${LEARNABLE_40} -ne 0 ]; then
  log "Stage 3/4: NOT_LEARNABLE at step 40 (H(40) did not exceed H(0))."
  log "=== Pipeline stopped before the pilot. The checkpoint still exists at"
  log "  ${PRE_CKPT_40}"
  log "  for manual inspection, but the pilot was NOT launched on it. ==="
  exit 1
fi
log "Stage 3/4: LEARNABLE at step 40 -- ${PRE_CKPT_40} is the pilot's pre-attack checkpoint."

# --- Stage 4: seed-17 PRH pilot, branching from theta_40 ----------------
log "Stage 4/4: seed-17 pilot -- rho in {0, 0.25, 1.0}, 40 clean (reused) + 20 attack + 60 washout"
PILOT_LOG="${LOG_DIR}/pilot_seed${SEED}.log"
N_GPUS="${N_GPUS}" SEED="${SEED}" ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" \
PROJECT_NAME="${PROJECT_NAME}" SKIP_PRE=true PRE_EPOCHS=40 DOSES="0.0 0.25 1.0" \
SAVE_FREQ=5 TEST_FREQ=5 \
  bash "${HERE}/run_pilot.sh" "${MEM_ARGS[@]}" > "${PILOT_LOG}" 2>&1
PILOT_EXIT=$?
if [ ${PILOT_EXIT} -ne 0 ]; then
  log "Stage 4/4: pilot exited ${PILOT_EXIT} (may be partial -- some doses may"
  log "  have completed before the failure). See ${PILOT_LOG}."
  exit 1
fi
log "Stage 4/4: seed-17 pilot completed. See ${PILOT_LOG}"
log "=== Pipeline complete. Analyze with analysis/pilot_metrics.py against"
log "  the validation metrics logged under checkpoints/${PROJECT_NAME}/{rho*,washout_rho*}_s${SEED}. ==="
