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
# GPU selection on this cluster is genuinely volatile: free memory has
# been observed to drop from ~15 GB to under 200 MB within the ~1-2
# minutes it takes a phase to start up (other tenants' usage spiking
# during that window, not just between checks). A single point-in-time
# "most free of 4" pick is not enough -- this script instead (a) requires
# a GPU to show at least MIN_FREE_MIB free before it will even try,
# waiting and re-polling all 4 GPUs if none currently qualify, and (b)
# retries a phase (re-picking a GPU each time) if it fails with vLLM's
# specific CUDA-out-of-memory signature, up to MAX_ATTEMPTS times. A
# failure that is NOT that OOM signature is never retried -- it almost
# certainly means a real bug, not transient contention, and retrying
# would just waste time reproducing it identically.
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
CUDA_DEVICE="${CUDA_DEVICE:-}"
PROJECT_NAME="${PROJECT_NAME:-prhbench_pilot}"
MIN_DELTA="${MIN_DELTA:-0.0}"

# GPU selection/backoff tuning. See header comment above for why both a
# minimum-free-memory threshold AND OOM-triggered retries are needed here.
AUTO_GPU="${AUTO_GPU:-true}"
if [ -n "${CUDA_DEVICE}" ]; then
  AUTO_GPU="false"
fi
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"       # ~20 GiB: comfortable margin above the
                                             # ~14 GiB peak a successful phase actually used
GPU_POLL_INTERVAL_SECONDS="${GPU_POLL_INTERVAL_SECONDS:-60}"
# GPU_MAX_WAIT_SECONDS=0 (the default) means wait indefinitely for a GPU
# to reach MIN_FREE_MIB free -- set a positive value to give up after
# that many seconds and proceed with whatever's best instead.
GPU_MAX_WAIT_SECONDS="${GPU_MAX_WAIT_SECONDS:-0}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"           # retries specifically for the CUDA-OOM signature
RETRY_BACKOFF_SECONDS="${RETRY_BACKOFF_SECONDS:-60}"

export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export WANDB_MODE="${WANDB_MODE:-offline}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${STATUS_FILE}"
}

best_gpu_and_free_mib() {
  # Prints "<index> <free_mib>" for the GPU with the most free memory right now.
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | tr -d ' ' | sort -t',' -k2 -n -r | head -1 | tr ',' ' '
}

select_gpu_for_stage() {
  # Sets CUDA_VISIBLE_DEVICES for the next attempt. If AUTO_GPU=true,
  # waits (polling all 4 GPUs) until one shows >= MIN_FREE_MIB free.
  # With GPU_MAX_WAIT_SECONDS=0 (the default), this waits indefinitely;
  # set it positive to give up after that many seconds and proceed with
  # whatever's best instead.
  local stage_name="$1"
  if [ "${AUTO_GPU}" != "true" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    log "${stage_name}: using fixed GPU ${CUDA_DEVICE} (AUTO_GPU=false)"
    return
  fi

  local waited=0
  local best_idx best_free
  while true; do
    read -r best_idx best_free <<< "$(best_gpu_and_free_mib)"
    if [ "${best_free}" -ge "${MIN_FREE_MIB}" ]; then
      break
    fi
    if [ "${GPU_MAX_WAIT_SECONDS}" -gt 0 ] && [ "${waited}" -ge "${GPU_MAX_WAIT_SECONDS}" ]; then
      log "${stage_name}: no GPU reached ${MIN_FREE_MIB} MiB free after ${waited}s of waiting; proceeding with GPU ${best_idx} (${best_free} MiB free) anyway."
      break
    fi
    local wait_limit_desc="${GPU_MAX_WAIT_SECONDS}s"
    [ "${GPU_MAX_WAIT_SECONDS}" -eq 0 ] && wait_limit_desc="no limit"
    log "${stage_name}: no GPU has ${MIN_FREE_MIB} MiB free yet (best: GPU ${best_idx} with ${best_free} MiB); waiting ${GPU_POLL_INTERVAL_SECONDS}s (${waited}s elapsed, ${wait_limit_desc})..."
    sleep "${GPU_POLL_INTERVAL_SECONDS}"
    waited=$((waited + GPU_POLL_INTERVAL_SECONDS))
  done

  CUDA_DEVICE="${best_idx}"
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
  log "${stage_name}: using GPU ${CUDA_DEVICE} (${best_free} MiB free at selection time)"
}

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

run_phase_with_retry() {
  # run_phase_with_retry <stage_name> <log_path> <run_phase.sh env assignments...> -- <run_phase.sh args...>
  # Retries only on vLLM's specific CUDA-out-of-memory signature; any
  # other failure returns immediately (non-OOM failures are almost
  # certainly real bugs, not transient contention -- retrying them would
  # just reproduce the same error and waste time).
  local stage_name="$1"; shift
  local log_path="$1"; shift
  local env_assignments=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    env_assignments+=("$1")
    shift
  done
  shift  # drop the "--" separator
  local phase_args=("$@")

  local attempt=1
  while [ "${attempt}" -le "${MAX_ATTEMPTS}" ]; do
    select_gpu_for_stage "${stage_name} (attempt ${attempt}/${MAX_ATTEMPTS})"
    env "${env_assignments[@]}" bash "${HERE}/run_phase.sh" "${phase_args[@]}" "${MEM_ARGS[@]}" > "${log_path}" 2>&1
    local exit_code=$?
    if [ "${exit_code}" -eq 0 ]; then
      return 0
    fi
    if grep -q "torch.OutOfMemoryError\|CUDA out of memory" "${log_path}" 2>/dev/null; then
      log "${stage_name}: attempt ${attempt}/${MAX_ATTEMPTS} hit a CUDA OOM on GPU ${CUDA_DEVICE} (see ${log_path}). Retrying on a freshly-selected GPU after ${RETRY_BACKOFF_SECONDS}s..."
      sleep "${RETRY_BACKOFF_SECONDS}"
      attempt=$((attempt + 1))
      continue
    else
      log "${stage_name}: FAILED with a non-OOM error (exit ${exit_code}) -- not retrying. See ${log_path}."
      return "${exit_code}"
    fi
  done
  log "${stage_name}: still failing with CUDA OOM after ${MAX_ATTEMPTS} attempts. See ${log_path}."
  return 1
}

log "=== PRHBench full pipeline starting (env=${ENV_NAME} model=${MODEL_PATH} seed=${SEED} gpu=$([ "${AUTO_GPU}" = "true" ] && echo "auto" || echo "${CUDA_DEVICE}") min_free_mib=${MIN_FREE_MIB} max_attempts=${MAX_ATTEMPTS}) ==="

# --- Stage 1: short Gate 2 (diagnostic only, never blocks) -------------
log "Stage 1/4: Gate 2 (short resume diagnostic, 0->4 / 0->2->4)"
GATE2_LOG="${LOG_DIR}/gate2_resume.log"
attempt=1
while [ "${attempt}" -le "${MAX_ATTEMPTS}" ]; do
  select_gpu_for_stage "Stage 1/4 (attempt ${attempt}/${MAX_ATTEMPTS})"
  CONDA_ENV="" N_GPUS="${N_GPUS}" SEED="${SEED}" ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" \
    bash "${HERE}/run_resume_smoke.sh" "${MEM_ARGS[@]}" > "${GATE2_LOG}" 2>&1
  GATE2_EXIT=$?
  if [ "${GATE2_EXIT}" -eq 0 ]; then
    break
  fi
  if grep -q "torch.OutOfMemoryError\|CUDA out of memory" "${GATE2_LOG}" 2>/dev/null && [ "${attempt}" -lt "${MAX_ATTEMPTS}" ]; then
    log "Stage 1/4: attempt ${attempt} hit a CUDA OOM; retrying after ${RETRY_BACKOFF_SECONDS}s..."
    sleep "${RETRY_BACKOFF_SECONDS}"
  else
    break
  fi
  attempt=$((attempt + 1))
done
if [ "${GATE2_EXIT}" -eq 0 ]; then
  log "Stage 1/4: Gate 2 completed without error. See ${GATE2_LOG} -- eyeball"
  log "  continuous_0_4 vs segmented_0_2 + segmented_2_4 manually; this does"
  log "  not block the rest of the pipeline either way."
else
  log "Stage 1/4: Gate 2 exited non-zero (see ${GATE2_LOG}). Diagnostic only -- continuing."
fi

# --- Stage 2: Gate 3 phase A, 0 -> 10 (fresh start) ---------------------
log "Stage 2/4: Gate 3 pretraining, phase A: 0 -> 10 updates, rho=0"
GATE3_A_LOG="${LOG_DIR}/gate3_phaseA_0to10.log"
run_phase_with_retry "Stage 2/4" "${GATE3_A_LOG}" \
  CONDA_ENV="" N_GPUS="${N_GPUS}" SEED="${SEED}" ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" \
  PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="pre_s${SEED}" PHASE_NAME="gate3_to10" \
  TOTAL_EPOCHS=10 SAVE_FREQ=5 TEST_FREQ=5 \
  PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
  --
PHASE_A_EXIT=$?
if [ ${PHASE_A_EXIT} -ne 0 ]; then
  log "Stage 2/4: FAILED -- Gate 3 phase A (0->10) did not complete. See ${GATE3_A_LOG}."
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
run_phase_with_retry "Stage 3/4" "${GATE3_B_LOG}" \
  CONDA_ENV="" N_GPUS="${N_GPUS}" SEED="${SEED}" ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" \
  PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="pre_s${SEED}" PHASE_NAME="gate3_10to40" \
  TOTAL_EPOCHS=40 SAVE_FREQ=5 TEST_FREQ=5 \
  PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
  RESUME_FROM="${PRE_CKPT_10}" \
  --
PHASE_B_EXIT=$?
if [ ${PHASE_B_EXIT} -ne 0 ]; then
  log "Stage 3/4: FAILED -- Gate 3 phase B (10->40) did not complete. See ${GATE3_B_LOG}."
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
# Inlined (rather than delegating to run_pilot.sh) so each of the 6
# attack/washout phases below gets its own retry-with-fresh-GPU-pick via
# run_phase_with_retry -- the pilot is the longest, most crash-prone part
# (each phase 20-60 real GPU-minutes), so a single check for the whole
# stage would barely help.
log "Stage 4/4: seed-17 pilot -- rho in {0, 0.25, 1.0}, 40 clean (reused) + 20 attack + 60 washout"
ATTACK_EPOCHS=60
WASHOUT_EPOCHS=120
DOSES="0.0 0.25 1.0"
PILOT_FAILED=false
for dose in ${DOSES}; do
  attack_experiment="rho${dose}_s${SEED}"
  washout_experiment="washout_rho${dose}_s${SEED}"
  attack_ckpt="${UPSTREAM_DIR}/checkpoints/${PROJECT_NAME}/${attack_experiment}/global_step_${ATTACK_EPOCHS}"

  ATTACK_LOG="${LOG_DIR}/pilot_rho${dose}_attack.log"
  run_phase_with_retry "Stage 4/4 (rho=${dose} attack)" "${ATTACK_LOG}" \
    N_GPUS="${N_GPUS}" SEED="${SEED}" ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" \
    PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="${attack_experiment}" PHASE_NAME="attack" \
    TOTAL_EPOCHS="${ATTACK_EPOCHS}" SAVE_FREQ=5 TEST_FREQ=5 \
    PRH_ENABLED=true PRH_POISON_PROB="${dose}" PRH_POISON_SEED=10017 \
    RESUME_FROM="${PRE_CKPT_40}" \
    --
  if [ $? -ne 0 ]; then
    log "Stage 4/4: FAILED -- rho=${dose} attack phase did not complete. See ${ATTACK_LOG}."
    PILOT_FAILED=true
    continue
  fi
  log "Stage 4/4: rho=${dose} attack (40->${ATTACK_EPOCHS}) completed. See ${ATTACK_LOG}"

  WASHOUT_LOG="${LOG_DIR}/pilot_rho${dose}_washout.log"
  run_phase_with_retry "Stage 4/4 (rho=${dose} washout)" "${WASHOUT_LOG}" \
    N_GPUS="${N_GPUS}" SEED="${SEED}" ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" \
    PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="${washout_experiment}" PHASE_NAME="washout" \
    TOTAL_EPOCHS="${WASHOUT_EPOCHS}" SAVE_FREQ=5 TEST_FREQ=5 \
    PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
    RESUME_FROM="${attack_ckpt}" \
    --
  if [ $? -ne 0 ]; then
    log "Stage 4/4: FAILED -- rho=${dose} washout phase did not complete. See ${WASHOUT_LOG}."
    PILOT_FAILED=true
    continue
  fi
  log "Stage 4/4: rho=${dose} washout (${ATTACK_EPOCHS}->${WASHOUT_EPOCHS}) completed. See ${WASHOUT_LOG}"
done

if [ "${PILOT_FAILED}" = "true" ]; then
  log "Stage 4/4: one or more doses failed (see per-dose logs above) -- some may"
  log "  have completed successfully regardless. Re-run just the failed"
  log "  dose/phase manually (RESUME_FROM the last good checkpoint) rather"
  log "  than the whole pipeline."
  exit 1
fi
log "Stage 4/4: seed-17 pilot completed for all doses [${DOSES}]."
log "=== Pipeline complete. Analyze with analysis/pilot_metrics.py against"
log "  the validation metrics logged under checkpoints/${PROJECT_NAME}/{rho*,washout_rho*}_s${SEED}. ==="
