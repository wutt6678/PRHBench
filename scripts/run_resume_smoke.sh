#!/bin/bash
# GPU Gate 2: checkpoint/resume diagnostic.
#
# Not a pass/fail unit test -- a diagnostic to run and eyeball before the
# real pilot. Runs the same 0->TOTAL_EPOCHS-update clean (rho=0) segment
# two ways:
#   A: one continuous TOTAL_EPOCHS-update run
#   B: 0->SPLIT_EPOCHS, checkpoint, then resume SPLIT_EPOCHS->TOTAL_EPOCHS
#      (two separate processes)
#
# The environment RNG is NOT checkpointed (a fresh run_phase.sh process
# re-seeds numpy on environment construction), so branch B is not expected
# to be bitwise-identical to branch A -- compare learning curves and final
# validation reward instead. If they diverge dramatically, a phase-stable
# environment-seed schedule should be implemented before trusting the
# segmented pilot's phase boundaries; if they track each other reasonably,
# the segmented-branch design (every dose compared against a matched rho=0
# branch from the same checkpoint, not against an uninterrupted run) is
# sound as-is and no further work is needed before the pilot.
#
# Usage (defaults to a short 0->4 / 0->2->4 diagnostic; the original
# pilot-scale 0->20 / 0->10->20 diagnostic is TOTAL_EPOCHS=20 SPLIT_EPOCHS=10):
#   HF_TOKEN=... N_GPUS=2 bash PRHBench/scripts/run_resume_smoke.sh
#   HF_TOKEN=... TOTAL_EPOCHS=20 SPLIT_EPOCHS=10 SAVE_FREQ=5 TEST_FREQ=5 \
#     bash PRHBench/scripts/run_resume_smoke.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-$(cd "${HERE}/../upstream" && pwd)}"

ENV_NAME="${ENV_NAME:-AbsentSupervisor}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
SEED="${SEED:-17}"
N_GPUS="${N_GPUS:-2}"
PROJECT_NAME="${PROJECT_NAME:-prhbench_gate2_resume}"

# Short diagnostic by default (this only needs to prove resume behaves
# reasonably, not measure anything scientific): 0->4 continuous vs.
# 0->2->4 segmented. Override to TOTAL_EPOCHS=20 SPLIT_EPOCHS=10 to
# reproduce the original pilot-scale version.
TOTAL_EPOCHS="${TOTAL_EPOCHS:-4}"
SPLIT_EPOCHS="${SPLIT_EPOCHS:-2}"
SAVE_FREQ="${SAVE_FREQ:-2}"
TEST_FREQ="${TEST_FREQ:-2}"

if [ "${SPLIT_EPOCHS}" -ge "${TOTAL_EPOCHS}" ]; then
  echo "SPLIT_EPOCHS (${SPLIT_EPOCHS}) must be < TOTAL_EPOCHS (${TOTAL_EPOCHS})" >&2
  exit 1
fi

CONTINUOUS_NAME="continuous_0_${TOTAL_EPOCHS}"
SEGMENT1_NAME="segmented_0_${SPLIT_EPOCHS}"
SEGMENT2_NAME="segmented_${SPLIT_EPOCHS}_${TOTAL_EPOCHS}"

echo "################################################################"
echo "# Branch A -- continuous: 0 -> ${TOTAL_EPOCHS} updates, rho=0, one process"
echo "################################################################"
ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" SEED="${SEED}" N_GPUS="${N_GPUS}" \
PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="${CONTINUOUS_NAME}" PHASE_NAME="resume_diagnostic_a_continuous" \
TOTAL_EPOCHS="${TOTAL_EPOCHS}" SAVE_FREQ="${SAVE_FREQ}" TEST_FREQ="${TEST_FREQ}" \
PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
bash "${HERE}/run_phase.sh" "$@"

echo "################################################################"
echo "# Branch B, part 1 -- 0 -> ${SPLIT_EPOCHS} updates, rho=0"
echo "################################################################"
ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" SEED="${SEED}" N_GPUS="${N_GPUS}" \
PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="${SEGMENT1_NAME}" PHASE_NAME="resume_diagnostic_b_part1" \
TOTAL_EPOCHS="${SPLIT_EPOCHS}" SAVE_FREQ="${SAVE_FREQ}" TEST_FREQ="${TEST_FREQ}" \
PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
bash "${HERE}/run_phase.sh" "$@"

SEGMENT_CKPT="${UPSTREAM_DIR}/checkpoints/${PROJECT_NAME}/${SEGMENT1_NAME}/global_step_${SPLIT_EPOCHS}"
echo "################################################################"
echo "# Branch B, part 2 -- resume from ${SEGMENT_CKPT}, ${SPLIT_EPOCHS} -> ${TOTAL_EPOCHS} updates"
echo "################################################################"
ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" SEED="${SEED}" N_GPUS="${N_GPUS}" \
PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="${SEGMENT2_NAME}" PHASE_NAME="resume_diagnostic_b_part2" \
TOTAL_EPOCHS="${TOTAL_EPOCHS}" SAVE_FREQ="${SAVE_FREQ}" TEST_FREQ="${TEST_FREQ}" \
PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
RESUME_FROM="${SEGMENT_CKPT}" \
bash "${HERE}/run_phase.sh" "$@"

echo "################################################################"
echo "# Gate 2 complete. Compare (via WandB or the console logs of each run):"
echo "#   ${CONTINUOUS_NAME}                     (branch A, updates 0..${TOTAL_EPOCHS})"
echo "#   ${SEGMENT1_NAME} + ${SEGMENT2_NAME}  (branch B, updates 0..${SPLIT_EPOCHS} then ${SPLIT_EPOCHS}..${TOTAL_EPOCHS})"
echo "# Check specifically that the resumed process (branch B part 2) logs"
echo "# loading global_step_${SPLIT_EPOCHS}, that optimizer/scheduler state loads"
echo "# without error, that training continues to ${TOTAL_EPOCHS}, and that final"
echo "# val/cumulative_hidden_reward_mean is broadly consistent between branches"
echo "# -- on episode/hidden_reward_mean and val/cumulative_hidden_reward_mean."
echo "# Do NOT require identical weights or trajectories (the environment RNG"
echo "# is not checkpointed -- see this script's header comment)."
echo "################################################################"
