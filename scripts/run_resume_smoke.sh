#!/bin/bash
# GPU Gate 2: checkpoint/resume diagnostic.
#
# Not a pass/fail unit test -- a diagnostic to run and eyeball before the
# real pilot. Runs the same 0->20-update clean (rho=0) segment two ways:
#   A: one continuous 20-update run
#   B: 0->10, checkpoint, then resume 10->20 (two separate processes)
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
# Usage:
#   HF_TOKEN=... N_GPUS=2 bash PRHBench/scripts/run_resume_smoke.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-$(cd "${HERE}/../upstream" && pwd)}"

ENV_NAME="${ENV_NAME:-AbsentSupervisor}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
SEED="${SEED:-17}"
N_GPUS="${N_GPUS:-2}"
PROJECT_NAME="${PROJECT_NAME:-prhbench_gate2_resume}"

echo "################################################################"
echo "# Branch A -- continuous: 0 -> 20 updates, rho=0, one process"
echo "################################################################"
ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" SEED="${SEED}" N_GPUS="${N_GPUS}" \
PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="continuous_0_20" PHASE_NAME="resume_diagnostic_a_continuous" \
TOTAL_EPOCHS=20 SAVE_FREQ=5 TEST_FREQ=5 \
PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
bash "${HERE}/run_phase.sh"

echo "################################################################"
echo "# Branch B, part 1 -- 0 -> 10 updates, rho=0"
echo "################################################################"
ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" SEED="${SEED}" N_GPUS="${N_GPUS}" \
PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="segmented_0_10" PHASE_NAME="resume_diagnostic_b_part1" \
TOTAL_EPOCHS=10 SAVE_FREQ=5 TEST_FREQ=5 \
PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
bash "${HERE}/run_phase.sh"

SEGMENT_CKPT="${UPSTREAM_DIR}/checkpoints/${PROJECT_NAME}/segmented_0_10/global_step_10"
echo "################################################################"
echo "# Branch B, part 2 -- resume from ${SEGMENT_CKPT}, 10 -> 20 updates"
echo "################################################################"
ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" SEED="${SEED}" N_GPUS="${N_GPUS}" \
PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="segmented_10_20" PHASE_NAME="resume_diagnostic_b_part2" \
TOTAL_EPOCHS=20 SAVE_FREQ=5 TEST_FREQ=5 \
PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
RESUME_FROM="${SEGMENT_CKPT}" \
bash "${HERE}/run_phase.sh"

echo "################################################################"
echo "# Gate 2 complete. Compare (via WandB or the console logs of each run):"
echo "#   continuous_0_20            (branch A, updates 0..20)"
echo "#   segmented_0_10 + segmented_10_20  (branch B, updates 0..10 then 10..20)"
echo "# on val/cumulative_hidden_reward_mean and episode/hidden_reward_mean."
echo "# Expect similar learning curves and similar final validation reward,"
echo "# NOT bitwise-identical trajectories (the environment RNG is not"
echo "# checkpointed -- see this script's header comment)."
echo "################################################################"
