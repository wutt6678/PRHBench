#!/bin/bash
# GPU Gate 1: end-to-end smoke test. Proves the whole path
#   Gridworld -> PRH router -> trajectory collector -> GRPO -> FSDP
#   checkpoint -> resume
# actually works, BEFORE attempting the real 40+20+60 pilot. Each of the
# three runs below is tiny (a handful of updates) and independent (fresh
# start each time, no resume between them) -- this gate is about proving
# the pipeline runs end to end, not about measuring anything scientific.
#
# Usage:
#   HF_TOKEN=... N_GPUS=2 bash PRHBench/scripts/run_grpo_smoke.sh
#   # extra Hydra overrides (e.g. a lower gpu_memory_utilization on a
#   # shared/contended GPU) are forwarded to every phase:
#   HF_TOKEN=... bash PRHBench/scripts/run_grpo_smoke.sh \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.25
#
# Requires the FULL training stack (prhbench-train environment: see
# environment-train.yml / scripts/setup_training_env.sh), not the
# lightweight `prhbench` smoke-test environment.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="${ENV_NAME:-AbsentSupervisor}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
SEED="${SEED:-17}"
N_GPUS="${N_GPUS:-2}"
PROJECT_NAME="${PROJECT_NAME:-prhbench_gate1_smoke}"

echo "################################################################"
echo "# Smoke A -- ordinary upstream (PRH disabled), 2 updates"
echo "################################################################"
echo "Acceptance: GRPO performs an optimizer update; validation completes;"
echo "a checkpoint is written; hidden/observed rewards appear in logs;"
echo "no prh_* keys appear anywhere (PRH is fully disabled)."
ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" SEED="${SEED}" N_GPUS="${N_GPUS}" \
PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="smoke_a_disabled" PHASE_NAME="smoke_a" \
TOTAL_EPOCHS=2 SAVE_FREQ=1 TEST_FREQ=1 \
PRH_ENABLED=false \
bash "${HERE}/run_phase.sh" "$@"

echo "################################################################"
echo "# Smoke B -- clean PRH training (rho=0), 3 updates"
echo "################################################################"
echo "Acceptance: r_train == hidden_reward every step (prh_poisoned=false"
echo "everywhere; prh_training_reward == prh_clean_reward == hidden_reward)."
echo "NOT the same as Smoke A: upstream-disabled trains on observed reward,"
echo "this trains on hidden reward instead -- that substitution is the point."
ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" SEED="${SEED}" N_GPUS="${N_GPUS}" \
PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="smoke_b_rho0" PHASE_NAME="smoke_b" \
TOTAL_EPOCHS=3 SAVE_FREQ=1 TEST_FREQ=1 \
PRH_ENABLED=true PRH_POISON_PROB=0.0 PRH_POISON_SEED=10017 \
bash "${HERE}/run_phase.sh" "$@"

echo "################################################################"
echo "# Smoke C -- full hijack (rho=1), 3 updates"
echo "################################################################"
echo "Acceptance: r_train == observed_reward every step (prh_poisoned=true"
echo "everywhere). No behavioral change is expected yet -- this only"
echo "verifies the reward channel actually reaches GRPO."
ENV_NAME="${ENV_NAME}" MODEL_PATH="${MODEL_PATH}" SEED="${SEED}" N_GPUS="${N_GPUS}" \
PROJECT_NAME="${PROJECT_NAME}" EXPERIMENT_NAME="smoke_c_rho1" PHASE_NAME="smoke_c" \
TOTAL_EPOCHS=3 SAVE_FREQ=1 TEST_FREQ=1 \
PRH_ENABLED=true PRH_POISON_PROB=1.0 PRH_POISON_SEED=10017 \
bash "${HERE}/run_phase.sh" "$@"

echo "################################################################"
echo "# Gate 1 complete. Manually check for each run:"
echo "#   - training completed without error, checkpoint dir exists"
echo "#   - console/wandb logs show episode/hidden_reward_mean,"
echo "#     episode/observed_reward_mean, episode/proxy_hidden_gap_mean"
echo "#   - Smoke B/C also show prh/poison_prob_nominal,"
echo "#     prh/poison_rate_realized, prh/poison_rate_realized_cumulative"
echo "#     (Smoke B ~ 0.0, Smoke C ~ 1.0)"
echo "# before proceeding to Gate 2 (checkpoint/resume) and Gate 3"
echo "# (40-update clean learnability check)."
echo "################################################################"
