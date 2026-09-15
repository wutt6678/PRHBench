#!/bin/bash
# Install the REAL GRPO training stack into the currently-active Python
# environment (intended for `prhbench-train`, created via
# environment-train.yml -- see that file for why this is kept separate
# from the lightweight `prhbench` smoke-test environment).
#
# Follows upstream/README.md's documented installation order exactly,
# because it matters: vLLM pins its own torch/CUDA stack and will override
# whatever is already installed, so it must go FIRST.
#
#   1. vLLM (pins torch)
#   2. AI Safety Gridworlds stack (pycolab -> ai-safety-gridworlds -> safe-grid-gym)
#   3. requirements_safety.txt (verl/vLLM/transformers pinned set)
#   4. upstream package itself, editable (`pip install -e .`)
#
# Usage:
#   conda env create -f environment-train.yml
#   conda activate prhbench-train
#   bash scripts/setup_training_env.sh          # assumes upstream/ already exists
#
# Run scripts/setup_upstream.sh first if upstream/ doesn't exist yet.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-${HERE}/upstream}"
VLLM_VERSION="${VLLM_VERSION:-0.10.0}"

if [ ! -d "${UPSTREAM_DIR}" ]; then
  echo "upstream/ not found at ${UPSTREAM_DIR}. Run scripts/setup_upstream.sh first." >&2
  exit 1
fi

echo "=== [1/4] Installing vLLM==${VLLM_VERSION} first (pins torch/CUDA) ==="
pip install "vllm==${VLLM_VERSION}"

echo "=== [2/4] Installing the AI Safety Gridworlds stack (pycolab -> ai-safety-gridworlds -> safe-grid-gym) ==="
GRIDWORLDS_DIR="${UPSTREAM_DIR}/agent_system/environments/env_package/safe_gridworlds/safe-grid-gym"
pip install -e "${GRIDWORLDS_DIR}/ai-safety-gridworlds/pycolab"
pip install -e "${GRIDWORLDS_DIR}/ai-safety-gridworlds"
pip install -e "${GRIDWORLDS_DIR}"

echo "=== [3/4] Installing requirements_safety.txt (may reinstall/adjust torch -- vLLM will be reinstalled after if needed) ==="
# requirements_safety.txt is a raw `pip freeze` of the upstream authors' own
# dev environment, so it pins THEIR editable-installed local packages as if
# they were real PyPI releases -- verl==0.3.1.dev0, pycolab==1.2.0.dev0,
# ai_safety_gridworlds==0.1.0, safe_grid_gym==0.1. None of those exact dev
# versions exist on PyPI (confirmed: `pip install verl==0.3.1.dev0` fails
# outright, aborting the whole requirements.txt install since pip resolves
# it as one unit). All four are already installed from local source by step
# 2 (the gridworld packages) and step 4 below (`pip install -e .`, this
# repo's own pyproject.toml declares its package name as "verl"), so filter
# them out here rather than editing the vendored file.
FILTERED_REQS="$(mktemp)"
grep -vE '^(verl|pycolab|ai_safety_gridworlds|safe_grid_gym)==' "${UPSTREAM_DIR}/requirements_safety.txt" > "${FILTERED_REQS}"
pip install -r "${FILTERED_REQS}"
rm -f "${FILTERED_REQS}"

echo "=== [4/4] Installing upstream package in editable mode ==="
pip install -e "${UPSTREAM_DIR}"

echo "=== Verifying vLLM's torch pin survived steps 3-4 ==="
python - <<PY
import torch
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
try:
    import vllm
    print("vllm:", vllm.__version__)
except Exception as exc:
    print("WARNING: vllm import failed after full install:", exc)
    print("Per upstream/README.md, reinstall vllm now: pip install vllm==${VLLM_VERSION}")
PY

echo "=== Done. If torch was downgraded by step 3/4, reinstall vLLM: ==="
echo "    pip install vllm==${VLLM_VERSION}"
