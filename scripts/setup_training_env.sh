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
#   5. vLLM again -- requirements_safety.txt pins torch==2.8.0 explicitly,
#      which breaks vLLM's compiled CUDA extension's ABI (confirmed: it
#      imports fine at the top level but fails deep in the import chain
#      verl's trainer actually uses, at trainer startup). Reinstalling
#      vLLM restores a torch build its extension is actually compiled
#      against, per upstream/README.md's own guidance.
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

echo "=== [1/5] Installing vLLM==${VLLM_VERSION} first (pins torch/CUDA) ==="
pip install "vllm==${VLLM_VERSION}"

echo "=== [2/5] Installing the AI Safety Gridworlds stack (pycolab -> ai-safety-gridworlds -> safe-grid-gym) ==="
GRIDWORLDS_DIR="${UPSTREAM_DIR}/agent_system/environments/env_package/safe_gridworlds/safe-grid-gym"
pip install -e "${GRIDWORLDS_DIR}/ai-safety-gridworlds/pycolab"
pip install -e "${GRIDWORLDS_DIR}/ai-safety-gridworlds"
pip install -e "${GRIDWORLDS_DIR}"

echo "=== [3/5] Installing requirements_safety.txt (may reinstall/adjust torch -- vLLM will be reinstalled after if needed) ==="
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

echo "=== [4/5] Installing upstream package in editable mode ==="
pip install -e "${UPSTREAM_DIR}"

# requirements_safety.txt pins torch==2.8.0 explicitly (not commented out,
# unlike vllm/numpy/flash_attn), which WILL overwrite the torch==2.7.1 that
# vllm==0.10.0 actually needs -- confirmed in practice: vllm imports fine at
# the top level afterwards (lazy submodules), but its compiled CUDA
# extension is now ABI-incompatible (`vllm/_C.abi3.so: undefined symbol:
# _ZN3c104cuda9SetDeviceEa`), which only surfaces once something actually
# triggers the deep import chain (verl's own vllm_utils.py does, at
# trainer startup -- a real GPU Gate 1 run failed on exactly this before
# this step was added). So: always reinstall vLLM after steps 3-4, per
# upstream/README.md's own guidance, rather than just warning about it.
echo "=== [5/5] Reinstalling vLLM==${VLLM_VERSION} (requirements_safety.txt's torch==2.8.0 pin breaks vLLM's compiled CUDA extension otherwise) ==="
pip install "vllm==${VLLM_VERSION}"

echo "=== Verifying the actual import chain verl's trainer uses (not just 'import vllm') ==="
python - <<PY
import torch
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
# Exercises the same deep import chain as verl/utils/vllm_utils.py ->
# vllm.lora.models -> ... -> vllm.platforms.cuda -> vllm._C, which a bare
# "import vllm" does NOT trigger (lazy submodules) and so does not catch
# an ABI mismatch between vllm's compiled extension and whatever torch
# ended up installed.
from vllm.lora.models import LoRAModel  # noqa: F401
import vllm
print("vllm:", vllm.__version__, "-- deep import chain OK")
PY

echo "=== Done: vLLM's compiled CUDA extension verified against the installed torch build. ==="
