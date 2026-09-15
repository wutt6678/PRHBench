#!/bin/bash
# Reproduce PRHBench/upstream from scratch: clone the pinned upstream
# commit, apply the PRH patch series, and install the three bundled
# gridworld packages (pycolab, ai-safety-gridworlds, safe-grid-gym) into
# whatever Python environment is currently active.
#
# Run this from inside the `prhbench` conda environment (see environment.yml):
#   conda env create -f environment.yml && conda activate prhbench
#   bash scripts/setup_upstream.sh
#
# Safe to re-run: it refuses to clobber an existing upstream/ checkout.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${HERE}/upstream"
UPSTREAM_URL="https://github.com/asparius/verl-agent-safety.git"
UPSTREAM_COMMIT="5e20440fde006348141ff238ed0f696b3e1df961"

if [ -d "${UPSTREAM_DIR}" ]; then
  echo "upstream/ already exists at ${UPSTREAM_DIR} -- not re-cloning." >&2
  echo "Remove it first if you want a fresh checkout." >&2
  exit 1
fi

echo "=== Cloning ${UPSTREAM_URL} @ ${UPSTREAM_COMMIT} ==="
git clone "${UPSTREAM_URL}" "${UPSTREAM_DIR}"
git -C "${UPSTREAM_DIR}" checkout "${UPSTREAM_COMMIT}"
git -C "${UPSTREAM_DIR}" checkout -b prh/first-commit

echo "=== Applying PRH patch series ==="
git -C "${UPSTREAM_DIR}" am "${HERE}"/patches/*.patch

echo "=== Installing bundled pycolab / ai-safety-gridworlds / safe-grid-gym ==="
GRIDWORLDS_DIR="${UPSTREAM_DIR}/agent_system/environments/env_package/safe_gridworlds/safe-grid-gym"
pip install -e "${GRIDWORLDS_DIR}/ai-safety-gridworlds/pycolab" --no-deps
pip install -e "${GRIDWORLDS_DIR}/ai-safety-gridworlds" --no-deps
pip install -e "${GRIDWORLDS_DIR}" --no-deps

echo "=== Done. Verify with: ==="
echo "    export PYTHONPATH=${UPSTREAM_DIR}"
echo "    python3 -m unittest discover -s ${HERE}/tests -v"
