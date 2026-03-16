#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a local HipKittens checkout for this repo.
# Safe to re-run; pass --update to pull latest main.

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/../.." && pwd)"
THIRD_PARTY_DIR="${REPO_ROOT}/third_party"
HK_DIR="${THIRD_PARTY_DIR}/HipKittens"
HK_REPO="https://github.com/HazyResearch/HipKittens"

UPDATE=0
if [[ "${1:-}" == "--update" ]]; then
  UPDATE=1
fi

mkdir -p "${THIRD_PARTY_DIR}"

if [[ ! -d "${HK_DIR}/.git" ]]; then
  echo "[setup] Cloning HipKittens into ${HK_DIR}"
  git clone --depth 1 "${HK_REPO}" "${HK_DIR}"
elif [[ "${UPDATE}" -eq 1 ]]; then
  echo "[setup] Updating existing HipKittens checkout"
  git -C "${HK_DIR}" fetch --depth 1 origin main
  git -C "${HK_DIR}" checkout -q main
  git -C "${HK_DIR}" reset --hard -q origin/main
else
  echo "[setup] Using existing HipKittens checkout at ${HK_DIR}"
fi

required_paths=(
  "${HK_DIR}/include/kittens.cuh"
  "${HK_DIR}/include/ops/warp/register/tile/mma.cuh"
  "${HK_DIR}/kernels/torch_scaled/scaled_matmul.cu"
)

for p in "${required_paths[@]}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[error] Missing expected HipKittens file: ${p}" >&2
    exit 1
  fi
done

echo "[ok] HipKittens checkout verified."
echo
echo "Export these for local experiments:"
echo "  export HIPKITTENS_ROOT='${HK_DIR}'"
echo "  export HIPKITTENS_INCLUDE='${HK_DIR}/include'"
echo

if command -v hipcc >/dev/null 2>&1; then
  echo "[check] hipcc found. Running a tiny include smoke compile."
  TMP_CU="${REPO_ROOT}/.tmp_hk_smoke.cu"
  TMP_O="${REPO_ROOT}/.tmp_hk_smoke.o"
  cat > "${TMP_CU}" <<'EOF'
#include "kittens.cuh"
__global__ void hk_smoke() {}
int main() { return 0; }
EOF
  hipcc -std=c++20 --offload-arch=gfx950 -I"${HK_DIR}/include" -c "${TMP_CU}" -o "${TMP_O}"
  rm -f "${TMP_CU}" "${TMP_O}"
  echo "[ok] HipKittens headers compile with local hipcc."
else
  echo "[info] hipcc not found locally; skipping compile smoke test."
  echo "[info] Use an ROCm container or MI355X runner for compilation/runtime checks."
fi
