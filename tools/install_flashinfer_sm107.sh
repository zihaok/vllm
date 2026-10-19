#!/usr/bin/env bash
# Install the pinned Python/CuTe DSL port without replacing vLLM's native binaries.
set -euo pipefail

FLASHINFER_REVISION=d172bf90d3ff8bc69ea5216211630de84cb64e0f
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-$(command -v python3)}
FLASHINFER_BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/flashinfer-sm107.XXXXXX")
trap 'rm -rf "$FLASHINFER_BUILD_DIR"' EXIT

git init "$FLASHINFER_BUILD_DIR"
git -C "$FLASHINFER_BUILD_DIR" remote add origin https://github.com/zihaok/flashinfer.git
git -C "$FLASHINFER_BUILD_DIR" fetch --depth=1 origin "$FLASHINFER_REVISION"
git -C "$FLASHINFER_BUILD_DIR" checkout --detach FETCH_HEAD
test "$(git -C "$FLASHINFER_BUILD_DIR" rev-parse HEAD)" = "$FLASHINFER_REVISION"
git -C "$FLASHINFER_BUILD_DIR" submodule update --init --depth=1 \
    3rdparty/cutlass 3rdparty/spdlog 3rdparty/cccl

# The direct MegaMoE API uses Torch symmetric memory. It needs no NIXL/NCCL-EP build.
BUILD_NIXL_EP=0 BUILD_NCCL_EP=0 FLASHINFER_BUILD_NO_PIP=1 \
    uv pip install --system --python "$PYTHON_EXECUTABLE" --no-deps --no-build-isolation \
    "$FLASHINFER_BUILD_DIR"
"$PYTHON_EXECUTABLE" "$(dirname "${BASH_SOURCE[0]}")/patch_quack_cutedsl.py"
"$PYTHON_EXECUTABLE" -c \
    'import flashinfer.mega_moe_deepgemm; import cutlass.cute; import tvm_ffi; from cuda.bindings import driver'
echo "Installed FlashInfer SM107 MegaMoE from $FLASHINFER_REVISION"
