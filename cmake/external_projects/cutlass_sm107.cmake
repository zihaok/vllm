# CUTLASS with SM107 (Rubin) support, used by external projects whose pinned
# CUTLASS predates SM107:
#   - FlashMLA `_flashmla_C` (flashmla.cmake): its submodule CUTLASS (147f567)
#     has no SM107, so an sm_107a/sm_107f cubin compiles but traps at runtime
#     (TMEM alloc / tcgen05.ld become CUTE_INVALID_CONTROL_PATH).
#   - DeepGEMM's opt-in sm_107a JIT target of the Rubin MegaMoE kernel
#     (DG_MEGA_MOE_SM107_ARCH=107a), which reads `include/sm107_cutlass`.
# vLLM's own kernels keep CUTLASS_REVISION (CMakeLists.txt).
#
# Sets CUTLASS_SM107_SOURCE_DIR. Override the download with
# VLLM_CUTLASS_SM107_SRC_DIR (environment or cmake variable).
include_guard(GLOBAL)
include(FetchContent)

set(CUTLASS_SM107_REVISION "v4.8.0")

if(DEFINED ENV{VLLM_CUTLASS_SM107_SRC_DIR})
  set(VLLM_CUTLASS_SM107_SRC_DIR $ENV{VLLM_CUTLASS_SM107_SRC_DIR})
endif()

if(VLLM_CUTLASS_SM107_SRC_DIR)
  get_filename_component(CUTLASS_SM107_SOURCE_DIR
    "${VLLM_CUTLASS_SM107_SRC_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
else()
  set(_cutlass_sm107_fc_root "${FETCHCONTENT_BASE_DIR}")
  if(NOT _cutlass_sm107_fc_root)
    set(_cutlass_sm107_fc_root "${CMAKE_BINARY_DIR}/_deps")
  endif()
  set(CUTLASS_SM107_SOURCE_DIR "${_cutlass_sm107_fc_root}/cutlass_sm107-src")
  if(NOT EXISTS "${CUTLASS_SM107_SOURCE_DIR}/include/cutlass/version.h")
    # Populate only: CUTLASS's own CMakeLists.txt must not be loaded.
    FetchContent_Populate(
      cutlass_sm107
      SUBBUILD_DIR "${_cutlass_sm107_fc_root}/cutlass_sm107-subbuild"
      SOURCE_DIR "${CUTLASS_SM107_SOURCE_DIR}"
      BINARY_DIR "${_cutlass_sm107_fc_root}/cutlass_sm107-build"
      GIT_REPOSITORY https://github.com/nvidia/cutlass.git
      GIT_TAG ${CUTLASS_SM107_REVISION}
      GIT_SHALLOW TRUE
      GIT_PROGRESS TRUE
    )
  endif()
endif()

if(NOT EXISTS "${CUTLASS_SM107_SOURCE_DIR}/include/cutlass/version.h")
  message(FATAL_ERROR
    "CUTLASS for SM107 not found at '${CUTLASS_SM107_SOURCE_DIR}'")
endif()
message(STATUS "CUTLASS for SM107 kernels: ${CUTLASS_SM107_SOURCE_DIR}")
