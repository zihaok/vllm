# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin (sm_107) MXFP8 dense GEMM backends for FlashInferCutedslMxfp8LinearKernel.

FlashInfer ships a Rubin CuTe-DSL kernel (``dense_blockscaled_gemm_sm107``: K=64
block-scaled UMMA, B-reuse, MMA tiles up to 512 rows) but wires it only into
``mm_fp4``; ``mm_mxfp8(backend="cute-dsl")`` always runs the SM100 kernel
(K=32 UMMA). This module drives the Sm107 kernel for MXFP8 and also exposes
cuBLASLt (``torch._scaled_mm`` with e8m0 block scales). All backends consume the
exact operands of the cute-dsl path: A [M, K] e4m3 row-major with F8_128x4
swizzled ue8m0 scales, B [K, N] e4m3 column-major with swizzled scales; FP32
accumulation, BF16/FP16 output. Only the K summation order differs.

Opt-in through the existing FlashInferCutedslMxfp8LinearKernel knob
``VLLM_MXFP8_FI_LARGE_M_BACKEND=<policy>`` (unset = off, the default):
  - ``rubin``: per-(N, K, M) best of {cute-dsl SM100, cuBLASLt, Sm107 tactic}
    measured on VR200 (``_AUTO_TABLE``).
  - ``sm107``: always an Sm107 tactic (the best one per shape/M from the table).
  - ``cublaslt``: torch._scaled_mm for every shape.
  - ``rubin-cute-dsl``: route through this op but keep FlashInfer's SM100 kernel
    (A/A control for the dispatch).
``VLLM_MXFP8_FI_LARGE_M_THRESHOLD`` (default 0 for these policies) keeps
M below it on cute-dsl. The per-M choice is made inside one opaque custom op
at run time (M is concrete there, never a traced SymInt), so it is CUDA-graph
and torch.compile safe; every tactic a shape can use is compiled on its first
(eager warmup) call, before graph capture.
"""

from __future__ import annotations

import functools
import os
from typing import NamedTuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

RUBIN_POLICIES = ("rubin", "sm107", "cublaslt", "rubin-cute-dsl")
_ENV_BACKEND = os.environ.get("VLLM_MXFP8_FI_LARGE_M_BACKEND", "")
RUBIN_POLICY = _ENV_BACKEND if _ENV_BACKEND in RUBIN_POLICIES else ""
RUBIN_MIN_M = int(os.environ.get("VLLM_MXFP8_FI_LARGE_M_THRESHOLD", "0"))
# Programmatic dependent launch for the Sm107 kernel (see rubin_sm107_kernel).
SM107_PDL = os.environ.get("VLLM_MXFP8_SM107_PDL", "1") == "1"


class Sm107Tactic(NamedTuple):
    """Sm107BlockScaledPersistentDenseGemmKernel configuration (MXF8: tiler K 128,
    instruction K 64). ``tiler_m == 2 * inst_m`` enables B-reuse."""

    tiler_m: int
    tiler_n: int
    inst_m: int
    cluster_m: int
    cluster_n: int
    swap_ab: bool = False
    prefetch: int | None = 0

    def name(self) -> str:
        pf = "x" if self.prefetch is None else str(self.prefetch)
        return (
            f"t{self.tiler_m}x{self.tiler_n}_i{self.inst_m}"
            f"_c{self.cluster_m}x{self.cluster_n}_s{int(self.swap_ab)}_p{pf}"
        )

    @staticmethod
    def parse(s: str) -> Sm107Tactic:
        # t256x256_i256_c2x1_s0_p0
        t, i, c, sw, pf = s.split("_")
        tm, tn = (int(v) for v in t[1:].split("x"))
        cm, cn = (int(v) for v in c[1:].split("x"))
        return Sm107Tactic(
            tm, tn, int(i[1:]), cm, cn, sw == "s1", None if pf == "px" else int(pf[1:])
        )


@functools.cache
def _sm107_kernel_cls():
    from vllm.model_executor.kernels.linear.mxfp8.rubin_sm107_kernel import (
        Sm107Mxfp8GemmKernel,
    )

    return Sm107Mxfp8GemmKernel


def sm107_can_implement(t: Sm107Tactic, m: int, n: int, k: int) -> bool:
    import cutlass

    cls = _sm107_kernel_cls()
    kernel_m, kernel_n = (n, m) if t.swap_ab else (m, n)
    return cls._can_implement_impl(
        (kernel_m, kernel_n, k, 1),
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        cutlass.BFloat16,
        "k",
        "k",
        "m" if t.swap_ab else "n",
        32,
        (t.tiler_m, t.tiler_n, 128),
        (t.inst_m, t.tiler_n, 64),
        (t.cluster_m, t.cluster_n),
    )


_SM107_KERNELS: dict[tuple, object] = {}


def _sm107_compiled(
    t: Sm107Tactic,
    out_dtype: torch.dtype,
    device_index: int,
    pdl: bool | None = None,
):
    if pdl is None:
        pdl = SM107_PDL
    key = (t, out_dtype, device_index, pdl)
    compiled = _SM107_KERNELS.get(key)
    if compiled is not None:
        return compiled

    import cutlass
    from flashinfer.cute_dsl.utils import (
        get_max_active_clusters,
        torch_to_cutlass_dtype,
    )
    from flashinfer.gemm.gemm_mm_fp4_cute_dsl import (
        _make_blockscaled_gemm_compile_fn,
    )
    from flashinfer.gemm.kernels import (
        dense_blockscaled_gemm_sm100,
        dense_blockscaled_gemm_sm107,
    )
    from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    from vllm.model_executor.kernels.linear.mxfp8 import rubin_sm107_kernel

    gemm = _sm107_kernel_cls()(
        32,
        (t.inst_m, t.tiler_n, 64),
        (t.tiler_m, t.tiler_n, 128),
        (t.cluster_m, t.cluster_n),
        prefetch_dist=t.prefetch,
        enable_pdl=pdl,
    )
    # One CTA per SM (the kernels use nearly all shared memory). The DSL probe
    # falls back to sm_count when no driver context is current, which would
    # oversubscribe a persistent grid; clamp it.
    cluster_size = t.cluster_m * t.cluster_n
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    mac = min(get_max_active_clusters(cluster_size), sm_count // cluster_size)
    compile_fn = _make_blockscaled_gemm_compile_fn(
        gemm,
        ab_cutlass_dtype=cutlass.Float8E4M3FN,
        sf_dtype=cutlass.Float8E8M0FNU,
        c_cutlass_dtype=torch_to_cutlass_dtype(out_dtype),
        ab_assumed_align=16,
        swap_ab=t.swap_ab,
        sf_m=1,
        sf_n=1,
        sf_k=1,
        batch_size=1,
        max_active_clusters=mac,
    )
    dtype = str(out_dtype).removeprefix("torch.")
    compiled = build_and_load_cute_dsl_kernel(
        "vllm_mxfp8_sm107",
        f"{t.name()}_{dtype}_mac{mac}_pdl{int(pdl)}",
        compile_fn,
        extra_key_files=(
            __file__,
            rubin_sm107_kernel.__file__,
            dense_blockscaled_gemm_sm100.__file__,
            dense_blockscaled_gemm_sm107.__file__,
        ),
    )
    _SM107_KERNELS[key] = compiled
    return compiled


_ALPHA_ONE: dict[torch.device, torch.Tensor] = {}


def sm107_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype,
    t: Sm107Tactic,
    out: torch.Tensor | None = None,
    pdl: bool | None = None,
) -> torch.Tensor:
    """out[M, N] = (A * sfA) @ (B * sfB); a [M, K] row-major, b [K, N] col-major."""
    m, k = a.shape
    n = b.shape[1]
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    if m == 0:
        return out
    compiled = _sm107_compiled(t, out_dtype, a.device.index, pdl)
    if t.swap_ab:
        kernel_m, kernel_n = n, m
        ka, kb, ksfa, ksfb = b.T, a, b_scale, a_scale
        launch_out = out.as_strided(out.shape, (1, out.shape[0]))
    else:
        kernel_m, kernel_n = m, n
        ka, kb, ksfa, ksfb = a, b.T, a_scale, b_scale
        launch_out = out
    alpha = _ALPHA_ONE.get(a.device)
    if alpha is None:
        alpha = torch.ones(1, dtype=torch.float32, device=a.device)
        _ALPHA_ONE[a.device] = alpha
    compiled(
        ka,
        kb,
        launch_out,
        (kernel_m + 127) // 128,
        (kernel_n + 127) // 128,
        (k // 32 + 3) // 4,
        ksfa.data_ptr(),
        ksfb.data_ptr(),
        alpha,
    )
    return out


def cublaslt_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch._scaled_mm(
        a,
        b,
        scale_a=a_scale.view(torch.float8_e8m0fnu),
        scale_b=b_scale.view(torch.float8_e8m0fnu),
        out_dtype=out_dtype,
    )


# Per-(N, K) backend choice by M: list of (min_m, choice) sorted by min_m; the
# last entry with min_m <= M wins (M below the first entry stays on cute-dsl).
# Choices: "cute-dsl", "cublaslt", or an Sm107Tactic name. Measured on VR200
# (hecate, memclk 4752) at the DSV4.1 C2048 rank-0 token counts by
# kernels/dense/bench/bench_sm107.py + make_table.py (>= 3% faster than the
# autotuned cute-dsl kernel to switch). Shapes: wqa_wkv (1792, 5120), idx_wq_b
# (4096, 1280), wo_b (5120, 8192), dspark_in (5120, 15360), engram_wkv
# (25600, 6144), wq_b (32768, 1280).
_AUTO_TABLE: dict[tuple[int, int], list[tuple[int, str]]] = {
    (1792, 5120): [
        (1536, "cublaslt"),
        (1800, "cublaslt"),
        (2046, "t256x192_i256_c2x1_s0_p0"),
        (2280, "t256x192_i256_c2x2_s0_p0"),
        (2520, "t256x192_i256_c2x1_s0_p0"),
        (3072, "cublaslt"),
    ],
    (4096, 1280): [
        (1536, "cublaslt"),
        (1800, "cublaslt"),
        (2046, "cublaslt"),
        (2280, "cute-dsl"),
        (2520, "t256x256_i256_c2x2_s0_p0"),
        (3072, "t256x256_i256_c2x1_s0_p0"),
    ],
    (5120, 8192): [
        (1536, "t512x256_i256_c2x1_s0_p0"),
        (1800, "t512x256_i256_c2x1_s0_p0"),
        (2046, "t512x256_i256_c2x1_s0_p0"),
        (2280, "t512x256_i256_c2x1_s0_p0"),
        (2520, "t512x256_i256_c2x1_s0_p0"),
        (3072, "t256x256_i256_c4x1_s0_p0"),
    ],
    (5120, 15360): [
        (1536, "t512x256_i256_c2x1_s0_p0"),
        (1800, "t512x256_i256_c2x1_s0_p0"),
        (2046, "t512x256_i256_c2x1_s0_p0"),
        (2280, "cublaslt"),
        (2520, "t512x256_i256_c2x1_s0_p0"),
        (3072, "t256x256_i256_c4x1_s0_p0"),
    ],
    (25600, 6144): [
        (1536, "t512x256_i256_c2x1_s0_p0"),
        (1800, "t512x256_i256_c2x1_s0_p0"),
        (2046, "t512x256_i256_c2x1_s0_p0"),
        (2280, "cublaslt"),
        (2520, "t512x256_i256_c2x1_s0_p0"),
        (3072, "t512x256_i256_c2x1_s0_p0"),
    ],
    (32768, 1280): [
        (1536, "cublaslt"),
        (1800, "t256x256_i256_c2x1_s0_p0"),
        (2046, "t256x256_i256_c2x1_s0_p0"),
        (2280, "t256x256_i256_c2x1_s0_p0"),
        (2520, "cublaslt"),
        (3072, "cublaslt"),
    ],
}
_SM107_TABLE: dict[tuple[int, int], list[tuple[int, str]]] = {
    (1792, 5120): [
        (1536, "t256x128_i256_c2x1_s0_p0"),
        (1800, "t256x128_i256_c2x1_s1_p0"),
        (2046, "t256x192_i256_c2x1_s0_p0"),
        (2280, "t256x192_i256_c2x2_s0_p0"),
        (2520, "t256x192_i256_c2x1_s0_p0"),
        (3072, "t256x256_i256_c2x1_s0_p0"),
    ],
    (4096, 1280): [
        (1536, "t256x256_i256_c2x1_s0_p0"),
        (1800, "t256x192_i256_c2x1_s0_p0"),
        (2046, "t256x192_i256_c2x2_s0_p0"),
        (2280, "t256x192_i256_c2x1_s0_p0"),
        (2520, "t256x256_i256_c2x2_s0_p0"),
        (3072, "t256x256_i256_c2x1_s0_p0"),
    ],
    (5120, 8192): [
        (1536, "t512x256_i256_c2x1_s0_p0"),
        (1800, "t512x256_i256_c2x1_s0_p0"),
        (2046, "t512x256_i256_c2x1_s0_p0"),
        (2280, "t512x256_i256_c2x1_s0_p0"),
        (2520, "t512x256_i256_c2x1_s0_p0"),
        (3072, "t256x256_i256_c4x1_s0_p0"),
    ],
    (5120, 15360): [
        (1536, "t512x256_i256_c2x1_s0_p0"),
        (1800, "t512x256_i256_c2x1_s0_p0"),
        (2046, "t512x256_i256_c2x1_s0_p0"),
        (2280, "t512x256_i256_c2x1_s0_p0"),
        (2520, "t512x256_i256_c2x1_s0_p0"),
        (3072, "t256x256_i256_c4x1_s0_p0"),
    ],
    (25600, 6144): [
        (1536, "t512x256_i256_c2x1_s0_p0"),
        (1800, "t512x256_i256_c2x1_s0_p0"),
        (2046, "t512x256_i256_c2x1_s0_p0"),
        (2280, "t512x256_i256_c2x1_s0_p0"),
        (2520, "t512x256_i256_c2x1_s0_p0"),
        (3072, "t512x256_i256_c2x1_s0_p0"),
    ],
    (32768, 1280): [
        (1536, "t256x192_i256_c2x1_s0_p0"),
        (1800, "t256x256_i256_c2x1_s0_p0"),
        (2046, "t256x256_i256_c2x1_s0_p0"),
        (2280, "t256x256_i256_c2x1_s0_p0"),
        (2520, "t256x256_i256_c2x2_s0_p0"),
        (3072, "t256x256_i256_c2x1_s1_p0"),
    ],
}


def _lookup(table: dict, m: int, n: int, k: int) -> str:
    entries = table.get((n, k))
    if not entries:
        return "cute-dsl"
    choice = "cute-dsl"
    for min_m, c in entries:
        if m >= min_m:
            choice = c
    return choice


@functools.cache
def select_backend(m: int, n: int, k: int, policy: str) -> str:
    if m < RUBIN_MIN_M or policy == "rubin-cute-dsl":
        choice = "cute-dsl"
    elif policy == "rubin":
        choice = _lookup(_AUTO_TABLE, m, n, k)
    elif policy == "sm107":
        choice = _lookup(_SM107_TABLE, m, n, k)
    elif policy == "cublaslt":
        choice = "cublaslt"
    else:
        raise ValueError(f"Unknown Rubin MXFP8 policy {policy!r}")
    if choice not in ("cute-dsl", "cublaslt"):
        t = Sm107Tactic.parse(choice)
        if not sm107_can_implement(t, m, n, k):
            choice = "cute-dsl"
    return choice


_WARMED: set[tuple[int, int, torch.dtype]] = set()


@functools.cache
def _prime_cute_dsl_alpha(device_index: int) -> None:
    """FlashInfer's cute-dsl mm_mxfp8 allocates its alpha=1 tensor lazily per
    device (a host-to-device copy). When the eager warmup routes every M of a
    shape to cuBLASLt/Sm107, the first cute-dsl call (small M) would otherwise
    land inside CUDA-graph capture and fail."""
    try:
        from flashinfer.gemm.gemm_mm_fp4_cute_dsl import _prepare_alpha_for_launch
    except ImportError:
        return
    _prepare_alpha_for_launch(None, torch.device("cuda", device_index))


def _warm_shape(n: int, k: int, out_dtype: torch.dtype, device_index: int) -> None:
    """Compile (or load from the on-disk cache) every Sm107 tactic the tables
    list for this (N, K) on its first call, which is an eager warmup run, so no
    CuTe-DSL compile/module load happens during CUDA-graph capture."""
    key = (n, k, out_dtype)
    if key in _WARMED:
        return
    _WARMED.add(key)
    _prime_cute_dsl_alpha(device_index)
    for table in (_AUTO_TABLE, _SM107_TABLE):
        for _, choice in table.get((n, k), ()):
            if choice not in ("cute-dsl", "cublaslt"):
                _sm107_compiled(Sm107Tactic.parse(choice), out_dtype, device_index)


def _mm_mxfp8_rubin(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    m, k = A.shape
    n = B.shape[1]
    _warm_shape(n, k, out_dtype, A.device.index)
    choice = select_backend(m, n, k, RUBIN_POLICY)
    if choice == "cute-dsl":
        from vllm.utils import flashinfer as vllm_flashinfer

        return vllm_flashinfer.mm_mxfp8(
            A, B, A_scale, B_scale, out_dtype=out_dtype, backend="cute-dsl"
        )
    if choice == "cublaslt":
        return cublaslt_mm_mxfp8(A, B, A_scale, B_scale, out_dtype)
    return sm107_mm_mxfp8(A, B, A_scale, B_scale, out_dtype, Sm107Tactic.parse(choice))


def _mm_mxfp8_rubin_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(A.shape[0], B.shape[1], dtype=out_dtype, device=A.device)


if RUBIN_POLICY:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="mm_mxfp8_rubin",
        op_func=_mm_mxfp8_rubin,
        fake_impl=_mm_mxfp8_rubin_fake,
    )


def mm_mxfp8_rubin(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ops.vllm.mm_mxfp8_rubin(A, B, A_scale, B_scale, out_dtype)
