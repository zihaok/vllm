# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOSCA LLN16B (switch ``VLLM_MOSCA_LLN16``, default off).

Above ``LOW_BATCH_MAX_ROWS`` the executor runs ``fused_wqa_wkv`` on the stock
``cute-dsl`` MXFP8 GEMM, because FlashInfer's low-latency block-scaled GEMM
(``cutedsl_low_latency``, the kernel this executor uses up to 8 rows) only
accepts an 8-column token tile. The kernel itself takes any token tile that is
a multiple of 8: here it runs with a 16-column tile for 9-16 rows and a
24-column tile for 17-24 rows, K tile 128, 8 stages, cluster split-K 4, PDL,
compiled once per device outside any capture and launched exactly as
FlashInfer's runner launches it (A = the weight [N, K], B = the MXFP8 input,
SFA / SFB = their swizzled 128x4 scales, C = the output transposed).

Numerics: FP32 accumulation over a 4-way cluster split-K, so the output can
differ from the ``cute-dsl`` GEMM's by a BF16 ulp in rare elements (ulp class,
measured, not assumed); the switch is off by default.
"""

from __future__ import annotations

import os
from functools import cache

import torch
from torch import nn

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation

MOSCA_LLN16 = os.environ.get("VLLM_MOSCA_LLN16", "0") == "1"
MIN_ROWS = 9
MAX_ROWS = 24
TACTIC = (128, 8, 4)  # (cta_k, AB stages, cluster split-K)


def tile_n(rows: int) -> int:
    """The token tile for ``rows`` (a multiple of 8 covering every row)."""
    return 16 if rows <= 16 else 24


def accepts(rows: int) -> bool:
    return MOSCA_LLN16 and MIN_ROWS <= rows <= MAX_ROWS


@cache
def _kernel(device_index: int, cta_n: int):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_ptr
    from flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm import (
        LowLatencyBlockscaledGemmKernel,
    )

    cta_k, stages, split_k = TACTIC
    gemm = LowLatencyBlockscaledGemmKernel(
        acc_dtype=cutlass.Float32,
        mma_tiler_mnk=(128, cta_n, cta_k),
        num_ab_stage=stages,
        num_sfb_tmem_stage=1,
        sf_vec_size=32,
        use_pdl=True,
        split_k=split_k,
    )

    def fake(dtype, align):
        return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)

    with torch.accelerator.device_index(device_index):
        return cute.compile(
            gemm,
            fake(cutlass.Float8E4M3FN, 16),
            fake(cutlass.Float8E8M0FNU, 32),
            fake(cutlass.Float8E4M3FN, 16),
            fake(cutlass.Float8E8M0FNU, 16),
            fake(cutlass.BFloat16, 16),
            None,
            None,
            tuple(cutlass.Int32(v) for v in (128, cta_n, cta_k, 1)),
            make_fake_stream(),
        )


def precompile(device_index: int) -> None:
    """Compile both token tiles now (never inside a CUDA graph capture)."""
    if MOSCA_LLN16:
        for cta_n in (16, 24):
            _kernel(device_index, cta_n)


def lln16_mm(layer: nn.Module, qa: QuantizedActivation) -> torch.Tensor | None:
    """``layer``'s MXFP8 GEMM of ``qa`` (9-24 rows) on the wider-tile low-latency
    kernel, or None when the input is not the fused producer's layout."""
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    data, scale = qa.data, qa.scale
    if (
        data.dim() != 2
        or scale.ndim != 1
        or not data.is_contiguous()
        or qa.orig_dtype != torch.bfloat16
    ):
        return None
    rows, k = data.shape
    if not MIN_ROWS <= rows <= MAX_ROWS:
        return None
    weight, weight_scale = layer.weight, layer.weight_scale  # [K, N] column-major
    n = weight.shape[1]
    if weight.shape[0] != k or k % 128:
        return None
    device_index = data.device.index
    out = torch.empty((rows, n), dtype=qa.orig_dtype, device=data.device)

    def ptr(dtype, tensor, align):
        return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)

    _kernel(device_index, tile_n(rows))(
        ptr(cutlass.Float8E4M3FN, weight, 16),
        ptr(cutlass.Float8E8M0FNU, weight_scale, 32),
        ptr(cutlass.Float8E4M3FN, data, 16),
        ptr(cutlass.Float8E8M0FNU, scale, 16),
        ptr(cutlass.BFloat16, out, 16),
        None,
        None,
        tuple(cutlass.Int32(v) for v in (n, rows, k, 1)),
        cuda_driver.CUstream(torch.cuda.current_stream(data.device).cuda_stream),
    )
    return out.view(*qa.orig_shape[:-1], n)
